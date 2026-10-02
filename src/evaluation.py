"""Supplementary DAY 2 checks. External labels never select a new model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from features import FEATURE_SETS, feature_matrix

ORDER = {"Ridge": 0, "ElasticNet": 1, "RandomForest": 2, "GradientBoosting": 3}


def predict(spec, fitting, evaluation):
    fitted = clone(spec["estimator"]).fit(feature_matrix(fitting, spec["features"]), fitting.cycle_life)
    return np.maximum(fitted.predict(feature_matrix(evaluation, spec["features"])), 1.0)


def mape(actual, predicted):
    return float(np.mean(100 * np.abs(np.asarray(predicted) - np.asarray(actual)) / np.asarray(actual)))


def nested_group_cv(train, specs):
    """Outer folds are frozen; all candidate selection uses inner groups only."""
    folds, predictions, candidates, audits = [], [], [], []
    for outer in range(1, 6):
        mask = train.cv_fold.eq(outer)
        fitting = train.loc[~mask].reset_index(drop=True)
        evaluation = train.loc[mask].reset_index(drop=True)
        inner_splits = list(GroupKFold(n_splits=4).split(fitting, groups=fitting.policy_group))
        for inner, (fit_idx, valid_idx) in enumerate(inner_splits, 1):
            a, b = fitting.iloc[fit_idx], fitting.iloc[valid_idx]
            overlap = len(set(a.policy_group) & set(b.policy_group))
            external_overlap = len(set(fitting.cell_id) & set(evaluation.cell_id))
            if overlap or external_overlap:
                raise ValueError("Nested CV partitions overlap")
            audits.append({"outer_fold": outer, "inner_fold": inner,
                "fit_cell_ids": ";".join(a.cell_id), "valid_cell_ids": ";".join(b.cell_id),
                "outer_valid_cell_ids": ";".join(evaluation.cell_id),
                "policy_overlap": overlap, "outer_valid_overlap": external_overlap})
        ranked = []
        for spec in specs:
            losses = [mape(fitting.iloc[v].cycle_life, predict(spec, fitting.iloc[t], fitting.iloc[v])) for t, v in inner_splits]
            row = {"outer_fold": outer, "candidate": spec["id"], "feature_set": spec["feature_set"],
                "model": spec["model"], "inner_mape_pct": np.mean(losses), "inner_sd_pct": np.std(losses, ddof=1),
                "feature_count": len(spec["features"]), "model_order": ORDER[spec["model"]]}
            ranked.append(row)
        ranking = pd.DataFrame(ranked).sort_values(["inner_mape_pct", "inner_sd_pct", "feature_count", "model_order", "candidate"])
        chosen = ranking.iloc[0]
        candidates.extend(ranking.to_dict("records"))
        spec = next(s for s in specs if s["id"] == chosen.candidate)
        p = predict(spec, fitting, evaluation)
        baseline = np.full(len(evaluation), fitting.cycle_life.median())
        folds.append({"outer_fold": outer, "train_n": len(fitting), "valid_n": len(evaluation),
            "chosen_candidate": spec["id"], "chosen_feature_set": spec["feature_set"],
            "inner_selection_mape_pct": chosen.inner_mape_pct,
            "outer_mape_pct": mape(evaluation.cycle_life, p), "baseline_mape_pct": mape(evaluation.cycle_life, baseline)})
        for row, value in zip(evaluation.itertuples(), p):
            predictions.append({"cell_id": row.cell_id, "policy_group": row.policy_group, "outer_fold": outer,
                "actual": row.cycle_life, "predicted": value, "chosen_candidate": spec["id"]})
        print(f"Nested CV outer {outer}/5: {spec['id']}", flush=True)
    return tuple(pd.DataFrame(x) for x in (folds, predictions, candidates, audits))


def fixed_model_ablation(train, specs, winner):
    """Hold model, hyperparameters and target scale constant while changing X."""
    rows = []
    for group in FEATURE_SETS:
        spec = next(s for s in specs if s["feature_set"] == group and
            all(s[k] == winner[k] for k in ("model", "settings", "target_scale")))
        for fold in range(1, 6):
            v = train.cv_fold.eq(fold)
            rows.append({"feature_set": group, "fold": fold,
                "mape_pct": mape(train.loc[v].cycle_life, predict(spec, train.loc[~v], train.loc[v]))})
    detail = pd.DataFrame(rows)
    base = detail.loc[detail.feature_set == "G0_delta"].set_index("fold").mape_pct
    detail["delta_vs_g0_pctp"] = [r.mape_pct - base.loc[r.fold] for r in detail.itertuples()]
    summary = detail.groupby("feature_set", sort=False).agg(cv_mape_pct=("mape_pct", "mean"),
        cv_sd_pct=("mape_pct", "std"), delta_vs_g0_pctp=("delta_vs_g0_pctp", "mean"),
        folds_improved=("delta_vs_g0_pctp", lambda x: int((x < 0).sum()))).reset_index()
    return detail, summary


def baseline_comparison(train, holdout, external, pred):
    rows, paired = [], None
    for phase, data, fitting in [("Valid", holdout, train)] + [(k, v, pd.concat([train, holdout])) for k, v in external.items()]:
        p = pred.loc[pred.phase == phase]
        baseline = float(fitting.cycle_life.median())
        baseline_errors = 100 * abs(p.actual - baseline) / p.actual
        rows.append({"phase": phase, "n": len(p), "baseline_median_cycles": baseline,
            "baseline_mape_pct": baseline_errors.mean(), "model_mape_pct": p.absolute_percentage_error.mean(),
            "improvement_pctp": baseline_errors.mean() - p.absolute_percentage_error.mean()})
        if phase == "Batch 2":
            paired = pd.DataFrame({"cell_id": p.cell_id, "policy_group": p.policy_group,
                "model_ape_pct": p.absolute_percentage_error, "baseline_ape_pct": baseline_errors})
    groups = [g for _, g in paired.groupby("policy_group")]
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(2000):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        diffs.append((sample.baseline_ape_pct - sample.model_ape_pct).mean())
    return pd.DataFrame(rows), paired, list(np.percentile(diffs, [2.5, 97.5]))


def run_diagnostics(root, train, holdout, external, specs, winner, pred):
    root = Path(root); results = root / "results"
    folds, nested_pred, inner_candidates, audit = nested_group_cv(train, specs)
    for frame, filename in [(folds,"day2_nested_folds.csv"),(nested_pred,"day2_nested_predictions.csv"),
        (inner_candidates,"day2_nested_candidates.csv"),(audit,"day2_nested_split_audit.csv")]:
        frame.to_csv(results / filename, index=False)
    ablation_detail, ablation = fixed_model_ablation(train, specs, winner)
    ablation_detail.to_csv(results / "day2_ablation_folds.csv", index=False)
    ablation.to_csv(results / "day2_ablation_summary.csv", index=False)
    baseline, paired, baseline_ci = baseline_comparison(train, holdout, external, pred)
    baseline.to_csv(results / "day2_baseline_comparison.csv", index=False)
    paired.to_csv(results / "day2_baseline_paired_errors.csv", index=False)
    full = pd.concat([train, holdout], ignore_index=True)
    refit_rows = []
    for batch, frame in external.items():
        train_only = predict(winner, train, frame)
        saved = pred.loc[pred.phase.eq(batch)].set_index("cell_id").loc[frame.cell_id, "predicted"].to_numpy()
        refit_rows.append({"batch":batch,"train29_mape_pct":mape(frame.cycle_life,train_only),
            "refit36_mape_pct":mape(frame.cycle_life,saved),
            "refit_delta_pctp":mape(frame.cycle_life,saved)-mape(frame.cycle_life,train_only)})
    refit = pd.DataFrame(refit_rows)
    refit.to_csv(results / "day2_refit_sensitivity.csv", index=False)
    names = list(dict.fromkeys(sum(FEATURE_SETS.values(), [])))
    drift = []
    for batch, frame in external.items():
        for feature in names:
            ref = full[feature]; values = frame[feature]
            drift.append({"batch": batch, "feature": feature, "train_mean": ref.mean(), "external_mean": values.mean(),
                "standardized_mean_difference": (values.mean()-ref.mean())/ref.std(ddof=1),
                "outside_train_range_n": int(((values < ref.min()) | (values > ref.max())).sum()),
                "n_observed": int(values.notna().sum()), "missing_n": int(values.isna().sum())})
    drift = pd.DataFrame(drift); drift.to_csv(results / "day2_input_drift.csv", index=False)
    enriched = pd.concat(list(external.values())).merge(pred[pred.phase.isin(external)][["cell_id","predicted","absolute_percentage_error","error_cycles"]], on="cell_id", validate="one_to_one")
    enriched["policy_seen_in_batch1"] = enriched.policy_group.isin(set(full.policy_group))
    policies = enriched.groupby(["batch","policy_seen_in_batch1"]).agg(n=("cell_id","size"),
        mape_pct=("absolute_percentage_error","mean"), bias_cycles=("error_cycles","mean")).reset_index()
    policies.to_csv(results / "day2_policy_transfer.csv", index=False)
    quality = pd.read_csv(root / "data/processed/cell_features.csv")
    quality_rows = []
    for batch, frame in quality.groupby("batch"):
        valid = frame.cycle_life.notna() & (frame.cycle_life > 0)
        quality_rows.append({"batch":batch,"all_n":len(frame),"missing_or_invalid_label_n":int((~valid).sum()),
            "valid_label_eol_uncertain_n":int((valid & frame.eol_not_observed).sum()),
            "main_cohort_n":int((valid & ~frame.eol_not_observed).sum())})
    quality_audit = pd.DataFrame(quality_rows)
    quality_audit.to_csv(results / "day2_label_audit.csv", index=False)
    excluded = quality.loc[quality.batch.eq("Batch 1") & quality.cycle_life.notna() & quality.eol_not_observed].copy()
    excluded["predicted"] = predict(winner, full, excluded)
    excluded["ape_pct"] = 100 * abs(excluded.predicted-excluded.cycle_life)/excluded.cycle_life
    excluded[["cell_id","cycle_life","qd_last_valid","predicted","ape_pct"]].to_csv(results / "day2_excluded_label_sensitivity.csv", index=False)
    risks = []
    for phase in ("Valid","Batch 2","Batch 3"):
        p = pred.loc[pred.phase.eq(phase)]
        true_short = p.actual < 500; predicted_short = p.predicted < 500
        fn = int((true_short & ~predicted_short).sum()); fp = int((~true_short & predicted_short).sum())
        risks.append({"phase":phase,"n":len(p),"actual_short_n":int(true_short.sum()),
            "short_detected_n":int((true_short & predicted_short).sum()),"short_missed_n":fn,
            "false_short_n":fp,"short_missed_pct":100*fn/true_short.sum() if true_short.any() else np.nan})
    risk = pd.DataFrame(risks); risk.to_csv(results / "day2_short_life_risk.csv", index=False)
    fitted = clone(winner["estimator"]).fit(feature_matrix(full,winner["features"]),full.cycle_life)
    import joblib
    from importlib.metadata import version
    joblib.dump(fitted, results / "day2_model.joblib")
    manifest = {"features":winner["features"],"candidate_id":winner["id"],"target":"cycle_life",
        "training_cell_ids":list(full.cell_id),"training_n":len(full),"prediction_min_cycles":1,
        "sklearn_version":version("scikit-learn"),"joblib_version":version("joblib"),
        "observation_max_cycle":100,"use":"research; not an ESS safety or warranty decision"}
    if winner["target_scale"] == "raw" and winner["model"] in ("Ridge","ElasticNet"):
        scaler = fitted.named_steps["scale"]; linear = fitted.named_steps["model"]
        coefficients = linear.coef_/scaler.scale_
        intercept = float(linear.intercept_ - np.sum(coefficients*scaler.mean_))
        manifest.update(raw_input_intercept_cycles=intercept,raw_input_coefficients=dict(zip(winner["features"],map(float,coefficients))))
    (results / "day2_model_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    fig, ax = plt.subplots(1,2,figsize=(11,4),layout="constrained")
    ax[0].bar(ablation.feature_set,ablation.cv_mape_pct,color="#2878b5")
    ax[0].errorbar(range(len(ablation)),ablation.cv_mape_pct,yerr=ablation.cv_sd_pct,fmt="none",ecolor="#333",capsize=3)
    ax[0].set(title="Same Ridge recipe: feature ablation",ylabel="CV MAPE (%)")
    ax[0].tick_params(axis="x",rotation=25)
    ax[1].plot(folds.outer_fold,folds.outer_mape_pct,"o-",label="Nested selection")
    ax[1].plot(folds.outer_fold,folds.baseline_mape_pct,"o--",label="Median baseline")
    ax[1].set(title="Outer validation: selection repeated inside",xlabel="Frozen outer fold",ylabel="MAPE (%)")
    ax[1].legend(fontsize=8)
    fig.savefig(results / "day2_validation_and_ablation.png",dpi=180);plt.close(fig)
    summary = {"nested_mape_mean_pct":float(folds.outer_mape_pct.mean()),"nested_mape_sd_pct":float(folds.outer_mape_pct.std()),
        "nested_oof_mape_pct":mape(nested_pred.actual,nested_pred.predicted),
        "nested_baseline_mean_pct":float(folds.baseline_mape_pct.mean()),
        "nested_group_choices":folds.chosen_feature_set.value_counts().to_dict(),
        "batch2_baseline_improvement_ci95_pctp":list(map(float,baseline_ci)),
        "fixed_winner_unchanged":winner["id"],"external_results_used_for_reselection":False}
    (results / "day2_validation_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    lines = ["", "## 7. 모델 선택 편향과 피처 추가 효과", "",
        "### 선택 과정을 포함한 중첩 교차검증", "",
        "Batch 1 Train 29셀만 사용했다. 고정된 5개 outer fold 각각에서 남은 정책 그룹을 GroupKFold 4개 inner fold로 나눠 80개 후보를 다시 선택하고, 선택에 쓰이지 않은 outer fold에서 평가했다. 모든 inner/outer 경계의 셀·정책 그룹 중복을 검사했다. Hold-out·Batch 2·3는 이 과정에 들어가지 않는다.", "",
        f"중첩 CV outer MAPE는 **{summary['nested_mape_mean_pct']:.2f} ± {summary['nested_mape_sd_pct']:.2f}%**, 전체 outer 예측을 모은 OOF MAPE는 **{summary['nested_oof_mape_pct']:.2f}%**다. 동일 outer fold의 중앙값 기준은 {summary['nested_baseline_mean_pct']:.2f}%다. 기존 7.93%는 후보를 고를 때 사용한 CV 점수이고, 중첩 결과는 후보 선택을 반복하는 절차에 대한 보조 추정이다. 평균 fold MAPE와 전체 OOF MAPE는 fold 크기가 달라 같지 않을 수 있다.", "",
        "| Outer fold | 선택 피처 | 선택 모델 | Outer MAPE (%) | 중앙값 MAPE (%) |", "| --- | --- | --- | ---: | ---: |"]
    for r in folds.itertuples():
        lines.append(f"| {r.outer_fold} | {r.chosen_feature_set} | {r.chosen_candidate.split('__')[1]} | {r.outer_mape_pct:.2f} | {r.baseline_mape_pct:.2f} |")
    lines += ["", "Outer fold 3에서는 inner MAPE 6.92%로 G3의 로그 타깃 Ridge가 선택됐지만 outer MAPE는 36.74%였다. 평균 온도까지 더한 후보가 작은 학습 집단에서 선택된 뒤 다른 정책군에서 실패하는 구체적인 예다. 최저 CV 점수 하나를 실제 운영 성능으로 해석하기 어렵다. 중첩 CV도 5~6셀씩의 검증으로 불안정하며, DAY 1의 전체 Train EDA·피처 설계까지 중첩한 것은 아니다. 따라서 완전한 독립 검증이나 모집단 성능 보증으로 해석하지 않는다. 이번 추가 검증 결과로 외부 결과를 다시 튜닝하지 않았다.", "",
        "### 같은 모델에서 피처만 변경한 비교", "",
        f"모델·설정·타깃을 {winner['model']} / {winner['settings']} / {winner['target_scale']}로 고정하고 같은 CV fold에서 입력만 바꿨다. Δ는 해당 묶음 MAPE − G0 MAPE이며 음수일 때 개선이다. 서로 다른 최적 모델을 비교하는 것보다 변수 추가 효과를 분리해서 확인할 수 있다.", "",
        "| 피처 묶음 | CV MAPE (%) | G0 대비 Δ (%p) | 개선된 fold / 5 |", "| --- | ---: | ---: | ---: |"]
    for r in ablation.itertuples():
        lines.append(f"| {r.feature_set} | {r.cv_mape_pct:.2f} ± {r.cv_sd_pct:.2f} | {r.delta_vs_g0_pctp:+.2f} | {r.folds_improved}/5 |")
    lines += ["", "작은 표본에서 추가 피처가 도움이 되는지는 fold마다 달라질 수 있다. 이 표와 저장된 fold별 값을 근거로 G0의 간결함을 판단하며, 낮은 상관이나 CV 악화만으로 충전·온도가 물리적으로 중요하지 않다고 결론 내리지 않는다.", "",
        "![중첩 검증과 피처 비교](../results/day2_validation_and_ablation.png)", "",
        "## 8. 외부 성능 차이의 추가 근거", "",
        "### 같은 학습 데이터로 만든 중앙값 기준과 비교", "",
        "Valid의 기준값은 Train 29셀의 중앙값, 외부 배치의 기준값은 재학습에 사용한 Batch 1 전체 36셀의 중앙값이다. 외부 실제 수명으로 기준값을 정하지 않았다.", "",
        "| 평가 집단 | 중앙값 기준 MAPE (%) | 모델 MAPE (%) | 개선 (%p) |", "| --- | ---: | ---: | ---: |"]
    for r in baseline.itertuples():
        lines.append(f"| {r.phase} | {r.baseline_mape_pct:.2f} | {r.model_mape_pct:.2f} | {r.improvement_pctp:+.2f} |")
    lines += ["", "### Hold-out 포함 재학습의 영향", "",
        "Valid는 Train 29셀 모델, 외부 평가는 Hold-out을 합친 36셀 모델로 예측했다. 따라서 Valid−Test Gap에는 평가 배치 차이와 재학습 영향이 함께 들어간다. 고정 사양을 Train 29셀로만 적합한 외부 예측도 계산해 다음처럼 분리했다. 외부 결과를 사양 선택에 사용하지 않았다.", "",
        "| 배치 | Train 29셀 모델 MAPE (%) | 재학습 36셀 MAPE (%) | 재학습 차이 (%p) |", "| --- | ---: | ---: | ---: |"]
    for r in refit.itertuples():
        lines.append(f"| {r.batch} | {r.train29_mape_pct:.2f} | {r.refit36_mape_pct:.2f} | {r.refit_delta_pctp:+.2f} |")
    lines += ["", f"Batch 2의 기준 대비 개선에 대한 정책군 단위 paired bootstrap 2,000회의 탐색적 95% 구간은 [{baseline_ci[0]:.2f}, {baseline_ci[1]:.2f}]%p다. 고정 모델과 관측된 정책군에 대한 구간이며 새 배치나 모델 선택의 불확실성은 포함하지 않는다.", "",
        "### 입력 분포와 정책 구성", "",
        "아래 입력 분포 차이는 최종 학습 Batch 1의 36셀을 기준으로 (외부 평균 − 학습 평균) / 학습 표준편차로 계산했다. 절댓값이 크면 측정된 입력 분포 차이가 크다는 뜻이며 인과 관계나 실패 원인의 확정은 아니다.", "",
        "| Batch 2 입력 | 표준화 평균 차이 | 학습 입력 범위 밖 / 관측 셀 |", "| --- | ---: | ---: |"]
    for r in drift.loc[drift.batch.eq("Batch 2")].itertuples():
        lines.append(f"| {r.feature} | {r.standardized_mean_difference:+.2f} | {r.outside_train_range_n}/{r.n_observed} |")
    lines += ["", "mean_chargetime의 표준화 평균 차이는 +6.34로 크지만 최종 G0 모델은 충전시간을 직접 입력받지 않는다. 이 차이는 실험 조건 구성의 차이를 뒷받침하며, 해당 변수가 예측 오차를 일으켰다는 직접 증거는 아니다. 정책의 동일 여부는 c1·전환 SOC·c2의 정확한 조합으로 판단했다. 서로 다른 제조·운전 배치를 같은 정책이라는 이유만으로 동일 분포로 간주할 수 없다.", "",
        "| 외부 배치 | Batch 1에 같은 정책 존재 | n | MAPE (%) | 평균 예측−실제 |", "| --- | --- | ---: | ---: | ---: |"]
    for r in policies.itertuples():
        lines.append(f"| {r.batch} | {'있음' if r.policy_seen_in_batch1 else '없음'} | {r.n} | {r.mape_pct:.2f} | {r.bias_cycles:.2f} |")
    lines += ["", "Batch 2에서 기존 정책 셀도 MAPE 27.89%로 신규 정책 셀의 24.93%보다 높다. 그러므로 신규 정책이라는 사실만으로 성능 저하를 설명할 수 없다. Batch 3의 기존 정책 집단은 6셀뿐이어서 작은 집단의 평균 차이를 확정적인 효과로 해석하지 않는다.", "", "### 라벨 기준과 제외 영향", "",
        "| 배치 | 전체 | 결측·무효 라벨 | 유효 라벨이지만 종료 불확실 | 주 분석 |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in quality_audit.itertuples():
        lines.append(f"| {r.batch} | {r.all_n} | {r.missing_or_invalid_label_n} | {r.valid_label_eol_uncertain_n} | {r.main_cohort_n} |")
    lines += ["", f"Batch 1에서 종료 불확실로 제외한 {len(excluded)}셀에 고정 모델을 적용한 탐색적 MAPE는 {excluded.ape_pct.mean():.2f}%다. 이 값은 신뢰성이 불확실한 제공 라벨과의 일치도이며 정답 수명에 대한 검증이 아니다. 해당 셀을 성능 개선 목적으로 재학습에 추가하지 않았다. 제외 기준이 긴 수명 셀을 배제하여 학습 범위를 좁힐 수 있다는 모집단 선택 한계를 보여준다.", "",
        "## 9. 단수명 위험과 사용 범위", "",
        "DAY 1의 단수명 정의인 실제 수명 <500사이클을 사용해 회귀 예측을 사후 점검했다. 예측도 <500이면 단수명 경고로 본다. 이 임계값을 외부 성능에 맞춰 조정하지 않았고 별도 분류 모델을 선택한 것도 아니다.", "",
        "| 평가 집단 | 실제 단수명 셀 | 탐지 | 놓침 | 잘못된 단수명 경고 | 놓침 비율 (%) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in risk.itertuples():
        rate = f"{r.short_missed_pct:.2f}" if np.isfinite(r.short_missed_pct) else "해당 셀 없음"
        lines.append(f"| {r.phase} | {r.actual_short_n} | {r.short_detected_n} | {r.short_missed_n} | {r.false_short_n} | {rate} |")
    lines += ["", "이 결과로 현 모델이 빠르게 열화할 셀을 충분히 경고하는지 확인할 수 있다. 실제 단수명 라벨은 예측 시점에는 알 수 없으므로 이 표로 개별 셀을 사전 선별할 수는 없다. 실제 운영에서는 새로운 데이터에서 위험 탐지율·허위 경고율과 비용을 함께 검증해야 한다.", "",
        "### 다음 실험과 통과 조건", "",
        "1. 초기 100사이클 측정 후 수명 완료까지 추적한 새 셀을 충전정책·수명 범위별로 확보한다. 종료 정의를 통일하고 누락된 완료 라벨을 확인한다.",
        "2. 모델과 입력·전처리·<500 위험 기준을 사전에 고정하고 새로운 배치를 한 번만 평가한다. 현재 관찰한 Batch 2에 맞춘 변경은 후속 실험으로 구분한다.",
        "3. 평가 전에 MAPE 목표와 단수명 놓침 허용률을 사용 목적에 맞게 합의한다. 논문 9.1% 재현을 주장하려면 원본 데이터 구성과 라벨 정의까지 일치시킨다.",
        "4. 목표 미달 또는 단수명 놓침이 크면 수명 예측만으로 셀을 승인하지 않고 추가 시험 대상으로 사용한다. 현장 ESS 보증·안전 판단은 추가 검증 전까지 지원하지 않는다.", "",
        "## 10. 저장 모델과 평가 항목 대응", "",
        "선택 사양을 Batch 1의 36셀로 재학습한 모델은 `results/day2_model.joblib`, 입력 목록·학습 셀·패키지 버전은 `results/day2_model_manifest.json`에 저장했다. 새 CSV에서 `src/predict.py`로 같은 전처리와 입력 규칙을 적용한다. 초기 측정으로 만든 피처만 제공해야 하며 총수명 라벨이나 종료값을 입력할 필요는 없다.", ""]
    if "raw_input_intercept_cycles" in manifest:
        equation = " + ".join(f"({v:.2f}) × {k}" for k,v in manifest["raw_input_coefficients"].items())
        lines += [f"선택 회귀의 원 입력 단위 식은 `예측 사이클 = {manifest['raw_input_intercept_cycles']:.2f} + {equation}`이며 최솟값은 1사이클이다. ΔQ log 분산 증가에 따른 예측 변화는 통계적 관계이고 충전조건의 인과 효과를 나타내지 않는다.", ""]
    lines += ["| 평가 항목 | 구현 및 근거 |", "| --- | --- |",
        "| 전략 → 구현 (20) | 1절 전략 대응, 02 피처 노트북, G0~G4의 동일 모델 비교 |",
        "| Pipeline (40) | 고정 정책 분할·fold 내 전처리·화이트리스트·중첩 CV·저장 모델·검증 테스트 |",
        "| 성능 리포팅 (20) | 3절 가이드 성능표, 후보·fold·셀별 CSV, 기준 비교·Gap·탐색적 구간 |",
        "| 결과 해석 (20) | 4·8절 배치/입력/정책/라벨 분석, 9절 위험 점검·다음 실험과 적용 한계 |", "",
        "이번 추가 검증은 기존 외부 결과를 관찰한 뒤 보완한 분석이다. 새 데이터가 없어 독립적인 추가 외부 시험은 수행하지 못했다. `python src/train.py --root .`가 본 보완 결과까지 재생성한다.", ""]
    report = root / "report/DAY2_모델평가.md"
    base_report = report.read_text()
    key_findings = (
        f"\n**핵심 판단**\n\n"
        f"- 선택용 CV는 7.93%지만 선택 과정을 다시 수행한 중첩 CV는 {summary['nested_mape_mean_pct']:.2f}%다. 작은 표본에서 모델 선택이 불안정함을 확인했다.\n"
        f"- Batch 2 MAPE는 25.69%로 학습 중앙값 기준 58.73%보다 개선됐지만 논문 비교 기준 9.1%에는 미달했다.\n"
        f"- Batch 2의 실제 단수명 28셀 중 21셀(75%)을 경고하지 못해 단수명 선별·보증·안전 판단에 직접 사용하기 어렵다.\n"
    )
    base_report = base_report.replace("## 1. DAY 1 전략 → 구현",key_findings+"\n## 1. DAY 1 전략 → 구현",1)
    report.write_text(base_report+"\n".join(lines))
    return summary

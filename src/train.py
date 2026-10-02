"""Reproducible DAY 2 regression using the locked DAY 1 cohort and policy split.

Selection sees only Batch 1 train/CV. Hold-out and external batches are evaluated
after the winning specification has been fixed. All X columns are early-cycle data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from features import FEATURE_SETS, feature_matrix
from preprocess import load_data


MODEL_ORDER = {"Ridge": 0, "ElasticNet": 1, "RandomForest": 2, "GradientBoosting": 3}


def scores(y, pred):
    """Physical cycle predictions are positive; report errors in original units."""
    y = np.asarray(y, dtype=float)
    pred = np.maximum(np.asarray(pred, dtype=float), 1.0)
    return {
        "mape_pct": 100 * mean_absolute_percentage_error(y, pred),
        "mae_cycles": mean_absolute_error(y, pred),
        "rmse_cycles": np.sqrt(mean_squared_error(y, pred)),
    }


def specifications():
    """Small, fixed search space; no hold-out or external data influence it."""
    recipes = [
        ("Ridge", "alpha=0.1", Ridge(alpha=0.1), True),
        ("Ridge", "alpha=10", Ridge(alpha=10), True),
        ("ElasticNet", "alpha=0.01,l1=0.5", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000), True),
        ("ElasticNet", "alpha=0.1,l1=0.5", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000), True),
        ("RandomForest", "depth=2,leaf=2", RandomForestRegressor(n_estimators=100, max_depth=2, min_samples_leaf=2, random_state=42, n_jobs=1), False),
        ("RandomForest", "depth=3,leaf=2", RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=2, random_state=42, n_jobs=1), False),
        ("GradientBoosting", "trees=60,depth=1", GradientBoostingRegressor(n_estimators=60, learning_rate=0.05, max_depth=1, min_samples_leaf=3, random_state=42), False),
        ("GradientBoosting", "trees=120,depth=2", GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=2, min_samples_leaf=3, random_state=42), False),
    ]
    for group, features in FEATURE_SETS.items():
        for model, settings, estimator, linear in recipes:
            for target in ("raw", "log"):
                steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
                if linear:
                    steps.append(("scale", StandardScaler()))
                steps.append(("model", clone(estimator)))
                fitted = Pipeline(steps)
                if target == "log":
                    fitted = TransformedTargetRegressor(regressor=fitted, func=np.log, inverse_func=np.exp)
                yield {
                    "id": f"{group}__{model}__{settings}__{target}",
                    "feature_set": group, "features": features, "model": model,
                    "settings": settings, "target_scale": target, "estimator": fitted,
                }


def evaluate_cv(train, spec):
    X = feature_matrix(train, spec["features"])
    y = train.cycle_life.to_numpy(dtype=float)
    folds, predictions = [], []
    for fold in range(1, 6):
        valid = train.cv_fold.to_numpy() == fold
        fitted = clone(spec["estimator"]).fit(X.loc[~valid], y[~valid])
        pred = np.maximum(fitted.predict(X.loc[valid]), 1.0)
        metric = scores(y[valid], pred)
        folds.append({"candidate": spec["id"], "fold": fold, "train_n": int((~valid).sum()), "valid_n": int(valid.sum()), **metric})
        predictions.extend({"candidate": spec["id"], "cell_id": row.cell_id, "fold": fold,
                            "actual": float(row.cycle_life), "predicted": float(p)}
                           for row, p in zip(train.loc[valid].itertuples(), pred))
    fold_mape = [r["mape_pct"] for r in folds]
    all_pred = pd.DataFrame(predictions)
    summary = {k: v for k, v in spec.items() if k != "estimator"}
    summary["features"] = ", ".join(spec["features"])
    summary.update({"cv_mape_pct": float(np.mean(fold_mape)),
                    "cv_mape_sd_pct": float(np.std(fold_mape, ddof=1)),
                    "cv_oof_mape_pct": scores(all_pred.actual, all_pred.predicted)["mape_pct"],
                    "cv_mae_cycles": float(np.mean([r["mae_cycles"] for r in folds])),
                    "cv_rmse_cycles": float(np.mean([r["rmse_cycles"] for r in folds])),
                    "feature_count": len(spec["features"])})
    return summary, folds, predictions


def baseline_cv(train):
    rows = []
    for fold in range(1, 6):
        v = train.cv_fold == fold
        model = DummyRegressor(strategy="median").fit(np.zeros((int((~v).sum()), 1)), train.loc[~v, "cycle_life"])
        metric = scores(train.loc[v, "cycle_life"], model.predict(np.zeros((int(v.sum()), 1))))
        rows.append({"fold": fold, "valid_n": int(v.sum()), **metric})
    return pd.DataFrame(rows)


def prediction_frame(frame, pred, phase, train_min, train_max, fold=None):
    out = frame[["cell_id", "batch", "cycle_life", "policy_group", "c1", "soc_switch", "c2"]].copy()
    out = out.rename(columns={"cycle_life": "actual"})
    out["predicted"] = np.maximum(pred, 1.0)
    out["error_cycles"] = out.predicted - out.actual
    out["absolute_percentage_error"] = 100 * abs(out.error_cycles) / out.actual
    out["target_range"] = np.select([out.actual < train_min, out.actual > train_max], ["below_train", "above_train"], default="within_train")
    out["phase"] = phase
    if fold is not None:
        out["cv_fold"] = fold
    return out


def cluster_mape_interval(frame, repetitions=2000, seed=42):
    """Exploratory uncertainty for a fixed model and observed policy groups."""
    groups = [g for _, g in frame.groupby("policy_group")]
    assert len(groups) >= 2
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        sampled = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(scores(sampled.actual, sampled.predicted)["mape_pct"])
    return np.percentile(draws, [2.5, 97.5])


def save_plots(root, pred):
    results = Path(root) / "results"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    colors = {"CV": "#2878b5", "Valid": "#136c71", "Batch 2": "#dc784a", "Batch 3": "#6e66a7"}
    for name, g in pred.groupby("phase"):
        axes[0].scatter(g.actual, g.predicted, label=f"{name} (n={len(g)})", color=colors[name], alpha=.75, s=30)
    lim = max(pred.actual.max(), pred.predicted.max()) * 1.03
    axes[0].plot([0, lim], [0, lim], "--", color="#333", lw=1)
    axes[0].set(xlabel="Actual cycle life", ylabel="Predicted cycle life", title="Fixed model: predictions vs actual")
    axes[0].legend(fontsize=8)
    for name in ("Batch 2", "Batch 3"):
        g = pred[pred.phase == name]
        axes[1].scatter(g.actual, g.absolute_percentage_error, label=name, color=colors[name], alpha=.75, s=30)
    axes[1].axvline(534, color="#555", linestyle="--", linewidth=1)
    axes[1].axvline(1054, color="#555", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Actual cycle life", ylabel="Absolute percentage error (%)", title="External errors and Batch 1 train target range")
    axes[1].legend(fontsize=8)
    fig.savefig(results / "day2_predictions_and_errors.png", dpi=180)
    plt.close(fig)


def write_report(root, winner, candidate, baseline, performance, pred, ci):
    root = Path(root)
    get = lambda name: performance.set_index("index").loc[name, "mape_pct"]
    cv, valid, test2, test3 = (get(x) for x in ("Train (Batch 1 CV)", "Valid (Batch 1 Hold-out)", "Test (Batch 2)", "Test (Batch 3)"))
    def f(x): return f"{x:.2f}"
    p = pred[pred.phase == "Batch 2"]
    worst = p.nlargest(5, "absolute_percentage_error")
    slices = p.groupby("target_range").agg(n=("cell_id", "size"), mape_pct=("absolute_percentage_error", "mean"), mean_error_cycles=("error_cycles", "mean"))
    delta = pd.read_csv(root / "data/processed/cell_features.csv", usecols=["cell_id", "log_var_delta_q"])
    enriched = pred.merge(delta, on="cell_id", validate="many_to_one")
    train_delta = enriched.loc[enriched.phase == "CV", "log_var_delta_q"]
    xlo, xhi = train_delta.min(), train_delta.max()
    b2 = enriched.loc[enriched.phase == "Batch 2"].copy()
    b2["delta_range"] = np.where(b2.log_var_delta_q.between(xlo, xhi), "within_input", "outside_input")
    input_slices = b2.groupby("delta_range").agg(n=("cell_id", "size"), mape_pct=("absolute_percentage_error", "mean"), mean_error_cycles=("error_cycles", "mean"))
    policy_slices = p.groupby("policy_group").agg(n=("cell_id", "size"), median_life=("actual", "median"), mape_pct=("absolute_percentage_error", "mean"), mean_error_cycles=("error_cycles", "mean"))
    lines = [
        "# DAY 2 | 배터리 수명 예측 모델 개발 및 평가", "",
        "> **결론** · Batch 1의 정책 그룹 분리 교차검증으로 모델을 선택하고, 별도 Hold-out과 Batch 2에 순서대로 적용했다. 모든 수치는 실제 실행 결과다.", "",
        "## 1. DAY 1 전략 → 구현", "",
        "| DAY 1 관찰 | DAY 2 구현 |", "| --- | --- |",
        "| 초기 ΔQ log 분산의 일관된 수명 관계 | G0 핵심 입력으로 사용; G1~G3 및 RMS 전류 G4를 단계적으로 비교 |",
        "| 충전·온도 변수의 조건 의존성 및 ΔQ 변수 중복 | 좁은 피처 묶음과 Ridge/ElasticNet 정규화, 작은 트리 후보 비교 |",
        "| 셀 수가 적고 같은 충전 정책의 셀이 존재 | 셀 단위 표본, 정책 그룹이 겹치지 않는 Batch 1 CV와 Hold-out 사용 |",
        "| 전체 기록으로 계산한 knee는 미래 정보 | 초기 피처 화이트리스트로 knee·종료 QD·라벨·셀 ID 차단 |", "",
        f"선택된 사양은 **{winner['feature_set']} / {winner['model']} ({winner['settings']}) / {winner['target_scale']} 타깃**이다. 입력은 `{', '.join(winner['features'])}`다.", "",
        "## 2. 분할과 파이프라인", "",
        "Batch 1에서 수명 라벨이 있고 종료 유효 QD≤0.885 Ah인 36셀을 주 분석에 사용했다. 사전에 고정된 정책 그룹 분할(seed=42)로 Train/CV 29셀·16그룹과 Hold-out 7셀·4그룹을 나눴다. Train의 GroupKFold 5개 fold는 검증 6/6/6/6/5셀이고 각 경계에서 정책 중복 0개다. Batch 2는 39셀, Batch 3는 44셀을 같은 라벨 품질 기준으로 평가했다.", "",
        "각 fold의 학습 부분만으로 결측 중앙값과 선형 모델의 표준화를 적합했다. 원 타깃과 자연로그 타깃을 제한된 후보로 비교했고, 로그 예측은 exp로 원 사이클 단위에 복원했다. 모든 예측은 최소 1사이클로 제한했다. Hold-out과 외부 배치의 라벨은 피처·모델·설정 선택에 사용하지 않았다.", "",
        f"중앙값 기준 모델의 CV 평균 MAPE는 **{f(baseline.mape_pct.mean())}%**다. 총 {len(candidate)}개 사전 정의 후보를 같은 분할에서 비교해 평균 CV MAPE가 가장 작은 사양을 선택했다. 동률이면 fold 표준편차·피처 수·모델 단순성을 순서대로 적용한다. 작은 데이터에서 후보 수가 많아 CV 선택 편향이 남을 수 있다.", "",
        "상위 후보의 동일 분할 비교는 다음과 같다. 이 표의 수치는 Hold-out이나 Batch 2를 본 뒤 재선정한 값이 아니다.", "",
        "| 피처 묶음 | 모델 | 타깃 | CV MAPE 평균 (%) | fold 표준편차 (%p) |", "| --- | --- | --- | ---: | ---: |",
    ]
    for row in candidate.head(6).itertuples():
        lines.append(f"| {row.feature_set} | {row.model} ({row.settings}) | {row.target_scale} | {f(row.cv_mape_pct)} | {f(row.cv_mape_sd_pct)} |")
    lines += ["", "## 3. 가이드 형식 성능 결과", "",
        "| 구분 | MAPE (%) | 비고 |", "| --- | ---: | --- |",
        f"| Train (Batch 1 CV) | {f(cv)} ± {f(performance.set_index('index').loc['Train (Batch 1 CV)', 'mape_sd_pct'])} | 5개 검증 fold MAPE의 평균 ± 표준편차 |",
        f"| Valid (Batch 1 Hold-out) | {f(valid)} | 선택된 모델을 Train 29셀로 적합 후 7셀에서 1회 평가 |",
        f"| Test (Batch 2) | {f(test2)} | 사양 고정 후 Batch 1의 36셀로 재학습해 39셀 평가 |",
        f"| Gap (Train-Valid) | {f(valid-cv)} %p | Valid − CV; 양수면 내부 검증 악화 |",
        f"| Gap (Valid-Test) | {f(test2-valid)} %p | Test − Valid; 양수면 외부 배치 악화 |",
        f"| Gap (Target-Test) | {f(test2-9.1)} %p | Test − 9.1%; 원논문 비교 기준과 데이터·라벨 정책 차이 있음 |",
        f"| Test (Batch 3, 추가) | {f(test3)} | 모델 변경 없이 44셀 평가 |",
        f"| Gap (Batch 2-Batch 3) | {f(test3-test2)} %p | Batch 3 − Batch 2 |", "",
        f"보조 지표 MAE/RMSE는 Batch 2에서 {f(performance.set_index('index').loc['Test (Batch 2)', 'mae_cycles'])}/{f(performance.set_index('index').loc['Test (Batch 2)', 'rmse_cycles'])}사이클이다. Batch 2 MAPE의 정책군 복원추출 2,000회 탐색적 95% 구간은 [{f(ci[0])}, {f(ci[1])}]%다. 이는 모델 선택 불확실성·미관측 배치 차이를 포함하지 않는다.", "",
        "![실제 수명과 예측, 외부 배치 오차](../results/day2_predictions_and_errors.png)", "",
        "## 4. Batch 2 오류 분석", "",
        "모델 선택용 Train 29셀의 실제 수명 범위는 534~1054사이클이다. Batch 2 평가 39셀 중 30셀은 이보다 짧고 2셀은 길다. 외부 평가에 사용한 최종 모델은 Hold-out까지 포함한 Batch 1의 36셀(534~1074사이클)로 재학습했다. Batch 2의 범위 밖 셀 수는 이 기준에서도 30/2셀로 같다. 아래 표는 모델 선택용 Train 범위별 오차다.", "",
        "| 실제 수명 구간 | n | MAPE (%) | 평균 예측−실제 (사이클) |", "| --- | ---: | ---: | ---: |",
    ]
    for idx, row in slices.iterrows():
        lines.append(f"| {idx} | {int(row['n'])} | {f(row['mape_pct'])} | {f(row['mean_error_cycles'])} |")
    lines += ["", f"핵심 입력 `log_var_delta_q`의 Batch 1 Train 범위는 {xlo:.3f}~{xhi:.3f}이다. Batch 2에서는 이 입력 범위 안에 있는 셀도 크게 틀렸다. 따라서 타깃의 범위 차이만으로 오차를 설명하거나, 모든 실패를 입력 외삽 탓으로 돌릴 수 없다.", "",
              "| ΔQ 입력 범위 | n | MAPE (%) | 평균 예측−실제 (사이클) |", "| --- | ---: | ---: | ---: |"]
    for idx, row in input_slices.iterrows():
        lines.append(f"| {idx} | {int(row['n'])} | {f(row['mape_pct'])} | {f(row['mean_error_cycles'])} |")
    lines += ["", "충전 정책군별로도 오차가 균일하지 않다. 다음은 큰 오차 정책군 2개와, 단수명임에도 상대적으로 잘 맞은 정책군 1개다. 각 군의 표본 수가 작아 정책 효과의 인과 추정은 아니다.", "",
              "| 정책군 (c1 / 전환 SOC / c2) | n | 실제 수명 중앙값 | MAPE (%) | 평균 예측−실제 |", "| --- | ---: | ---: | ---: | ---: |"]
    examples = list(policy_slices.sort_values("mape_pct", ascending=False).head(2).iterrows()) + list(policy_slices.sort_values("mape_pct").head(1).iterrows())
    for key, row in examples:
        lines.append(f"| {key.replace('|', ' / ')} | {int(row['n'])} | {f(row['median_life'])} | {f(row['mape_pct'])} | {f(row['mean_error_cycles'])} |")
    lines += ["", "절대 백분율 오차가 큰 5셀은 다음과 같다. 단수명 셀은 MAPE에서 같은 사이클 오차도 더 크게 반영된다.", "",
              "| 셀 | 실제 | 예측 | 절대오차율 (%) | 충전 정책군 |", "| --- | ---: | ---: | ---: | --- |"]
    for row in worst.itertuples():
        lines.append(f"| {row.cell_id} | {f(row.actual)} | {f(row.predicted)} | {f(row.absolute_percentage_error)} | {row.policy_group.replace('|', ' / ')} |")
    lines += ["", "Batch 2에서 짧은 셀의 수명을 평균적으로 과대 예측했다. ESS 셀 선별에 그대로 쓰면 빨리 열화될 셀을 오래갈 것으로 판단할 수 있으므로, 초기 신호만으로 확정하지 않고 짧은 수명 위험 구간의 추가 시험이 필요하다. Batch 3에서는 모델 선택용 Train 상한 1054사이클보다 긴 17셀의 수명을 평균적으로 과소 예측했다(외부 평가용 재학습 36셀 상한 1074를 기준으로 하면 16셀).", "",
              "오차의 원인 가설은 Batch 2의 짧은 수명 집중, 충전 정책 구성 차이, 종료 라벨 정의와 기록 품질 차이다. 이 자료만으로 특정 충전 조건의 인과 효과를 확정할 수는 없다.", "",
              "## 5. ESS 관점과 한계", "",
              "초기 시험 신호가 신뢰할 수 있는 범위에서 예상 수명은 셀 선별·추가 시험 순서·교체 우선순위의 보조 지표가 될 수 있다. 실제 ESS 운영의 보증·안전 판단에 직접 쓰려면 운전 온도, 부분 충방전, 캘린더 열화, 제조 편차를 포함한 현장 데이터에서 다시 검증해야 한다.", "",
              "Hold-out은 7셀이고 Batch 2 정책군은 9개라 추정이 불안정하다. 종료 QD 기준은 라벨 진실성을 보증하지 않고 일부 셀을 제외해 모집단이 바뀐다. DAY 1 EDA에서 외부 배치의 라벨·분포를 이미 관찰했으므로 완전히 눈가림한 시험도 아니다. 논문의 9.1%와 직접 같은 실험으로 간주하지 않는다.", "",
              "## 6. 재현", "",
              "`python src/train.py --root .`로 같은 CSV·PNG·보고서를 다시 생성한다. `notebooks/03_modeling.ipynb`는 실행 과정과 결과 해석을 보여준다. 후보별 결과는 `results/day2_candidates.csv`, fold별 결과는 `results/day2_cv_folds.csv`, 셀별 예측은 `results/day2_predictions.csv`, 가이드 형식 표는 `results/model_performance.csv`, 상세 수치는 `results/day2_performance.csv`에 저장된다.", ""]
    (root / "report/DAY2_모델평가.md").write_text("\n".join(lines))


def run(root):
    root = Path(root)
    results = root / "results"
    train, holdout, external, plan = load_data(root)
    baseline = baseline_cv(train)
    baseline.to_csv(results / "day2_baseline_folds.csv", index=False)
    summaries, fold_rows, all_cv_pred = [], [], []
    specs = list(specifications())
    for spec in specs:
        summary, folds, pred = evaluate_cv(train, spec)
        summaries.append(summary); fold_rows.extend(folds); all_cv_pred.extend(pred)
    candidates = pd.DataFrame(summaries)
    candidates["model_order"] = candidates.model.map(MODEL_ORDER)
    candidates = candidates.sort_values(["cv_mape_pct", "cv_mape_sd_pct", "feature_count", "model_order", "id"]).reset_index(drop=True)
    candidates.to_csv(results / "day2_candidates.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(results / "day2_cv_folds.csv", index=False)
    winner_id = candidates.iloc[0].id
    winner = next(spec for spec in specs if spec["id"] == winner_id)
    cv = candidates.iloc[0]
    cv_pred = pd.DataFrame([x for x in all_cv_pred if x["candidate"] == winner_id])
    assert len(cv_pred) == len(train) and cv_pred.cell_id.is_unique
    cv_frame = train.merge(cv_pred[["cell_id", "predicted"]], on="cell_id", validate="one_to_one")
    pred_parts = [prediction_frame(cv_frame, cv_frame.predicted, "CV", plan["train_target_min"], plan["train_target_max"], cv_frame.cv_fold)]
    fitted_train = clone(winner["estimator"]).fit(feature_matrix(train, winner["features"]), train.cycle_life)
    holdout_pred = fitted_train.predict(feature_matrix(holdout, winner["features"]))
    pred_parts.append(prediction_frame(holdout, holdout_pred, "Valid", plan["train_target_min"], plan["train_target_max"]))
    full_batch1 = pd.concat([train, holdout], ignore_index=True)
    fitted_full = clone(winner["estimator"]).fit(feature_matrix(full_batch1, winner["features"]), full_batch1.cycle_life)
    for batch, frame in external.items():
        p = fitted_full.predict(feature_matrix(frame, winner["features"]))
        pred_parts.append(prediction_frame(frame, p, batch, plan["train_target_min"], plan["train_target_max"]))
    pred = pd.concat(pred_parts, ignore_index=True)
    pred.to_csv(results / "day2_predictions.csv", index=False)
    valid = scores(pred_parts[1].actual, pred_parts[1].predicted)
    test2 = scores(pred_parts[2].actual, pred_parts[2].predicted)
    test3 = scores(pred_parts[3].actual, pred_parts[3].predicted)
    rows = [
        {"index": "Train (Batch 1 CV)", "n": 29, "mape_pct": cv.cv_mape_pct, "mape_sd_pct": cv.cv_mape_sd_pct, "mae_cycles": cv.cv_mae_cycles, "rmse_cycles": cv.cv_rmse_cycles},
        {"index": "Valid (Batch 1 Hold-out)", "n": 7, **valid},
        {"index": "Test (Batch 2)", "n": 39, **test2},
        {"index": "Gap (Train-Valid)", "mape_pct": valid["mape_pct"] - cv.cv_mape_pct},
        {"index": "Gap (Valid-Test)", "mape_pct": test2["mape_pct"] - valid["mape_pct"]},
        {"index": "Gap (Target-Test)", "mape_pct": test2["mape_pct"] - 9.1},
        {"index": "Test (Batch 3)", "n": 44, **test3},
        {"index": "Gap (Batch 2-Batch 3)", "mape_pct": test3["mape_pct"] - test2["mape_pct"]},
    ]
    performance = pd.DataFrame(rows)
    performance.to_csv(results / "day2_performance.csv", index=False)
    # Guide-format summary for submission; source metrics remain in day2_performance.csv.
    notes = [
        f"5개 정책 그룹 CV fold MAPE 평균 ± 표준편차 {cv.cv_mape_sd_pct:.2f}%p",
        "선택 후 Hold-out 7셀에서 1회 평가",
        "Batch 1 전체 재학습 후 Batch 2 39셀 평가",
        "Valid − Train CV (%p)",
        "Batch 2 − Valid (%p)",
        "Batch 2 − 논문 비교 기준 9.1% (%p)",
        "모델 변경 없이 Batch 3 44셀 추가 평가",
        "Batch 3 − Batch 2 (%p)",
    ]
    guide = performance[["index", "mape_pct"]].rename(columns={"index": "구분", "mape_pct": "MAPE (%)"})
    guide["MAPE (%)"] = guide["MAPE (%)"].round(2)
    guide["비고"] = notes
    guide.to_csv(results / "model_performance.csv", index=False)
    ci = cluster_mape_interval(pred_parts[2])
    (results / "day2_selected_model.json").write_text(json.dumps({
        "candidate_id": winner_id, "feature_set": winner["feature_set"],
        "features": winner["features"], "model": winner["model"],
        "settings": winner["settings"], "target_scale": winner["target_scale"],
        "selection": "lowest mean fixed-fold Batch 1 train CV MAPE; tie by SD, feature count, model order",
        "baseline_cv_mape_pct": float(baseline.mape_pct.mean()),
        "batch2_mape_policy_cluster_ci95": list(map(float, ci)),
        "batch2_bootstrap_repetitions": 2000, "models_compared": len(candidates),
    }, ensure_ascii=False, indent=2))
    save_plots(root, pred)
    write_report(root, winner, candidates, baseline, performance, pred, ci)
    print("Selected:", winner_id)
    print(performance.to_string(index=False))
    return {"winner": winner, "candidates": candidates, "performance": performance, "predictions": pred}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    run(args.root)

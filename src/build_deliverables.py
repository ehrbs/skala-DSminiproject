"""Build the detailed DAY1 report and its executable EDA notebook."""
from pathlib import Path
import json,sys
import pandas as pd
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]

def measured_insights(cells,stats,corr):
    b1=cells[cells.batch=='Batch 1']
    ranked=corr[corr.batch=='Batch 1'].assign(mag=lambda d:d.spearman.abs()).sort_values('mag',ascending=False)
    top=ranked.head(4)
    lines=[f"{r.feature}: Spearman ρ={r.spearman:.3f}, n={int(r.n)}" for _,r in top.iterrows()]
    return lines

def build_report(cells,stats,corr):
    from detailed_day1 import build_detailed_report
    return build_detailed_report(ROOT)

def build_notebook(cells,stats,corr):
    nb=nbf.v4.new_notebook(); cs=[]
    def md(s):cs.append(nbf.v4.new_markdown_cell(s))
    def code(s):cs.append(nbf.v4.new_code_cell(s))
    md('# ESS 배터리 수명 예측 — DAY 1 EDA\n\n가이드의 5개 질문을 세 배치에서 확인하고 회귀 모델 설계로 연결합니다. 원본 scratch를 참고하여 확장했습니다. **모든 통계는 원본 .mat에서 직접 계산한 결과입니다.**\n\n예측 시점: 초기 100사이클 / 타깃: 총수명 cycle_life / DAY 2 학습·평가는 03_modeling.ipynb에서 확인할 수 있습니다.')
    code('from pathlib import Path\nimport sys, json\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nif ROOT.name == "notebooks": ROOT = ROOT.parent\nassert (ROOT / "src/eda.py").exists(), "ess_day1 또는 notebooks 폴더에서 실행하세요"\nsys.path.insert(0, str(ROOT / "src"))\nimport eda\nDATA_DIR = ROOT / "data/raw"\nPROCESSED = ROOT / "data/processed"\nprint("원본 데이터:", DATA_DIR.relative_to(ROOT))')
    md('## 0. 로딩과 데이터 품질\n\n원본 HDF5에서 summary와 초기 곡선만 읽습니다. 반복 실행에서는 이미 계산한 CSV/NPZ를 사용합니다. 원본을 바꾸면 `REBUILD=True`로 다시 생성하세요. 전압별 곡선과 summary의 실제 사이클 번호를 대응시킵니다.')
    code('REBUILD = False\nif REBUILD or not (PROCESSED / "cell_features.csv").exists():\n    cells, summary, curves = eda.extract(DATA_DIR, PROCESSED)\nelse:\n    cells = pd.read_csv(PROCESSED / "cell_features.csv")\n    summary = pd.read_csv(PROCESSED / "cycle_summary.csv")\n    z = np.load(PROCESSED / "delta_curves.npz")\n    curves = {k:z[k] for k in z.files}\neda.plots(cells, summary, curves, ROOT / "results")\neda.tables(cells, ROOT / "results")\nstats = pd.read_csv(ROOT / "results/batch_statistics.csv")\ncorr = pd.read_csv(ROOT / "results/feature_correlations.csv")\ndisplay(stats)\ndisplay(cells.groupby("batch")[[c for c in cells if c.startswith("invalid_")]].sum())')
    md('**품질 기준:** 용량 ≤0 또는 >1.32 Ah, IR ≤0, 온도 ≤0 또는 >100°C, 충전시간 ≤0은 피처 계산에서 결측 처리합니다. 원본 요약 CSV에는 원래 값이 남습니다. 말기 저용량을 일괄 제거하지 않습니다. 종료 용량 >0.885 Ah는 수명 완료 불확실로 표시합니다. 결측 수명 라벨은 분포/상관/클래스 비율 계산에서 제외하고 별도로 셉니다. 회색 곡선은 라벨 결측 셀입니다.')
    code('display(cells.loc[cells.eol_not_observed | ~cells.delta_available | cells.cycle_life.isna(), ["cell_id","batch","cycle_life","last_cycle","qd_last_valid","eol_not_observed","delta_available"]])\ndisplay(pd.DataFrame(json.loads((PROCESSED / "data_manifest.json").read_text())))')
    md('## 1. Cycle life 분포\n\n장수명 >1,000 / 단수명 <500은 EDA 그룹이고, 분류 라벨 ≥550 기준과는 다릅니다. 이상치는 자동 제거하지 않고 종료 여부·측정 품질을 함께 확인합니다.')
    code('display(Image(filename=str(ROOT / "results/01_life_distribution.png")))\ndisplay(stats[["batch","n","min","median","mean","max","short_lt500","long_gt1000","label0_lt550","label1_ge550"]])\ndisplay(cells.nsmallest(10,"cycle_life")[["cell_id","batch","cycle_life","charging_policy","eol_not_observed"]])')
    md('\n'.join(f'- {r.batch}: n={int(r.n)}, 중앙값 {r["median"]:.1f}, 단수명(<500) {int(r.short_lt500)}개, 장수명(>1000) {int(r.long_gt1000)}개.' for _,r in stats.iterrows())+'\n\n**전략:** 결측 라벨은 추정해 채우지 않습니다. 수명 범위 차이는 외삽 오차와 연결될 수 있습니다. DAY 2에서 수명 구간별 잔차와 MAPE를 확인합니다.')
    md('## 2. 열화 곡선과 knee 후보\n\n색상은 실제 총수명에 대응합니다. EOL 표시선은 공칭 1.1 Ah × 80% = 0.88 Ah입니다. 전체 수명 궤적은 탐색용이며 모델 입력은 초기 100사이클 이내로 제한합니다.')
    code('display(Image(filename=str(ROOT / "results/02_degradation.png")))\ndisplay(Image(filename=str(ROOT / "results/09_knee_examples.png")))\ndisplay(Image(filename=str(ROOT / "results/07_knee_candidates.png")))\ndisplay(cells.groupby("batch").knee_cycle.agg(["count","median","min","max"]))')
    md('knee는 11점 중앙값 평활화 뒤 연속 두 직선의 전환점을 탐색한 **후보**입니다. 단일 직선 대비 SSE ≥20% 감소, 후반 하강 기울기 크기 ≥전반 1.5배 조건을 사용합니다. 물리적 확정값은 아닙니다. **전략:** 전체 곡선의 knee 대신 초기 QD 기울기·표준편차를 피처로 검토합니다.')
    md('## 3. ΔQ(V) — Q100(V) − Q10(V)\n\nQdlin의 두 사이클을 같은 보간 위치에서 뺍니다. raw Qd를 시간 위치에 따라 직접 빼지 않습니다. 유효점 900개 이상인 1,000포인트 곡선에서 log 분산·최솟값·평균을 계산합니다.')
    code('display(Image(filename=str(ROOT / "results/03_delta_q.png")))\ndisplay(Image(filename=str(ROOT / "results/08_delta_examples.png")))\ndisplay(cells[["cell_id","batch","cycle_life","log_var_delta_q","min_delta_q","mean_delta_q","delta_available"]].head(12))')
    md('**전략:** 작은 셀 표본 수를 고려해 1,000개 곡선 위치 대신 요약 통계를 1차 후보로 사용합니다. 외부 배치의 수명 라벨을 보고 전압 구간을 최적화하지 않습니다. 분류 과제를 선택한다면 이 100−10 피처는 사용할 수 없습니다.')
    md('## 4. 충전 조건과 수명\n\nC-rate 단계와 전환 SOC, 초기 실제 전류 및 충전시간을 비교합니다. 정책별 평균만 보지 않고 표준편차와 표본 수도 함께 확인합니다.')
    code('display(Image(filename=str(ROOT / "results/05_policy.png")))\ndisplay(pd.read_csv(ROOT / "results/policy_statistics.csv"))\ndisplay(Image(filename=str(ROOT / "results/10_charging_measurements.png")))')
    md('**전략:** 정책 문자열의 단순 암기 대신 c1 / soc_switch / c2를 수치로 분해해 검토합니다. n=1의 SD는 미확정입니다. 상관관계만으로 충전 속도의 인과 효과를 확정하지 않습니다.')
    md('## 5. 초기 신호의 상관과 다중공선성\n\n셀 단위 초기 100사이클 피처를 사용합니다. Spearman·Pearson과 유효 표본 수를 기록합니다. 0인 IR 값은 평균에 포함하지 않습니다.')
    code('display(Image(filename=str(ROOT / "results/04_early_signals.png")))\ndisplay(Image(filename=str(ROOT / "results/06_correlations.png")))\ndisplay(corr.sort_values(["batch","spearman"]))\n# Batch 1 수명 미완료 가능 셀 제외에 따른 민감도\neligible = cells[(cells.batch == "Batch 1") & ~cells.eol_not_observed]\nsensitivity = eligible[eda.FEATURES + ["cycle_life"]].corr(method="spearman")["cycle_life"].drop("cycle_life").rename("spearman_terminal_screened")\nsensitivity.to_csv(ROOT / "results/batch1_eol_sensitivity.csv")\ndisplay(sensitivity.sort_values(key=abs,ascending=False))')
    md('Batch 1의 강한 초기 신호:\n\n'+'\n'.join('- '+s for s in measured_insights(cells,stats,corr))+'\n\n**전략:** 중복 온도 신호는 하나를 선택하거나 Ridge 정규화로 제어합니다. 약한 선형 상관만으로 비선형 피처를 삭제하지 않습니다. p-value는 탐색용이며 확증 검정 결과가 아닙니다.')
    md('## 6. 회귀 모델 설계 전략\n\n- 목표: 초기 100사이클로 총수명 cycle_life 예측. MAPE가 주 지표, RMSE/MAE 보조.\n- 피처 후보: ΔQ log 분산·최솟값·평균, 초기 QD 기울기, 온도·IR·충전시간, 단계별 C-rate 및 전환 SOC.\n- 후보 모델: 중앙값 기준 모델 → Ridge/ElasticNet → Random Forest → 얕은 Gradient Boosting.\n- Batch 1 셀 단위 Hold-out을 고정하고 나머지로 CV. 정책 외삽을 평가하려면 그룹 분리 가능성을 확인.\n- 전처리·대체·스케일링·피처 선택은 각 학습 fold에서만 fit. 난수 시드 42.\n- Batch 2 최종 평가, Batch 3 선택 추가 평가. DAY1 요구에 따라 세 배치 EDA를 이미 관찰했음을 공개.\n- 종료 사이클·전체 knee·말기 용량·타깃 파생값은 입력 금지.\n- MAPE Gap: Valid−CV / Test−Valid / Test−9.1% (%p).\n\n가이드 Batch2(2018-02-20)는 연구진 공개 로더 Batch2(2017-06-30)와 다르므로 원논문 셀 병합·제외 인덱스를 그대로 적용하지 않았습니다. 원논문 9.1%는 비교 목표이며 동일 조건의 재현 결과가 아닙니다.')
    md('## 7. 재현 환경과 참고 자료\n\n실습 가이드: https://actually-war-1ea.notion.site/DS-Mini-Project-32d7f4c8669380338a27f90c471c1fcb\n\n데이터: https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle\n\n연구진 로더: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation')
    code('import sys\nimport importlib.metadata as metadata\n\nprint(f"Python: {sys.version.split()[0]}")\nprint(f"Kernel executable: {Path(sys.executable).name}")\npackages = ["numpy", "pandas", "matplotlib", "scipy", "h5py", "scikit-learn", "nbformat", "nbclient"]\nversions = []\nmissing_packages = []\nfor package in packages:\n    try:\n        versions.append(metadata.version(package))\n    except metadata.PackageNotFoundError:\n        versions.append("미설치")\n        missing_packages.append(package)\ndisplay(pd.DataFrame({"package": packages, "version": versions}))\nif missing_packages:\n    print("현재 커널에서 확인되지 않은 패키지: " + ", ".join(missing_packages))\n    print("필요하면 이 노트북의 새 셀에서 실행: %pip install " + " ".join(missing_packages))\n    print("프로젝트 권장 실행 환경은 README의 Python 3.12 / ESS DAY1 커널입니다.")\n')
    nb.cells=cs;nb.metadata={'kernelspec':{'display_name':'ESS DAY1','language':'python','name':'ess-day1'},'language_info':{'name':'python','version':sys.version.split()[0]}}
    path=ROOT/'notebooks/01_EDA.ipynb';nbf.write(nb,path)
    from detailed_day1 import enhance_notebook
    return enhance_notebook(ROOT)

if __name__=='__main__':
    cells=pd.read_csv(ROOT/'data/processed/cell_features.csv')
    stats=pd.read_csv(ROOT/'results/batch_statistics.csv');corr=pd.read_csv(ROOT/'results/feature_correlations.csv')
    print(build_report(cells,stats,corr));print(build_notebook(cells,stats,corr))

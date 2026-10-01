# ESS 배터리 초기 신호 기반 수명 예측

배터리의 초기 충·방전 데이터에서 열화 신호를 찾고, 이를 이용해 총수명을 예측하는 데이터 분석 프로젝트입니다. MIT–Stanford 배터리 실험 데이터의 세 배치를 탐색하여 수명과 관련된 피처를 선정하고, ESS 셀 선별과 교체 우선순위 판단에 활용할 수 있는 회귀 모델을 설계합니다.

현재 **탐색적 데이터 분석(EDA)과 모델 설계**를 완료했습니다. 질문별 발견을 피처·처리·모델 선택으로 연결한 12쪽 제출 보고서와 상세 실행 노트북을 제공합니다. 모델 학습과 외부 배치 성능 평가는 후속 단계로 진행할 예정입니다.

## 1. 프로젝트 목표

- 초기 100사이클의 측정값으로 총수명 `cycle_life`를 예측합니다.
- 방전용량 변화, 충전 조건, 온도, 내부저항 중 수명과 관련된 신호를 탐색합니다.
- 배치별 분포 차이와 데이터 품질을 확인하여 검증 전략을 수립합니다.

타깃은 공칭 용량의 80%에 도달할 때까지의 **총 사이클 수**입니다. 특정 시점에서의 잔여수명(RUL)과 구분합니다. 회귀의 주 평가 지표는 MAPE(%), 보조 지표는 MAE와 RMSE입니다.

## 2. 데이터셋

[MIT–Stanford 배터리 실험 데이터의 Kaggle 배포본](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)을 사용합니다. 각 셀에는 수명 라벨, 충전 정책, 사이클별 요약 측정값과 사이클 내부의 전압·전류·온도·용량 곡선이 포함되어 있습니다.

| 배치 | 원본 파일 날짜 | 전체 셀 | 유효 수명 라벨 | 수명 중앙값¹ | 모델링 용도 |
|---|---|---:|---:|---:|---|
| Batch 1 | 2017-05-12 | 46 | 46 | 858.5 | 학습 및 내부 검증 |
| Batch 2 | 2018-02-20 | 47 | 39 | 472.0 | 최종 외부 평가 |
| Batch 3 | 2018-04-12 | 46 | 44 | 1005.5 | 추가 외부 평가 |
| 합계 | — | **139** | **129** | — | |

¹ 단위: 사이클. 중앙값은 수명 라벨이 있는 셀을 기준으로 계산했습니다. 결측 라벨 10개는 임의로 채우지 않았습니다.

## 3. 분석 내용과 주요 결과

### 수명 분포와 열화 곡선

배치별 수명 분포, 장수명(>1,000사이클)·단수명(<500사이클) 비율, 방전용량 감소와 가속 열화 구간을 비교했습니다. Batch 2의 수명 중앙값은 Batch 1·3보다 낮아, 배치 간 분포 차이를 고려한 평가가 필요합니다.

![배치별 수명 분포](results/01_life_distribution.png)

전체 수명 곡선에서 구한 knee는 탐색적 전환점 후보입니다. 27개 탐지 설정의 민감도를 비교해 131/139셀에서 모두 탐지되는 것을 확인했습니다. 위치의 작은 차이는 탐색 격자 해상도(기록 길이의 1%)에 제한됩니다. 예측 시점 이후의 정보를 포함하므로 모델 입력으로 사용하지 않습니다.

### 초기 방전곡선 변화 ΔQ(V)

동일 전압 축으로 보간된 방전용량 `Qdlin`을 이용해 다음 변화량을 계산했습니다.

```text
ΔQ(V) = Q100(V) − Q10(V)
```

Batch 1에서 ΔQ의 log10 분산과 수명의 Spearman 상관계수는 **−0.871(n=46)**입니다. 종료 용량 기준으로 수명 완료가 불확실한 셀을 제외해도 **−0.826(n=36)**으로 유지되어, 초기 피처 후보로 선정했습니다. ΔQ log 분산 하나를 핵심 기본형으로 두고 용량 기울기·충전시간·C-rate·평균 온도를 단계적으로 추가하고, 시간 가중 RMS 전류를 별도 확장 후보로 검증할 계획입니다. 이는 탐색적 상관 결과이며 예측 모델의 성능을 뜻하지 않습니다.

![초기 신호와 수명 관계](results/04_early_signals.png)

### 충전 조건과 다중공선성

충전 정책을 단계별 C-rate와 전환 SOC로 분해하고, 실측 전류·충전시간·온도·내부저항과 수명의 관계를 비교했습니다. Batch 1에서 평균 Tavg와 Tmax의 Spearman 상관은 약 **0.95**로, 온도 변수 간 중복 신호가 큽니다. 충전 조건과 수명의 관계는 배치에 따라 달라지며, 상관만으로 인과 효과를 판단하지 않습니다.

### 데이터 품질

종료 시 유효 용량이 0.885 Ah를 초과하거나 제공되지 않은 셀 16개를 수명 완료 불확실로 표시했습니다. 결측 수명 라벨 목록과 일부 겹치므로 두 수를 합산해 제외 셀 수로 해석하지 않습니다. 주 분석에서는 해당 완료 불확실 셀을 제외하며, Batch 1 전체 46셀의 민감도 분석은 별도로 보고합니다. 라벨 정책은 고정한 분할 계획에 명시했습니다.

## 4. 피처 및 모델 설계

| 피처 후보 | 분석 근거 |
|---|---|
| ΔQ의 log10 분산·최솟값·평균 | 초기 방전곡선 변화와 수명의 관계 |
| 초기 방전용량 평균·표준편차, 10~100사이클 기울기 | 초기 용량 수준과 열화 추세 |
| 초기 온도·내부저항 및 저항 변화량 | 열적 조건과 셀 상태 |
| 충전시간·실측 양의 전류 | 실제 충전 조건 |
| 1·2단계 C-rate와 전환 SOC | 충전 프로토콜의 수치 표현 |

Batch 1에서 550사이클 미만 셀은 1개뿐이므로 해당 기준의 분류 검증은 불안정합니다. 연속적인 수명 차이를 활용하고 초기 100사이클의 ΔQ 신호를 반영하기 위해 회귀를 선택했습니다.

후보 모델은 중앙값 기준 모델, Ridge/ElasticNet, Random Forest, 얕은 Gradient Boosting입니다. 작은 셀 표본 수를 고려하여 단순한 모델부터 비교하고 복잡도를 제한할 계획입니다.

### 검증 계획

1. 라벨이 유효하고 종료 용량 ≤0.885 Ah인 셀을 주 분석 대상으로 고정합니다. Batch 1의 36셀·20정책 그룹을 학습 29셀·16그룹과 Hold-out 7셀·4그룹으로 분리합니다(seed=42). 학습 부분에서 정책 그룹이 겹치지 않는 5-fold CV를 사용합니다. 이 파일은 분할 계획이며 모델 학습 결과가 아닙니다.
2. 결측 대체·스케일링·피처 선택과 튜닝은 각 학습 fold에서만 수행합니다.
3. 최종 모델을 선택한 뒤 Batch 2에서 평가하고, Batch 3에서 추가 평가합니다. 외부 평가 결과를 보고 모델을 다시 선택하거나 튜닝하지 않습니다.
4. MAPE·MAE·RMSE와 수명 구간별 오차를 확인합니다. MAPE 차이는 `Valid−CV`, `Test−Valid`로 정의하며 단위는 퍼센트포인트(%p)입니다.

현재 EDA에서 세 배치의 수명 라벨과 분포를 관찰했으므로, 이후 외부 평가를 완전히 눈가림한 테스트라고 해석하지 않습니다. 과제의 비교 목표인 MAPE 9.1%는 본 프로젝트에서 달성한 성능이 아닙니다.

## 5. 프로젝트 구조

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_EDA.ipynb              # 실행 결과와 해석이 포함된 EDA
├── data/
│   ├── raw/                     # 원본 MAT 저장 위치
│   └── processed/               # 셀 피처, 사이클 요약, 전압별 곡선
├── src/
│   ├── eda.py                   # 원본 로딩·피처 계산·시각화
│   ├── build_deliverables.py     # 보고서·노트북 생성
│   ├── detailed_day1.py          # 질문별 추가 통계·상세 모델 전략 보고서
│   ├── strategy_checks.py       # Knee 민감도·초기 전류·고정 분할·입력 제한
│   └── review_day1.py            # 원본과 분석 결과의 독립 대조
├── results/                     # 분석 그래프·통계표·원본 대조 결과
└── report/
    ├── DAY1_모델설계.pdf        # 12쪽 제출본
    ├── DAY1_모델전략_압축.md    # 제출본 원고
    └── DAY1_모델전략_상세.md    # 상세 분석 근거
```

- [분석 노트북](notebooks/01_EDA.ipynb)
- [모델 설계 보고서](report/DAY1_모델설계.pdf)
- [압축 제출본 원고](report/DAY1_모델전략_압축.md)
- [상세 모델 전략 원고](report/DAY1_모델전략_상세.md)

## 6. 실행 방법

Python **3.12**를 사용했습니다. 이 README가 있는 프로젝트 폴더에서 환경을 구성합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ess-day1 --display-name "ESS DAY1"
```

Windows에서는 가상환경 활성화 명령을 `.venv\Scripts\activate`로 변경합니다. VS Code 또는 Jupyter에서 `notebooks/01_EDA.ipynb`를 열고 **ESS DAY1** 커널을 선택한 뒤 전체 셀을 실행합니다.

### 가공 데이터로 실행

기본값 `REBUILD=False`는 `data/processed/`의 CSV·NPZ를 사용합니다. 이 파일들이 있으면 대용량 원본을 내려받지 않고 EDA를 재실행할 수 있습니다.

### 원본 데이터부터 실행

Kaggle에서 아래 세 파일을 다운로드하여 `data/raw/`에 저장합니다.

```text
2017-05-12_batchdata_updated_struct_errorcorrect.mat
2018-02-20_batchdata_updated_struct_errorcorrect.mat
2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

노트북에서 `REBUILD=True`로 설정하거나 아래 명령을 실행합니다.

```bash
python src/eda.py --root .
```

원본과 저장된 결과를 독립적으로 대조하려면 다음 명령을 사용합니다.

```bash
python src/review_day1.py
```

보고서와 노트북 템플릿은 `python src/build_deliverables.py`로 재생성합니다. 보고서 생성에는 macOS AppleGothic을 사용하므로 다른 운영체제에서는 `src/detailed_day1.py`의 `FONT`를 사용 가능한 한글 TTF 경로로 변경해야 합니다. 이 명령은 노트북도 새로 생성하므로 실행 출력을 다시 저장해야 합니다.

## 7. 분석 기준과 적용 한계

- 원본 사이클 요약 CSV는 측정값을 보존합니다. 피처 계산에서는 용량 ≤0 또는 >1.32 Ah, IR ≤0, 온도 ≤0 또는 >100°C, 충전시간 ≤0을 결측 처리합니다. 정상적인 말기 용량 감소를 일괄 제거하지 않습니다.
- `eol_not_observed`는 종료 용량 기준의 완료 불확실 플래그이며, `eol_below_088_observed`는 실제 0.88 Ah 미만 관측 여부입니다. 기준 통과만으로 수명 라벨의 신뢰성이 확정되는 것은 아닙니다.
- 전압 축은 실제 `Vdlin`의 3.5→2.0 V를 사용했습니다. 이번 Batch 2 파일은 연구진 공개 로더의 2017-06-30 파일과 달라, 연구진의 셀 병합·제외 인덱스를 그대로 적용하지 않았습니다.
- 연구용 셀의 실험 조건과 실제 ESS 운전 조건은 다를 수 있습니다. 실제 교체 의사결정에 적용하려면 운전 조건 변화, 캘린더 열화와 예측 불확실성을 추가로 검증해야 합니다.

## 8. 참고 자료

- [미니 프로젝트 실습 가이드](https://actually-war-1ea.notion.site/DS-Mini-Project-32d7f4c8669380338a27f90c471c1fcb)
- [Kaggle 데이터셋](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)
- [연구진 공개 데이터 로더](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
- [Severson et al. (2019), Data-driven prediction of battery cycle life before capacity degradation](https://www.nature.com/articles/s41560-019-0356-8)

### 커널과 패키지 확인

노트북 마지막 셀은 현재 커널의 Python 경로와 패키지 버전을 표시합니다. 미설치 패키지는 `미설치`로 표시하며 셀 실행을 중단하지 않습니다. 필요한 패키지는 노트북에서 `%pip install nbformat nbclient`처럼 설치하면 현재 커널에 적용됩니다. 권장 재현 환경은 위의 Python 3.12 / ESS DAY1 커널입니다.

## 9. 추가 검증과 재현

- 전체 139셀에서 공통 cycle 2~5의 시간 가중 전류 평균·RMS·95분위·4 A 초과 시간 비율을 계산했습니다. Batch 1의 cycle 1에는 유효 충전 구간이 없어 공통 창을 사용했습니다. 새 입력은 `data/processed/early_current_features.csv`입니다.
- Knee 평활화 7/11/21 × 개선율 10/20/30% × 기울기 비율 1.2/1.5/2.0의 27설정 결과를 `results/knee_sensitivity_grid.csv`에 저장했습니다.
- 셀별 Hold-out/CV 배정은 `data/processed/batch1_split_plan.csv`, 라벨 정책과 초기 입력 화이트리스트는 `data/processed/modeling_plan.json`에 저장했습니다. 모든 분할의 정책 그룹 중복은 0입니다.
- Hold-out 7셀·CV fold 5~6셀의 작은 표본, 종료 기준에 의한 모집단 선택, 사전에 관찰한 EDA로 인해 평가에는 한계가 있습니다. DAY 2에서 오차와 불확실성을 보고합니다.

```bash
# 저장된 초기 전류 피처로 추가 분석 재실행
python src/strategy_checks.py
# 원본의 공통 초기 사이클에서 전류 피처 재추출
python src/strategy_checks.py --raw-dir data/raw
# 시간 가중·입력 누수·그룹 분리 검증
python -m unittest discover -s tests
```

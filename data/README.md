# 데이터와 재현

[Kaggle 배포본](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)의 아래 파일을 `data/raw/`에 둡니다. 원본 MAT는 Git에서 제외합니다.

| 배치 | 파일 | 전체 / 유효 라벨 / 모델링 셀 |
| --- | --- | --- |
| Batch 1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 46 / 46 / 36 |
| Batch 2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | 47 / 39 / 39 |
| Batch 3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 46 / 44 / 44 |

전체 139셀 중 결측 수명 라벨 10개는 채우지 않았습니다. 유효 라벨이 있는 Batch 1의 10셀은 종료 용량 기준에서 완료 불확실로 제외했습니다. 파일 해시·출처는 [data_manifest.json](processed/data_manifest.json)에 기록했습니다.

## 가공 데이터

- `cell_features.csv`, `early_current_features.csv`: 셀별 초기 피처와 라벨·품질 진단 값. 전체 열을 모델에 넣지 않고 `src/features.py`에서 허용된 초기 피처만 선택합니다.
- `batch1_split_plan.csv`, `modeling_plan.json`: 고정된 정책 그룹 분할과 DAY 1 설계. `models_trained: false`는 당시 설계 시점의 기록입니다. 최종 학습 결과는 `results/day2_*`에 있습니다.
- `cycle_summary.csv`: 원본 사이클별 측정값. EDA 재현에 사용하며 각 행을 독립 모델 표본으로 취급하지 않습니다.
- `q10/q100/delta/voltage_curves.npz`: 초기 방전곡선 및 전압 축. ΔQ 추출·EDA·원본 대조에 사용합니다.
- `current_profile_examples.csv/.npz`: DAY 1 실측 전류 예시의 메타데이터와 곡선입니다.

이 가공 파일들로 세 노트북을 실행할 수 있습니다. 전압 축은 원본 `Vdlin`의 3.5→2.0 V이며, Batch 2 파일 날짜가 연구진 공개 로더와 달라 셀 병합·제외 인덱스를 그대로 적용하지 않았습니다.

## 전처리와 원본 대조

프로젝트 루트에서 실행합니다.

```bash
python src/eda.py --root .
python src/strategy_checks.py --raw-dir data/raw
python src/review_day1.py --raw-dir data/raw
```

원본이 다른 위치에 있다면 `--raw-dir`에 그 폴더를 지정할 수 있습니다. DAY 1 EDA 노트북에서는 `REBUILD=True`가 원본 추출을 다시 실행합니다.

용량 ≤0 또는 >1.32 Ah, IR ≤0, 온도 ≤0 또는 >100°C, 충전시간 ≤0은 피처 계산에서 결측 처리합니다. 원본 사이클 CSV에는 측정값을 보존합니다. 모델링 집단은 유효 수명 라벨과 종료 유효 QD≤0.885 Ah 기준으로 정합니다. 이 기준이 라벨의 진실성을 보장하지는 않으며, 제외 셀 때문에 학습 수명 범위가 좁아질 수 있습니다.

전체 수명의 knee·종료 QD·종료 사이클·수명 라벨은 모델 입력에 사용하지 않습니다. 기존 `early_positive_current`와 공통 cycle 2~5에서 시간 가중으로 추출한 `charge_*`는 계산 방식이 다릅니다. 상세 방법과 한계는 [DAY 1 근거](../report/DAY1_모델전략_상세.md)를 참고하세요.

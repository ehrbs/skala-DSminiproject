# 결과 파일 안내

| 확인할 내용 | 파일 |
| --- | --- |
| 제출 형식 성능표 | [model_performance.csv](model_performance.csv) |
| MAE·RMSE·정확한 MAPE·Gap | [day2_performance.csv](day2_performance.csv) |
| 최종 모델 선택 근거 | `day2_candidates.csv`, `day2_cv_folds.csv`, `day2_selected_model.json` |
| 셀별 예측과 오차 | `day2_predictions.csv`, `day2_predictions_and_errors.png` |
| 중첩 CV와 분할 검증 근거 | `day2_nested_*.csv`, `day2_validation_summary.json` |
| 같은 모델의 피처 추가 효과 | `day2_ablation_*.csv`, `day2_validation_and_ablation.png` |
| 중앙값 기준 비교·짝지은 오차 | `day2_baseline_*.csv` |
| 배치 차이·품질·위험 분석 | `day2_input_drift.csv`, `day2_policy_transfer.csv`, `day2_label_audit.csv`, `day2_excluded_label_sensitivity.csv`, `day2_refit_sensitivity.csv`, `day2_short_life_risk.csv` |
| 저장 모델과 입력 정의 | `day2_model.joblib`, `day2_model_manifest.json` |
| DAY 1 EDA·민감도 분석 | `01_`~`16_` 그림과 나머지 통계 CSV |
| 원본 MAT와 가공 결과 대조 | `independent_review.json` |

성능표의 Gap 단위는 %p입니다. 원 수명 예측의 MAPE·MAE·RMSE를 보고하며, fold 평균과 모든 OOF 예측을 합친 지표는 구분합니다. 중첩 검증은 후보 선택의 불안정성을 점검하는 보조 결과입니다.

상세 CSV는 보고서의 수치를 다시 계산하고 테스트하는 근거입니다. `python src/train.py --root .`는 DAY 2 결과와 보고서를 재생성합니다. [DAY 2 보고서](../report/DAY2_모델평가.md)에 해석과 적용 한계가 있습니다.

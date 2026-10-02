# 데이터 안내

원본은 [Kaggle의 배터리 수명 데이터](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)에서 내려받습니다. 대용량 `.mat` 파일은 `data/raw/`에 두며 Git에는 포함하지 않습니다. 파일명은 루트 [README](../README.md)의 실행 방법에 있습니다.

`processed/`에는 프로젝트가 재실행에 사용하는 셀별 초기 피처(`cell_features.csv`, `early_current_features.csv`), 고정 분할(`batch1_split_plan.csv`), 라벨·입력 정책(`modeling_plan.json`)과 EDA용 중간 데이터가 있습니다. 가공 데이터만으로 세 노트북을 실행할 수 있습니다.

원본에서 다시 생성하려면 프로젝트 루트에서 `python src/eda.py --root .`와 `python src/strategy_checks.py --raw-dir data/raw`를 실행하세요. `src/preprocess.py`는 저장된 가공 데이터를 결합하고 분할·정책 중복을 검증하며, `src/features.py`는 허용된 초기 입력만 선택합니다.

수명(`cycle_life`)과 종료 시점 QD, 전체 기록의 knee 등 미래 정보는 모델 입력에 사용하지 않습니다. 배치별 제외 기준과 한계는 루트 README와 보고서를 참고하세요.

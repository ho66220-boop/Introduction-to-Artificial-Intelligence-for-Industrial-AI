# 학생 학업 성취도 및 중도 포기 예측 프로젝트

Team 11의 NOVA50101 Introduction to Artificial Intelligence for Industrial AI 최종 프로젝트 작업 공간입니다. Kaggle/UCI의 `Predict students' dropout and academic success` 데이터셋을 사용하여 전통적 머신러닝 모델과 MLP 딥러닝 모델을 비교합니다.

## 프로젝트 목표

학생 데이터를 바탕으로 `Dropout`, `Enrolled`, `Graduate` 세 클래스를 예측하고, 어떤 요인이 학업 성공과 중도 포기 예측에 중요한지 분석합니다. 단순히 성능만 비교하는 것이 아니라, 과제 가이드라인에 맞춰 EDA, 전처리 실험, ML/DL 비교, 오류 분석, 한계점 및 향후 연구까지 포함합니다.

## 실행 순서

1. 데이터 CSV를 `data/raw/` 아래에 넣습니다. 현재 프로젝트에는 UCI 공개 배포본이 이미 들어 있습니다.

2. 데이터 전처리를 실행합니다.

```powershell
python scripts/prepare_data.py
```

3. 기본 머신러닝 모델을 학습합니다.

```powershell
python scripts/train_ml.py
```

4. MLP 모델을 학습합니다.

```powershell
python scripts/train_mlp.py
```

5. GridSearchCV로 머신러닝 모델을 튜닝합니다.

```powershell
python scripts/tune_ml.py
```

6. PCA/SMOTE 비교 실험을 실행하고 결과를 요약합니다.

```powershell
python scripts/run_experiments.py
python scripts/summarize_metrics.py
```

7. Confusion matrix와 feature importance 분석 파일을 생성합니다.

```powershell
python scripts/analyze_results.py
```

8. 최종 보고서, 발표 자료, 가이드라인 맞춤 산출물을 생성합니다.

```powershell
python scripts/generate_final_assets.py
python scripts/generate_guideline_assets.py
```

VS Code의 `.venv`를 사용할 경우 다음처럼 실행할 수 있습니다.

```powershell
& "c:/Users/Administrator/project/AI team/.venv/Scripts/python.exe" "c:/Users/Administrator/project/AI team/scripts/generate_guideline_assets.py"
```

## 주요 산출물

- `data/processed/`: 전처리된 train/test split 및 전처리 파이프라인
- `models/`: 학습된 모델 파일
- `reports/metrics/summary.csv`: 전체 모델 성능 요약
- `reports/metrics/confusion_matrices.csv`: 모델별 confusion matrix
- `reports/metrics/random_forest_tuned_feature_importance.csv`: 변수 중요도
- `reports/final_report_guideline_aligned.md`: 과제 가이드라인에 맞춘 최종 보고서
- `reports/final_report_guideline_aligned.html`: HTML 보고서
- `reports/activity_appendix.md`: 기존/수정 활동 계획 및 팀원 기여도
- `reports/presentation_slides.html`: 발표 슬라이드 HTML

## 데이터셋

- Dataset name: `Predict students' dropout and academic success`
- Kaggle URL: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention
- UCI mirror: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- Dataset size: 4,424 rows / 35 input columns + target
- Target classes: `Dropout`, `Enrolled`, `Graduate`

## 사용 모델

- Logistic Regression
- SVM with RBF kernel
- Random Forest
- MLP trained with SGD/backpropagation
- KMeans for unsupervised analysis

## 현재 최고 결과

현재 실험에서 가장 좋은 모델은 `random_forest_smote`입니다.

```text
macro_f1 = 0.7035
accuracy = 0.7559
weighted_f1 = 0.7582
```

MLP는 `macro_f1 = 0.6792`, `accuracy = 0.7424`를 기록했습니다. 작은 tabular dataset에서는 Random Forest 계열 모델이 MLP보다 더 안정적으로 좋은 성능을 보였습니다.

## Public Code Baseline 관련 주의점

프로포절에 적은 Kaggle public code 3개는 외부 baseline/reference로 사용했습니다. 다만 Kaggle notebook의 정확한 split과 metric을 동일하게 재현한 것은 아니므로, 본 보고서에서는 “public code보다 절대적으로 우수하다”고 주장하지 않습니다. 대신 동일한 전처리와 train/test split 안에서 ML/DL 모델을 공정하게 비교하고, PCA/SMOTE ablation과 오류 분석을 추가한 점을 originality로 정리했습니다.

## 코드 구성

- `src/data.py`: 데이터 로드, target 추론, 전처리, split 저장
- `src/evaluation.py`: 공통 평가 지표 계산
- `scripts/prepare_data.py`: 전처리 실행
- `scripts/train_ml.py`: 기본 ML 모델 학습
- `scripts/train_mlp.py`: MLP 학습
- `scripts/tune_ml.py`: GridSearchCV 튜닝
- `scripts/run_experiments.py`: PCA/SMOTE 실험
- `scripts/analyze_results.py`: confusion matrix 및 feature importance 생성
- `scripts/generate_final_assets.py`: 일반 최종 보고서/슬라이드 생성
- `scripts/generate_guideline_assets.py`: 과제 가이드라인 맞춤 보고서/appendix 생성

## 참고

target column은 `Target`, `target`, `Status`, `status` 등의 이름에서 자동 추론합니다. 자동 추론이 실패하면 다음처럼 직접 지정할 수 있습니다.

```powershell
python scripts/prepare_data.py --target Target
```

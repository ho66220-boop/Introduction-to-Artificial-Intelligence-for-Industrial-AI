# 과제 가이드라인 맞춤 최종 보고서

## 프로젝트 개요

- 과목: NOVA50101 Introduction to Artificial Intelligence for Industrial AI
- 팀: Team 11
- 팀원: 박재우, 염지훈, 오형우
- 프로젝트 유형: Application project
- 프로젝트 제목: 학업 성취도 및 중도 포기 예측을 위한 머신러닝과 딥러닝 모델의 비교 연구
- 데이터셋: Predict Students' Dropout and Academic Success from Kaggle/UCI

본 프로젝트는 학생이 중도 포기할지, 계속 재학 중일지, 졸업할지를 예측하는 교육 AI 문제를 다룬다. 과제 가이드라인에 맞춰 전통적 머신러닝, 딥러닝, 전처리 실험, 오류 분석, 결과 해석을 함께 수행했다.

## Public Code Baseline 및 Originality

프로포절에서는 다음 Kaggle public code를 외부 baseline/reference로 제시했다.

- ML Algorithms Usage and Prediction: https://www.kaggle.com/code/sunayanagawde/ml-algorithms-usage-and-prediction  
  본 프로젝트에서의 역할: 전통적 ML workflow와 비교를 위한 public-code baseline.
- Dropout Graduate Analysis: https://www.kaggle.com/code/satyaprakashshukl/droput-graduate-analysis  
  본 프로젝트에서의 역할: EDA 중심의 dropout/graduate 분석을 참고하기 위한 public-code baseline.
- Student Dropout Analysis for School Education: https://www.kaggle.com/code/jeevabharathis/student-dropout-analysis-for-school-education  
  본 프로젝트에서의 역할: 교육 분야 dropout 분석을 참고하기 위한 public-code baseline.

Kaggle notebook의 출력값과 split 방식은 정적 페이지에서 항상 동일하게 재현하기 어렵기 때문에, 본 보고서는 해당 public code보다 절대적으로 성능이 우수하다고 주장하지 않는다. 대신 이 public code들을 참고 baseline으로 삼고, 모든 모델을 동일한 전처리와 train/test protocol 안에서 비교하는 재현 가능한 파이프라인을 구축했다.

본 프로젝트의 originality는 다음 지점에 있다.

- 하나의 공통 전처리 파이프라인 아래에서 ML/DL 모델을 통제된 방식으로 비교
- PCA와 SMOTE ablation 실험 수행
- 전통적 ML 모델에 대한 GridSearchCV 튜닝
- SGD/backpropagation 기반 MLP 학습
- feature importance와 confusion matrix 기반 오류 분석
- KMeans 비지도 분석을 통해 자연 cluster와 target label의 정렬 정도 확인

## 데이터셋 및 EDA

데이터셋은 4,424개의 학생 record와 target을 포함한 36개의 raw column으로 구성되어 있다. target class는 `Dropout`, `Enrolled`, `Graduate` 세 가지이다.

EDA 결과 target class가 불균형하며, 학기별 학업 성과 변수들이 class별로 뚜렷한 차이를 보였다. 따라서 stratified split, class balancing 실험, macro F1 중심 평가가 필요하다고 판단했다.

## 방법론

전처리:

- 중복 record 제거
- stratified train/test split 적용
- 수치형 feature 표준화
- 범주형 feature one-hot encoding
- PCA와 SMOTE를 전처리 변형 실험으로 적용

모델:

- Logistic Regression
- SVM with RBF kernel
- Random Forest
- MLP trained with SGD/backpropagation
- KMeans for unsupervised analysis

평가 지표:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- Confusion matrix

class distribution이 불균형하고 `Enrolled` class가 다른 class보다 분류하기 어렵기 때문에, accuracy뿐 아니라 macro F1을 핵심 지표로 사용했다.

## 실험 결과

macro F1 기준 상위 결과는 다음과 같다.

| model                     | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
| ------------------------- | -------- | --------------- | ------------ | -------- | ----------- |
| random_forest_smote       | 0.7559   | 0.7082          | 0.7035       | 0.7035   | 0.7582      |
| random_forest_base        | 0.7514   | 0.7086          | 0.7040       | 0.7027   | 0.7557      |
| random_forest_tuned       | 0.7514   | 0.7086          | 0.7040       | 0.7027   | 0.7557      |
| random_forest             | 0.7627   | 0.7116          | 0.6941       | 0.7006   | 0.7584      |
| logistic_regression_smote | 0.7345   | 0.7055          | 0.7057       | 0.6960   | 0.7472      |
| logistic_regression_base  | 0.7299   | 0.7056          | 0.7055       | 0.6936   | 0.7443      |
| logistic_regression_tuned | 0.7299   | 0.7056          | 0.7055       | 0.6936   | 0.7443      |
| logistic_regression       | 0.7288   | 0.7049          | 0.7047       | 0.6927   | 0.7434      |
| mlp_sgd                   | 0.7424   | 0.6863          | 0.6746       | 0.6792   | 0.7401      |
| mlp_sgd_base              | 0.7424   | 0.6863          | 0.6746       | 0.6792   | 0.7401      |

가장 좋은 모델은 `random_forest_smote`이며, macro F1은 `0.7035`, accuracy는 `0.7559`이다. MLP baseline은 macro F1 `0.6792`, accuracy `0.7424`를 기록했다.

가장 강한 성능을 보인 모델은 SMOTE를 적용한 tree-based ensemble이었다. 이 결과는 데이터셋이 작고 tabular 형태라는 점에서 타당하다. Random Forest는 대규모 representation learning이 필요한 neural network보다 작은 tabular dataset에서 비선형 feature interaction을 효율적으로 포착할 수 있다.

## Ablation 분석

SMOTE는 Random Forest의 macro F1을 소폭 개선했지만, PCA는 전반적으로 성능을 낮췄다. 이는 차원 축소 과정에서 tabular feature가 가진 유용한 신호가 일부 손실되었을 가능성을 시사한다. 반면 SMOTE는 minority class 패턴을 조금 더 반영하도록 돕는 효과가 있었다.

## 비지도 학습 분석

target class 수와 동일하게 KMeans를 3개 cluster로 실행했다. 결과는 다음과 같다.

- Normalized Mutual Information: `0.1559`
- Adjusted Rand Index: `0.1519`
- Inertia: `102807.14`

비지도 cluster와 실제 label의 정렬 정도가 낮다는 점은, 세 target class가 단순한 거리 기반 clustering만으로 명확히 분리되지 않음을 보여준다. 따라서 supervised learning model이 필요하다는 해석이 가능하다.

## 변수 중요도 및 인사이트

Random Forest 기준 상위 feature는 다음과 같다.

1. Curricular units 2nd sem (approved): 0.1559
2. Curricular units 2nd sem (grade): 0.1121
3. Curricular units 1st sem (approved): 0.0939
4. Curricular units 1st sem (grade): 0.0666
5. Tuition fees up to date: 0.0506
6. Curricular units 2nd sem (evaluations): 0.0496
7. Curricular units 1st sem (evaluations): 0.0448
8. Age at enrollment: 0.0410
9. Admission grade: 0.0362
10. Course: 0.0320

가장 중요한 예측 변수는 학기별 이수/승인 과목 수와 성적이었다. 등록금 납부 여부도 중요한 변수로 나타났다. 이는 중도 포기와 졸업 여부가 입학 당시 정보뿐 아니라, 입학 후 학업 진행 상황과 강하게 연결되어 있음을 의미한다.

## 오류 분석

confusion matrix를 보면 `Enrolled` class가 가장 분류하기 어려웠다. 이는 `Enrolled`가 중간 상태이기 때문이다. 현재 재학 중인 학생은 이후 졸업할 수도 있고 중도 포기할 수도 있으므로, `Dropout` 및 `Graduate`와 decision boundary가 겹칠 수 있다.

## 한계점

- 데이터셋 규모가 딥러닝에 비해 작아 MLP가 representation learning의 장점을 충분히 활용하기 어렵다.
- 일부 중요 feature는 학기별 성과 변수이므로, 입학 직후의 매우 이른 시점에는 사용할 수 없을 수 있다.
- Kaggle public-code baseline은 reference로 사용했지만, 정적 notebook 페이지에서 정확한 metric을 동일하게 재현하지는 못했다.
- 모델은 과거 tabular data에 기반하므로, 제도 변화나 개인적 상황을 모두 반영하지 못한다.
- 예측 결과는 학생 지원을 위한 decision-support 도구로 사용해야 하며, 학생에 대한 최종 판단이나 불이익 부여에 사용되어서는 안 된다.

## 향후 연구

- Top public Kaggle notebook을 동일 split과 metric으로 local에서 재실행하여 직접 비교한다.
- 과목 범위가 허용된다면 XGBoost, LightGBM, CatBoost 등 gradient boosting model을 추가한다.
- 입학 시점 feature 또는 1학기 feature만 사용하는 early-warning model을 별도로 구축한다.
- dropout recall을 높이기 위해 calibration 및 threshold tuning을 수행한다.
- gender, scholarship status, international status 등에 대한 fairness analysis를 수행한다.
- 여러 학년도 데이터가 확보되면 temporal validation을 수행한다.

## 가이드라인 충족 여부

- Technical soundness: 공통 전처리 파이프라인, ML/DL 비교, tuning, ablation, EDA, 오류 분석 포함
- Limitation and future work: 한계점과 구체적인 향후 실험 방향 제시
- Activities: 기존/수정 활동 계획 및 팀원 기여도를 appendix로 제시

## 그림

- [class_distribution.png](figures/class_distribution.png)
- [key_feature_distributions.png](figures/key_feature_distributions.png)
- [model_performance.png](figures/model_performance.png)
- [experiment_comparison.png](figures/experiment_comparison.png)
- [feature_importance.png](figures/feature_importance.png)
- [best_confusion_matrix.png](figures/best_confusion_matrix.png)

## Appendix

[activity_appendix.md](activity_appendix.md)를 참고한다.

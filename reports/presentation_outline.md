# Presentation Outline

## Slide 1. Title
A Comparative Study of ML and DL for Predicting Student Success and Dropout

## Slide 2. Problem and Goal
- Goal: Predict Dropout, Enrolled, Graduate
- Practical value: early warning and student support

## Slide 3. Dataset
- 4,424 rows
- 36 input features after preprocessing
- Target classes: Dropout, Enrolled, Graduate

## Slide 4. Preprocessing
- Duplicate removal
- Stratified train/test split
- Standardization and one-hot encoding
- PCA and SMOTE comparison experiments

## Slide 5. Models
- Logistic Regression
- SVM RBF
- Random Forest
- MLP with SGD/backpropagation

## Slide 6. Performance
- Best model: random_forest_smote
- Macro F1: 0.7035
- Accuracy: 0.7559
- Show `model_performance.png`

## Slide 7. PCA and SMOTE
- SMOTE slightly improved Random Forest
- PCA generally reduced performance
- Show `experiment_comparison.png`

## Slide 8. Feature Importance
- Semester approved units and grades were strongest predictors
- Tuition fees up to date was also important
- Show `feature_importance.png`

## Slide 9. Error Analysis
- Enrolled class is hardest to classify
- It is an intermediate and unstable academic state
- Show `best_confusion_matrix.png`

## Slide 10. Conclusion
- Random Forest performed best for this small tabular dataset
- MLP was competitive but not superior
- Best use case: decision-support early-warning system

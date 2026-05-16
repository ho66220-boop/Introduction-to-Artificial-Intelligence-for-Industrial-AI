from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODELS_DIR, RANDOM_STATE
from src.data import load_prepared_data
from src.evaluation import evaluate_classifier, save_metrics


def main() -> None:
    data = load_prepared_data()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        "svm_rbf": SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        model.fit(data.x_train, data.y_train)
        metrics = evaluate_classifier(model, data.x_test, data.y_test, data.class_names)
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        save_metrics(name, metrics)
        print(f"{name}: macro_f1={metrics['macro_f1']:.4f}, accuracy={metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()

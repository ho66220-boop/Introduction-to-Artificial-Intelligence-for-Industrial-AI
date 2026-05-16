from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_DIR, MODELS_DIR, RANDOM_STATE
from src.data import load_prepared_data
from src.evaluation import evaluate_classifier, save_metrics


def write_cv_results(name: str, search: GridSearchCV) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METRICS_DIR / f"{name}_cv_results.csv"
    rows = []
    for index, params in enumerate(search.cv_results_["params"]):
        rows.append(
            {
                "model": name,
                "rank": search.cv_results_["rank_test_score"][index],
                "mean_test_macro_f1": search.cv_results_["mean_test_score"][index],
                "std_test_macro_f1": search.cv_results_["std_test_score"][index],
                "params": params,
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    data = load_prepared_data()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    searches = {
        "logistic_regression_tuned": GridSearchCV(
            LogisticRegression(max_iter=4000, class_weight="balanced", random_state=RANDOM_STATE),
            param_grid={
                "C": [0.1, 1.0, 3.0, 10.0],
                "solver": ["lbfgs"],
            },
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
        ),
        "svm_rbf_tuned": GridSearchCV(
            SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
            param_grid={
                "C": [0.5, 1.0, 3.0, 10.0],
                "gamma": ["scale", 0.03, 0.1],
            },
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
        ),
        "random_forest_tuned": GridSearchCV(
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            param_grid={
                "max_depth": [None, 8, 14],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
        ),
    }

    for name, search in searches.items():
        search.fit(data.x_train, data.y_train)
        best_model = search.best_estimator_
        metrics = evaluate_classifier(best_model, data.x_test, data.y_test, data.class_names)
        metrics["best_cv_macro_f1"] = search.best_score_
        metrics["best_params"] = search.best_params_
        joblib.dump(best_model, MODELS_DIR / f"{name}.joblib")
        save_metrics(name, metrics)
        write_cv_results(name, search)
        print(
            f"{name}: test_macro_f1={metrics['macro_f1']:.4f}, "
            f"cv_macro_f1={search.best_score_:.4f}, params={search.best_params_}"
        )


if __name__ == "__main__":
    main()


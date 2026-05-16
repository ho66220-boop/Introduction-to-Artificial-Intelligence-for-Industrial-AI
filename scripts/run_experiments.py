from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODELS_DIR, RANDOM_STATE
from src.data import prepare_dataset
from src.evaluation import evaluate_classifier, save_metrics


def train_models(experiment_name: str, use_pca: bool, use_smote: bool) -> None:
    data = prepare_dataset(use_pca=use_pca, apply_smote=use_smote)
    models = {
        "logistic_regression": LogisticRegression(
            C=10.0,
            max_iter=4000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=14,
            max_features="log2",
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "mlp_sgd": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="sgd",
            learning_rate="adaptive",
            learning_rate_init=0.01,
            momentum=0.9,
            batch_size=64,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=RANDOM_STATE,
        ),
    }

    for model_name, model in models.items():
        output_name = f"{model_name}_{experiment_name}"
        model.fit(data.x_train, data.y_train)
        metrics = evaluate_classifier(model, data.x_test, data.y_test, data.class_names)
        metrics["experiment"] = {
            "use_pca": use_pca,
            "use_smote": use_smote,
        }
        joblib.dump(model, MODELS_DIR / f"{output_name}.joblib")
        save_metrics(output_name, metrics)
        print(f"{output_name}: macro_f1={metrics['macro_f1']:.4f}, accuracy={metrics['accuracy']:.4f}")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    experiments = {
        "base": {"use_pca": False, "use_smote": False},
        "smote": {"use_pca": False, "use_smote": True},
        "pca": {"use_pca": True, "use_smote": False},
        "pca_smote": {"use_pca": True, "use_smote": True},
    }

    for name, options in experiments.items():
        train_models(name, **options)

    prepare_dataset(use_pca=False, apply_smote=False)


if __name__ == "__main__":
    main()


from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODELS_DIR, RANDOM_STATE
from src.data import load_prepared_data
from src.evaluation import evaluate_classifier, save_metrics


def main() -> None:
    data = load_prepared_data()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = MLPClassifier(
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
    )

    model.fit(data.x_train, data.y_train)
    metrics = evaluate_classifier(model, data.x_test, data.y_test, data.class_names)
    joblib.dump(model, MODELS_DIR / "mlp_sgd.joblib")
    save_metrics("mlp_sgd", metrics)
    print(f"mlp_sgd: macro_f1={metrics['macro_f1']:.4f}, accuracy={metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()

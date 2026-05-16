from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare student success/dropout dataset.")
    parser.add_argument("--input", help="Path to input CSV. Defaults to first CSV in data/raw.")
    parser.add_argument("--target", help="Target column name. Defaults to inferred target.")
    parser.add_argument("--pca", action="store_true", help="Apply PCA after scaling/encoding.")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE to the training split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_dataset(
        input_path=args.input,
        target=args.target,
        use_pca=args.pca,
        apply_smote=args.smote,
    )
    print("Prepared dataset")
    print(f"train shape: {prepared.x_train.shape}")
    print(f"test shape: {prepared.x_test.shape}")
    print(f"classes: {prepared.class_names}")


if __name__ == "__main__":
    main()

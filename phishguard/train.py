"""
PhishGuard :: train
---------------------
CLI entry point to train (or retrain) the base classifier from a CSV
dataset and save it to models/.

Usage:
    python3 train.py --data data/phishing_dataset.csv
    python3 train.py --data path/to/your_real_dataset.csv
"""

import argparse
from src.model import train_from_csv


def main():
    ap = argparse.ArgumentParser(description="Train the PhishGuard classifier")
    ap.add_argument("--data", default="data/phishing_dataset.csv", help="path to training CSV")
    args = ap.parse_args()

    model, metrics = train_from_csv(args.data)
    print("Training complete. Holdout metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nTop global feature importances:")
    for item in model.global_importances(10):
        print(f"  {item['feature']}: {item['importance']}")
    print("\nModel saved to models/phishguard_model.joblib")


if __name__ == "__main__":
    main()

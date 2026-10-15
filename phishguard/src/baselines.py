"""
PhishGuard :: baseline models for the comparison report (v3)
--------------------------------------------------------------
The comparison report shows how "the way phishing has traditionally been
detected" would have judged the SAME URL. The traditional pattern (see the
literature survey, Section III-A and Gap 2): a single static classifier on
URL/HTML features that outputs a bare binary verdict with a 0.5 cutoff, with
no calibration and no explanation.

So we train the same four baselines used in evaluate.py (Logistic Regression,
Decision Tree, RBF-SVM, Gradient Boosting) on exactly the same training split
(80/20, random_state=42) as PhishGuard's own model, and ask each of them for
a verdict on the scanned URL's feature vector.

The baseline models are saved to models/baselines.joblib by train.py. If the
file is missing (or was saved by an incompatible scikit-learn), they are
re-trained once from data/phishing_dataset.csv the first time they are needed.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .features import FEATURE_NAMES
from .model import MODEL_DIR, _dataframe_from_csv, build_feature_matrix

BASELINE_PATH = os.path.join(MODEL_DIR, "baselines.joblib")
DEFAULT_DATA = os.path.join(os.path.dirname(MODEL_DIR), "data", "phishing_dataset.csv")

# Which survey category each baseline stands for.
BASELINE_NOTES = {
    "LogisticRegression": "Linear classifier (Classical ML)",
    "DecisionTree": "Single decision tree (Classical ML)",
    "SVM (RBF)": "Kernel SVM (Classical ML)",
    "GradientBoosting": "Boosted trees (Classical ML / ensemble)",
}


def _new_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "SVM (RBF)": SVC(probability=True, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }


def train_baselines(csv_path: str = DEFAULT_DATA, save: bool = True) -> dict:
    df = _dataframe_from_csv(csv_path)
    X = build_feature_matrix(df)
    y = df["label"].astype(int)
    Xtr, _Xte, ytr, _yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = _new_models()
    for clf in models.values():
        clf.fit(Xtr, ytr)
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(models, BASELINE_PATH)
    return models


_cache = {"models": None}


def get_baselines() -> dict:
    """Load saved baselines, or train them once if unavailable."""
    if _cache["models"] is not None:
        return _cache["models"]
    models = None
    if os.path.exists(BASELINE_PATH):
        try:
            models = joblib.load(BASELINE_PATH)
        except Exception:
            models = None
    if models is None and os.path.exists(DEFAULT_DATA):
        models = train_baselines(DEFAULT_DATA, save=True)
    _cache["models"] = models or {}
    return _cache["models"]


def compare_on_vector(feat_vector: list) -> list:
    """Ask every baseline for a binary verdict on one feature vector.
    Returns [{model, note, probability, verdict}] (empty list if no baselines)."""
    models = get_baselines()
    if not models:
        return []
    arr = pd.DataFrame([feat_vector], columns=FEATURE_NAMES)
    out = []
    for name, clf in models.items():
        p = float(clf.predict_proba(arr)[0][1])
        out.append({
            "model": name,
            "note": BASELINE_NOTES.get(name, ""),
            "probability": round(p, 4),
            "verdict": "PHISHING" if p >= 0.5 else "LEGITIMATE",
        })
    return out

"""
PhishGuard :: model
--------------------
Lightweight, real-time-capable classifier (RandomForest -- no GPU, no
transformer, sub-millisecond inference) plus a prediction-level
explainability layer (Direction 3 of the survey / Gap 2).

Explainability method
----------------------
`shap` is not available in this environment (no internet to install it),
so explanations are produced with a from-scratch, honestly-documented
local ablation method rather than pretending to be SHAP:

For a given prediction, each feature's contribution is estimated as the
change in predicted phishing-probability when that feature is reset to
its "background" value (the training-set median for that feature) while
holding all other features fixed:

    contribution(feature_i) = P(phish | x) - P(phish | x with x_i -> median_i)

This is a single-feature marginal-contribution (ablation) estimate, a
simplified, non-combinatorial special case of the Shapley-value idea
(it does not average over feature orderings/coalitions the way exact
SHAP does). It is cheap (num_features extra forward passes) and gives
a per-prediction, per-feature signed attribution that sums to roughly
the total swing in predicted probability -- which is what an analyst
needs to see "why did this URL get flagged".
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from .features import FEATURE_NAMES, extract_features, features_to_vector

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "phishguard_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "phishguard_meta.json")


def _dataframe_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain at least 'url' and 'label' columns")
    if "html" not in df.columns:
        df["html"] = ""
    df["html"] = df["html"].fillna("")
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        feats = extract_features(str(r["url"]), str(r.get("html", "")), base_url=str(r["url"]))
        rows.append(feats)
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


class PhishGuardModel:
    def __init__(self, clf=None, background=None):
        self.clf = clf or RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=2,
            n_jobs=-1, random_state=42, class_weight="balanced",
        )
        self.background = background  # median feature values, used for explanations
        self.metrics_ = {}

    def fit(self, X: pd.DataFrame, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.clf.fit(X_train, y_train)
        preds = self.clf.predict(X_test)
        self.metrics_ = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }
        self.background = X.median(numeric_only=True)
        return self.metrics_

    def predict_proba_vector(self, feat_vector: list) -> float:
        arr = pd.DataFrame([feat_vector], columns=FEATURE_NAMES)
        return float(self.clf.predict_proba(arr)[0][1])

    def explain(self, feats: dict, top_k: int = 8) -> list:
        """Local ablation-based explanation. Returns a list of
        {feature, value, contribution} sorted by |contribution| desc."""
        if self.background is None:
            raise RuntimeError("Model has no background stats -- fit() first")

        base_vec = [feats[name] for name in FEATURE_NAMES]
        base_p = self.predict_proba_vector(base_vec)

        contributions = []
        for i, name in enumerate(FEATURE_NAMES):
            ablated = list(base_vec)
            ablated[i] = float(self.background.get(name, 0))
            ablated_p = self.predict_proba_vector(ablated)
            contributions.append({
                "feature": name,
                "value": feats[name],
                "contribution": round(base_p - ablated_p, 4),
            })

        contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
        return {
            "predicted_probability": round(base_p, 4),
            "top_features": contributions[:top_k],
        }

    def global_importances(self, top_k: int = 15) -> list:
        importances = getattr(self.clf, "feature_importances_", None)
        if importances is None:
            return []
        pairs = sorted(zip(FEATURE_NAMES, importances), key=lambda p: p[1], reverse=True)
        return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs[:top_k]]

    def save(self, model_path=MODEL_PATH, meta_path=META_PATH):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({"clf": self.clf, "background": self.background}, model_path)
        with open(meta_path, "w") as f:
            json.dump({"metrics": self.metrics_, "feature_names": FEATURE_NAMES}, f, indent=2)

    @classmethod
    def load(cls, model_path=MODEL_PATH, meta_path=META_PATH):
        payload = joblib.load(model_path)
        m = cls(clf=payload["clf"], background=payload["background"])
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                m.metrics_ = json.load(f).get("metrics", {})
        return m


def train_from_csv(csv_path: str, save=True) -> PhishGuardModel:
    df = _dataframe_from_csv(csv_path)
    X = build_feature_matrix(df)
    y = df["label"].astype(int)
    model = PhishGuardModel()
    metrics = model.fit(X, y)
    if save:
        model.save()
    return model, metrics

"""
PhishGuard :: evaluation suite
--------------------------------
Produces every figure and table a conference paper needs, from whatever
dataset you point it at. Runs fully offline on the current CSV; drop in a
real dataset (same url/html/label columns) and re-run to get real numbers.

Outputs (into ./results/):
  - baseline_comparison.csv      RF vs LogReg / DecisionTree / SVM / GradBoost
  - baseline_comparison.png      bar chart of the above
  - roc_curves.png               ROC + AUC for every model
  - confusion_matrix.png         confusion matrix for PhishGuard's RF
  - feature_ablation.csv/.png    URL-only vs HTML-only vs hybrid feature sets
  - calibration_curve.png        reliability of calibrated probabilities
  - drift_recovery.png           streaming accuracy + drift trigger + recovery
  - metrics_summary.json         machine-readable dump of everything

Usage:
  python3 evaluate.py --data data/phishing_dataset.csv
  python3 evaluate.py --data data/real_phishing.csv --skip-drift
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, roc_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve

from src.features import (
    FEATURE_NAMES, URL_FEATURE_NAMES, HTML_FEATURE_NAMES, extract_features,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_matrix(csv_path):
    df = pd.read_csv(csv_path)
    if "html" not in df.columns:
        df["html"] = ""
    df["html"] = df["html"].fillna("")
    rows = [extract_features(str(r.url), str(r.html), base_url=str(r.url))
            for r in df.itertuples()]
    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = df["label"].astype(int)
    return X, y


def _metrics(y_true, pred, proba):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, pred), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4) if len(set(y_true)) > 1 else None,
        "fpr": round(fp / max(fp + tn, 1), 4),
    }


def baseline_comparison(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "SVM (RBF)": SVC(probability=True, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "PhishGuard (RandomForest)": RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=2,
            n_jobs=-1, random_state=42, class_weight="balanced"),
    }
    rows, roc_data = [], {}
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = _metrics(yte, pred, proba)
        m["model"] = name
        rows.append(m)
        fpr, tpr, _ = roc_curve(yte, proba)
        roc_data[name] = (fpr, tpr, m["roc_auc"])
    table = pd.DataFrame(rows)[["model", "accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "fpr"]]
    table.to_csv(os.path.join(RESULTS_DIR, "baseline_comparison.csv"), index=False)

    # bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    xpos = np.arange(len(table))
    width = 0.2
    for i, met in enumerate(metrics_to_plot):
        ax.bar(xpos + i * width, table[met], width, label=met)
    ax.set_xticks(xpos + 1.5 * width)
    ax.set_xticklabels(table["model"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison on Held-Out Test Set")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "baseline_comparison.png"), dpi=150)
    plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)

    # confusion matrix for RF
    rf = models["PhishGuard (RandomForest)"]
    pred = rf.predict(Xte)
    cm = confusion_matrix(yte, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Phish"]); ax.set_yticklabels(["Legit", "Phish"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("PhishGuard Confusion Matrix")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # calibration curve
    rf_proba = rf.predict_proba(Xte)[:, 1]
    try:
        frac_pos, mean_pred = calibration_curve(yte, rf_proba, n_bins=10)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(mean_pred, frac_pos, "o-", label="RandomForest")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration (Reliability) Curve")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "calibration_curve.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  (calibration curve skipped: {e})")

    return table.to_dict(orient="records")


def feature_ablation(X, y):
    """The headline ablation: prove hybrid > URL-only and > HTML-only.
    This directly substantiates the survey's Gap 5 claim with our own data."""
    families = {
        "URL-only (24 feats)": URL_FEATURE_NAMES,
        "HTML-only (16 feats)": HTML_FEATURE_NAMES,
        "Hybrid (all feats)": FEATURE_NAMES,
    }
    Xtr_full, Xte_full, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    rows = []
    for name, cols in families.items():
        clf = RandomForestClassifier(n_estimators=200, max_depth=14,
                                     min_samples_leaf=2, n_jobs=-1,
                                     random_state=42, class_weight="balanced")
        clf.fit(Xtr_full[cols], ytr)
        proba = clf.predict_proba(Xte_full[cols])[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = _metrics(yte, pred, proba)
        m["feature_set"] = name
        rows.append(m)
    table = pd.DataFrame(rows)[["feature_set", "accuracy", "precision", "recall", "f1", "fpr"]]
    table.to_csv(os.path.join(RESULTS_DIR, "feature_ablation.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    xpos = np.arange(len(table))
    for i, met in enumerate(["accuracy", "f1"]):
        ax.bar(xpos + i * 0.35, table[met], 0.35, label=met)
    ax.set_xticks(xpos + 0.175)
    ax.set_xticklabels(table["feature_set"], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Feature-Family Ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "feature_ablation.png"), dpi=150)
    plt.close(fig)
    return table.to_dict(orient="records")


def drift_recovery_plot():
    """Run the streaming sim twice (stable + injected drift) and plot the
    rolling accuracy, the drift trigger, and the post-retrain recovery.
    This is the figure no reviewed paper in the survey has."""
    from src.streaming_simulation import run_stream

    def rolling_acc(log, window=30):
        correct = [1 if r["pred"] == r["label"] else 0 for r in log]
        out = []
        for i in range(len(correct)):
            lo = max(0, i - window + 1)
            out.append(np.mean(correct[lo:i + 1]))
        return out

    # retrain from the base dataset first so both runs start clean
    from src.model import train_from_csv
    train_from_csv(os.path.join(os.path.dirname(__file__), "data", "phishing_dataset.csv"))
    log_stable, _, _ = run_stream(n_steps=400, simulate_drift=False, seed=7)

    train_from_csv(os.path.join(os.path.dirname(__file__), "data", "phishing_dataset.csv"))
    log_drift, _, _ = run_stream(n_steps=400, drift_at=200, simulate_drift=True, seed=7)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rolling_acc(log_stable), label="Stable stream (no drift)", color="#2c7fb8")
    ax.plot(rolling_acc(log_drift), label="Injected tactic shift @ step 200", color="#d95f0e")
    ax.axvline(200, color="gray", linestyle=":", label="Drift injected")
    for r in log_drift:
        if r.get("action") == "RETRAINED":
            ax.axvline(r["step"], color="green", linestyle="--", alpha=0.7)
            ax.text(r["step"] + 2, 0.45, "retrain", rotation=90, color="green", fontsize=8)
    ax.set_xlabel("Stream step")
    ax.set_ylabel("Rolling accuracy (window=30)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Self-Healing: Drift Detection and Recovery")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "drift_recovery.png"), dpi=150)
    plt.close(fig)
    retrains = [r["step"] for r in log_drift if r.get("action") == "RETRAINED"]
    return {"retrain_steps": retrains, "stable_final_acc": round(rolling_acc(log_stable)[-1], 4),
            "drift_final_acc": round(rolling_acc(log_drift)[-1], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phishing_dataset.csv")
    ap.add_argument("--skip-drift", action="store_true")
    ap.add_argument("--skip-adversarial", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading + extracting features...")
    X, y = load_matrix(args.data)
    print(f"  {len(X)} samples, {X.shape[1]} features, "
          f"{int(y.sum())} phishing / {int((1-y).sum())} legit")

    summary = {}
    print("Baseline comparison + ROC + confusion + calibration...")
    summary["baselines"] = baseline_comparison(X, y)
    print("Feature-family ablation...")
    summary["feature_ablation"] = feature_ablation(X, y)
    if not args.skip_drift:
        print("Drift-recovery simulation...")
        summary["drift"] = drift_recovery_plot()
    if not args.skip_adversarial:
        print("Adversarial robustness (URL-surface)...")
        try:
            from src.adversarial import run as adv_run, plot as adv_plot
            base_detect, overall, table, n_caught = adv_run(
                args.data, n=400, threshold=0.45, url_only=True)
            table.to_csv(os.path.join(RESULTS_DIR, "adversarial_robustness.csv"), index=False)
            adv_plot(table, overall, os.path.join(RESULTS_DIR, "adversarial_robustness.png"))
            summary["adversarial"] = {"baseline_detection": base_detect,
                                      "mean_robustness": overall,
                                      "per_mutation": table.to_dict(orient="records")}
        except Exception as e:
            print(f"  (adversarial arm skipped: {e})")

    with open(os.path.join(RESULTS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. All artifacts written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

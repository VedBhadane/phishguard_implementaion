"""
PhishGuard :: streaming simulation
------------------------------------
Simulates a real-time stream of incoming URLs (optionally with a
mid-stream "phishing tactics evolve" shift), scores each one with the
current model, feeds the outcome to the DDM drift detector, and
automatically retrains on a buffered recent window when DRIFT fires.

This demonstrates the "self-healing" property end to end without
needing a live production feed: run this after training the base
model to see drift detection + automatic retraining in action.
"""

import argparse
import random
import pandas as pd

from .model import PhishGuardModel, build_feature_matrix, MODEL_PATH, META_PATH
from .drift import DDMDriftDetector
from .features import FEATURE_NAMES

try:
    # generator lives in ../data relative to this file
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    from generate_dataset import make_legit_url, make_legit_html, make_phish_url, make_phish_html
except ImportError:
    make_legit_url = make_legit_html = make_phish_url = make_phish_html = None


def _shifted_phish_example():
    """A mutated phishing style meant to simulate 'tactics evolving' --
    used only in --simulate-drift mode to demonstrate the detector firing."""
    import string
    host = "".join(random.choice(string.ascii_lowercase) for _ in range(12)) + ".shop"
    url = f"https://{host}/secure-update/verify"  # note: now HTTPS, no IP, no obvious hyphen-spoof
    html = f"""<html><head><title>Account Center</title></head>
<body><form action="/api/session" method="post">
<input type="text" name="u"><input type="password" name="p"></form>
<p>Please confirm your identity to continue.</p>
</body></html>"""
    return url, html, 1


def run_stream(n_steps=400, drift_at=200, simulate_drift=True, retrain_buffer=150, seed=7):
    random.seed(seed)
    model = PhishGuardModel.load(MODEL_PATH, META_PATH)
    detector = DDMDriftDetector()

    buffer_rows = []
    log = []

    for step in range(1, n_steps + 1):
        use_shifted = simulate_drift and step > drift_at
        if use_shifted and make_phish_url is not None:
            if random.random() < 0.6:
                url, html, label = _shifted_phish_example()
            else:
                url, html, label = make_legit_url(), "", 0
        else:
            if random.random() < 0.5 and make_phish_url is not None:
                url, html, label = make_phish_url(), make_phish_html(""), 1
            else:
                url, html, label = make_legit_url(), "", 0

        from .features import extract_features
        feats = extract_features(url, html, base_url=url)
        vec = [feats[n] for n in FEATURE_NAMES]
        proba = model.predict_proba_vector(vec)
        pred = int(proba >= 0.5)
        correct = pred == label

        state = detector.update(correct)
        buffer_rows.append({"url": url, "html": html, "label": label})
        if len(buffer_rows) > retrain_buffer:
            buffer_rows.pop(0)

        log.append({
            "step": step, "pred": pred, "label": label, "proba": round(proba, 3),
            "drift_state": state,
        })

        if state == "DRIFT":
            df = pd.DataFrame(buffer_rows)
            if df["label"].nunique() > 1:
                X = build_feature_matrix(df)
                y = df["label"].astype(int)
                model.fit(X, y)
                model.save()
                detector.reset()
                log[-1]["action"] = "RETRAINED"

    return log, detector.stats(), model.metrics_


def main():
    ap = argparse.ArgumentParser(description="PhishGuard streaming + drift simulation")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--drift-at", type=int, default=200)
    ap.add_argument("--no-drift", action="store_true", help="disable the simulated tactic shift")
    args = ap.parse_args()

    log, stats, metrics = run_stream(
        n_steps=args.steps, drift_at=args.drift_at, simulate_drift=not args.no_drift
    )

    retrains = [r for r in log if r.get("action") == "RETRAINED"]
    correct = sum(1 for r in log if r["pred"] == r["label"])
    print(f"Stream length: {len(log)}")
    print(f"Overall streaming accuracy: {correct/len(log):.3f}")
    print(f"Retrain events triggered by drift detector: {len(retrains)}")
    for r in retrains:
        print(f"  -> retrained at step {r['step']}")
    print(f"Final detector stats: {stats}")
    print(f"Final model holdout metrics after last retrain: {metrics}")


if __name__ == "__main__":
    main()

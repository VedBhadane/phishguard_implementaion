"""
PhishGuard :: web app
-----------------------
Flask app exposing:
  GET  /                 - simple form UI (paste a URL, optionally HTML)
  POST /                 - form submit -> renders result + explanation
  POST /api/scan         - JSON REST API: {"url": "...", "html": "..."}
  GET  /api/health       - liveness check + current model metrics

Run with:  python3 run_app.py   (from the project root)
"""

import os
import time
from flask import Flask, request, jsonify, render_template

from .model import PhishGuardModel, MODEL_PATH, META_PATH
from .features import extract_features, FEATURE_NAMES
from .llm_explainer import explain_in_words

app = Flask(__name__)

_model_cache = {"model": None, "loaded_at": None}


def get_model():
    if _model_cache["model"] is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "No trained model found. Run `python3 train.py` first."
            )
        _model_cache["model"] = PhishGuardModel.load(MODEL_PATH, META_PATH)
        _model_cache["loaded_at"] = time.time()
    return _model_cache["model"]


def score(url: str, html: str = ""):
    model = get_model()
    feats = extract_features(url, html, base_url=url)
    explanation = model.explain(feats)
    proba = explanation["predicted_probability"]
    verdict = model.verdict_band(proba)
    confidence = proba if proba >= 0.5 else 1 - proba
    rationale = explain_in_words(verdict, proba, explanation["top_features"], url)
    return {
        "url": url,
        "verdict": verdict,
        "phishing_probability": proba,
        "confidence": round(confidence, 4),
        "rationale": rationale,
        "top_features": explanation["top_features"],
        "used_html": bool(html.strip()),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    url_val = ""
    html_val = ""
    if request.method == "POST":
        url_val = request.form.get("url", "").strip()
        html_val = request.form.get("html", "").strip()
        try:
            if not url_val:
                raise ValueError("Please enter a URL.")
            result = score(url_val, html_val)
        except Exception as e:
            error = str(e)
    return render_template("index.html", result=result, error=error, url_val=url_val, html_val=html_val)


@app.route("/api/scan", methods=["POST"])
def api_scan():
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or "").strip()
    html = (payload.get("html") or "").strip()
    if not url:
        return jsonify({"error": "field 'url' is required"}), 400
    try:
        result = score(url, html)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    try:
        model = get_model()
        return jsonify({
            "status": "ok",
            "model_metrics": model.metrics_,
            "num_features": len(FEATURE_NAMES),
        })
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

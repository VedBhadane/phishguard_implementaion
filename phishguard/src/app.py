"""
PhishGuard :: web app
-----------------------
Flask app exposing:
  GET  /                 - simple form UI (enter a URL; the page HTML is fetched automatically)
  POST /                 - form submit -> renders result + explanation + comparison report
  POST /api/scan         - JSON REST API: {"url": "..."}  (optional "html" overrides the fetch;
                           optional "report": true adds the comparison report to the response)
  GET  /report           - project-level comparison report (HTML)
  GET  /report.pdf       - project-level comparison report (PDF)
  GET  /report/scan/<id>.pdf - comparison report for one scan (PDF)
  GET  /api/health       - liveness check + current model metrics

Set PHISHGUARD_FETCH=0 to turn automatic page fetching off (URL-only scoring).

Run with:  python3 run_app.py   (from the project root)
"""

import os
import time
import uuid
from collections import OrderedDict

from flask import Flask, request, jsonify, render_template, Response, abort

from .model import PhishGuardModel, MODEL_PATH, META_PATH
from .features import extract_features, features_to_vector, FEATURE_NAMES
from .llm_explainer import explain_in_words
from . import fetcher
from .baselines import compare_on_vector
from .report import build_report, render_pdf

app = Flask(__name__)

_model_cache = {"model": None, "loaded_at": None}


def get_model():
    if _model_cache["model"] is None:
        data_csv = os.path.join(os.path.dirname(os.path.dirname(MODEL_PATH)), "data", "phishing_dataset.csv")
        model = None
        if os.path.exists(MODEL_PATH):
            try:
                model = PhishGuardModel.load(MODEL_PATH, META_PATH)
            except Exception as e:
                # The saved file was written by a different pandas/numpy/scikit-learn
                # than the one installed here. Rebuild it once from the dataset.
                print(f"[PhishGuard] Saved model could not be loaded ({type(e).__name__}: {e}).")
                model = None
        if model is None:
            if not os.path.exists(data_csv):
                raise RuntimeError("No usable trained model found and no dataset to retrain from. "
                                   "Run `python train.py` first.")
            print("[PhishGuard] Retraining the model from data/phishing_dataset.csv "
                  "(one-time, about a minute)...")
            from .model import train_from_csv
            model, _ = train_from_csv(data_csv)
        _model_cache["model"] = model
        _model_cache["loaded_at"] = time.time()
    return _model_cache["model"]


_SCAN_CACHE = OrderedDict()   # scan_id -> result, newest last (for the per-scan PDF)
_SCAN_CACHE_MAX = 100


def _fetch_enabled() -> bool:
    return os.getenv("PHISHGUARD_FETCH", "1") not in ("0", "false", "False", "no")


def _remember(result: dict) -> str:
    scan_id = uuid.uuid4().hex[:12]
    _SCAN_CACHE[scan_id] = result
    while len(_SCAN_CACHE) > _SCAN_CACHE_MAX:
        _SCAN_CACHE.popitem(last=False)
    return scan_id


def _fetch_info(fetch_result):
    """Small, JSON-safe summary of what the fetch step did."""
    if fetch_result is None:
        return {"attempted": False, "status": "disabled",
                "message": fetcher.reason_text("disabled")}
    return {
        "attempted": True,
        "status": fetch_result["reason"],
        "message": fetcher.reason_text(fetch_result["reason"]),
        "final_url": fetch_result.get("final_url", ""),
        "http_status": fetch_result.get("status_code"),
        "bytes": fetch_result.get("bytes", 0),
        "truncated": fetch_result.get("truncated", False),
        "redirects": fetch_result.get("redirects", 0),
        "elapsed_ms": fetch_result.get("elapsed_ms", 0),
    }


def score(url: str, html: str = None):
    """Score one URL.

    html=None  -> fetch the page behind the URL automatically (falls back to
                  URL-only scoring if the fetch fails; see src/fetcher.py).
    html=str   -> use the supplied HTML and do not fetch (API callers/tests).
    """
    model = get_model()

    fetch_result = None
    if html is None or not html.strip():
        html = ""
        if _fetch_enabled():
            fetch_result = fetcher.fetch_html(url)
            if fetch_result["ok"]:
                html = fetch_result["html"]
        fetch = _fetch_info(fetch_result)
    else:
        fetch = {"attempted": False, "status": "provided",
                 "message": fetcher.reason_text("provided")}

    feats = extract_features(url, html, base_url=url)
    explanation = model.explain(feats)
    proba = explanation["predicted_probability"]
    verdict = model.verdict_band(proba)
    confidence = proba if proba >= 0.5 else 1 - proba
    rationale = explain_in_words(verdict, proba, explanation["top_features"], url)
    used_html = bool(html.strip())

    result = {
        "url": url,
        "verdict": verdict,
        "phishing_probability": proba,
        "confidence": round(confidence, 4),
        "rationale": rationale,
        "top_features": explanation["top_features"],
        "used_html": used_html,
        "fetch": fetch,
    }

    # --- inputs for the comparison report -------------------------------
    vec = features_to_vector(feats)
    try:
        result["baselines"] = compare_on_vector(vec)
    except Exception:
        result["baselines"] = []

    if used_html:
        url_only = extract_features(url, "", base_url=url)
        p_url = model.predict_proba_vector(features_to_vector(url_only))
        result["signal_comparison"] = {
            "url_only_probability": round(p_url, 4),
            "url_only_verdict": model.verdict_band(p_url),
            "hybrid_probability": proba,
            "hybrid_verdict": verdict,
            "note": ("Adding the fetched page moved the phishing probability from "
                     f"{p_url:.1%} to {proba:.1%}."),
        }
    else:
        result["signal_comparison"] = None

    result["scan_id"] = _remember(result)
    return result


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    report = None
    error = None
    url_val = ""
    if request.method == "POST":
        url_val = request.form.get("url", "").strip()
        try:
            if not url_val:
                raise ValueError("Please enter a URL.")
            result = score(url_val)          # page HTML is fetched automatically
            report = build_report(result)
        except Exception as e:
            error = str(e)
    return render_template("index.html", result=result, report=report, error=error, url_val=url_val)


@app.route("/api/scan", methods=["POST"])
def api_scan():
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or "").strip()
    html = payload.get("html")                      # optional override; normally omitted
    html = html.strip() if isinstance(html, str) else None
    if not url:
        return jsonify({"error": "field 'url' is required"}), 400
    try:
        result = score(url, html)
        out = {k: v for k, v in result.items() if k not in ("baselines", "signal_comparison")}
        if payload.get("report"):
            out["report"] = build_report(result)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _pdf_response(report, filename):
    return Response(render_pdf(report), mimetype="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="{filename}"'})


@app.route("/report")
def project_report():
    return render_template("report.html", report=build_report(None))


@app.route("/report.pdf")
def project_report_pdf():
    return _pdf_response(build_report(None), "phishguard_comparison_report.pdf")


@app.route("/report/scan/<scan_id>.pdf")
def scan_report_pdf(scan_id):
    result = _SCAN_CACHE.get(scan_id)
    if result is None:
        abort(404)
    return _pdf_response(build_report(result), f"phishguard_scan_report_{scan_id}.pdf")


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

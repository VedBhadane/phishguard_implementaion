# PhishGuard v3 changes

## New
- `src/fetcher.py`: fetches the page behind a URL (http/https only, ports 80/443/8080/8443,
  every resolved IP must be public, redirects re-validated per hop (max 3), 4 s timeout,
  1.5 MB cap, HTML content types only, no JavaScript, no cookies, TLS not verified on purpose).
  Never raises; failure falls back to URL-only scoring.
- `src/baselines.py`: Logistic Regression, Decision Tree, RBF-SVM and Gradient Boosting trained on
  the same split as PhishGuard; used to show how traditional models judge the same URL.
  Saved to `models/baselines.joblib` (re-trained automatically if missing/incompatible).
- `src/literature_data.py`: content from the literature survey (capability comparison, six gaps,
  reported accuracies, limitations), with the survey's reference numbers.
- `src/report.py`: builds the comparison report (per scan or project-level) and renders it as PDF.
- `templates/report.html`, `templates/_report_sections.html`: report page and shared sections.
- `tests/test_v3.py`: 12 tests (SSRF blocks, redirect checks, size cap, fallback, report, PDFs).

## Changed
- `src/app.py`: `score()` fetches HTML automatically; returns `fetch` details, `scan_id`;
  new routes `/report`, `/report.pdf`, `/report/scan/<id>.pdf`; `/api/scan` accepts `"report": true`.
- `src/templates/index.html`: URL-only form (paste box removed), shows the rationale, fetch status,
  report section and PDF download.
- `src/static/style.css`: styles for the report; SUSPICIOUS colour.
- `train.py`: also trains the baselines. `requirements.txt`: + requests, reportlab.
- README / POSITIONING wording: "no third-party lookups" instead of "no network calls".

## Fixed
- Web page displayed SUSPICIOUS verdicts as "LEGITIMATE" (template only handled two bands).
- The one-sentence rationale was computed but never shown on the page.

## Not changed
- Model, features, calibration, thresholds (0.35 / 0.65), drift monitor, evaluation numbers.

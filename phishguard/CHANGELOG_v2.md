# PhishGuard v2 — What Changed vs. the Original Repo

This is the exact list of what to commit so your code matches the v2
implementation paper. Nothing here needs the internet or an API key; it all
runs offline. Files are grouped by "edited" vs "new".

## Files EDITED (replace your originals with these)

### `src/features.py`  (40 → 48 features)
- Added 8 evasion-resistant URL features: `suspicious_tld`, `has_punycode`,
  `brand_in_subdomain`, `brand_lookalike`, `vowel_consonant_anomaly`,
  `hex_or_encoded_chars`, `num_query_params`, `path_depth`.
- Added supporting lists (suspicious TLDs, target brands, homoglyph map) and
  helper functions.
- Fixed the `has_double_slash_redirect` bug (was false-positiving on long
  legitimate URLs).

### `src/model.py`  (calibration + richer metrics)
- Wrapped the RandomForest in isotonic `CalibratedClassifierCV` so reported
  probabilities are meaningful.
- `fit()` now records MCC, ROC-AUC, FPR, and the full confusion matrix, not
  just accuracy/precision/recall/F1.
- Added `verdict_band()` → PHISHING / SUSPICIOUS / LEGITIMATE (three bands).
- `save()`/`load()` now persist the calibrator alongside the model.

### `src/app.py`  (three-band verdict + rationale)
- `score()` now returns a three-band verdict, a calibrated probability, a
  confidence value, and a plain-English `rationale`.

### `data/generate_dataset.py`  (realistic-difficulty option)
- Added `--label-noise` flag to inject class overlap so the benchmark can
  actually discriminate between models (without it, everything scores 100%).

### `requirements.txt`
- Added `matplotlib>=3.7` (needed by evaluate.py).

## Files NEW (add these)

### `evaluate.py`  (the evaluation suite)
One command → produces every paper figure/table into `results/`:
baseline comparison, ROC curves, confusion matrix, calibration curve,
feature-family ablation, and the drift-recovery plot.

### `src/llm_explainer.py`  (plain-English rationale, Direction 3)
Turns the numeric ablation into a one-line analyst sentence. Deterministic
template offline; upgrades to a live LLM if `ANTHROPIC_API_KEY` is set.

### `src/adversarial.py`  (adversarial robustness harness, Track A.4)
Mutates phishing URLs the way a real attacker would (homoglyph, subdomain
padding, URL-encoding, etc.) and reports a **robustness score** — the metric
that separates a security detector from a static benchmark number.

## How to run everything (all offline)

```bash
pip install -r requirements.txt
python3 data/generate_dataset.py --n 2500 --seed 42 --label-noise 0.12
python3 train.py --data data/phishing_dataset.csv       # trains + prints metrics
python3 evaluate.py --data data/phishing_dataset.csv     # all figures → results/
python3 -m src.adversarial                               # robustness score
python3 run_app.py                                       # web app + API
```

## How to commit to GitHub

The simplest path: replace your local repo's `phishguard/` folder contents
with this folder's contents, then:

```bash
git add -A
git commit -m "v2: evasion-resistant features, calibration, 3-band verdict,
                evaluation suite, LLM rationale, adversarial harness"
git push
```

After this, the repo matches the v2 implementation paper. The ONLY remaining
gap between paper and code is the dataset: both currently use synthetic data.
Swapping in a real dataset (PhishTank/UCI) is a drop-in — `evaluate.py` takes
any CSV with url/html/label columns — and is the last step to make every
number real.
```

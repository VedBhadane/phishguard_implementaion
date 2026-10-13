# PhishGuard — Enhancements (v2)

What changed from the original repo, why, and how to run it. Every change is
offline-safe (no internet, no API key required) and backward-compatible.

## Summary of changes

| # | Enhancement | Files touched | Paper/patent value |
|---|---|---|---|
| 1 | **8 evasion-resistant features** + fixed `//` bug | `src/features.py` | Methodology; robustness claim |
| 2 | **Calibrated probabilities + 3-band verdict** | `src/model.py`, `src/app.py` | Honesty; reliability-curve figure |
| 3 | **Richer metrics** (MCC, ROC-AUC, FPR, confusion) | `src/model.py` | Fills the paper's results table |
| 4 | **Full evaluation suite** (baselines, ROC, ablation, drift plot) | `evaluate.py` (new) | Every figure the paper needs |
| 5 | **LLM-assisted plain-English rationale** | `src/llm_explainer.py` (new), `src/app.py` | Survey Direction 3; novelty |
| 6 | **Harder synthetic data option** (`--label-noise`) | `data/generate_dataset.py` | Meaningful benchmark spread |
| 7 | **Adversarial robustness harness** | `src/adversarial.py` (new) | Robustness score + figure; security credibility |

## 1. Evasion-resistant features

Added to `features.py` (total is now 48, up from 40):
`suspicious_tld`, `has_punycode`, `brand_in_subdomain`, `brand_lookalike`,
`vowel_consonant_anomaly`, `hex_or_encoded_chars`, `num_query_params`,
`path_depth`. Also fixed `has_double_slash_redirect`, which previously
misfired on long legitimate paths.

These catch tricks the baseline feature set misses: `paypal.com.evil.ru`
(brand-in-subdomain), `amaz0n.tk` (homoglyph look-alike), `xn--pple-43d.com`
(punycode), `djfkslqwmn.top` (machine-generated).

## 2. Calibration + three-band verdict

RandomForest votes are poorly calibrated, so a raw "0.8" doesn't mean 80%.
`model.fit()` now wraps the classifier in isotonic `CalibratedClassifierCV`
on the held-out split. Verdicts are PHISHING (≥0.65) / SUSPICIOUS / LEGITIMATE
(≤0.35) instead of a hard 0.5 cut. The calibrator is persisted with the model.

## 3. Richer metrics

`model.fit()` now records MCC, ROC-AUC, FPR, and the full confusion matrix
alongside accuracy/precision/recall/F1. These populate the comparison table
reviewers expect (the reference paper reports exactly these).

## 4. Evaluation suite — `evaluate.py`

One command produces every artifact into `results/`:

```bash
python3 evaluate.py --data data/phishing_dataset.csv
```

- `baseline_comparison.csv/.png` — RF vs LogisticRegression / DecisionTree /
  SVM / GradientBoosting on identical splits
- `roc_curves.png` — ROC + AUC for all models
- `confusion_matrix.png`, `calibration_curve.png`
- `feature_ablation.csv/.png` — URL-only vs HTML-only vs hybrid
- `drift_recovery.png` — **the differentiator figure**: rolling accuracy on a
  stable stream vs an injected tactic shift, with the retrain point marked
- `metrics_summary.json` — machine-readable dump

## 5. LLM-assisted explanation — `src/llm_explainer.py`

Converts the numeric ablation into one analyst-readable sentence. Fully
deterministic offline (template path); if `ANTHROPIC_API_KEY` is set it calls
a model to phrase it, falling back to the template on any error. The numeric
attribution remains the source of truth — the LLM only verbalizes it.

Example output:
> Flagged as phishing (91%) mainly because a form submits data to an external
> domain; the host is a raw IP address; the page collects a password.

## 6. Harder benchmark option

`generate_dataset.py --label-noise 0.12` flips 12% of labels to create
realistic class overlap. Without it, the rule-generated classes are perfectly
separable and *every* model scores 100%, which tells you nothing. With noise,
models spread apart (e.g. DecisionTree ~82% vs others ~89%) and the benchmark
becomes meaningful. This is a stand-in for a real dataset, not a replacement.

## 7. Adversarial robustness harness — `src/adversarial.py`

Phishing is adversarial: attackers mutate URLs until a detector stops flagging
them. This harness measures that directly. It takes phishing examples the model
already catches, applies nine realistic evasion mutations (homoglyph swaps,
added subdomains, URL-encoding, benign-word padding, case-mixing, ports,
hyphenation, TLD swaps, path-lengthening), re-scores each, and reports a
**robustness score** = fraction of caught phish still caught after mutation.

```bash
# URL+HTML (HTML signals dominate, so robustness is very high)
python3 -m src.adversarial --data data/phishing_dataset.csv --n 400

# URL-only — the realistic attacker setting, where evasion actually bites
python3 -m src.adversarial --data data/phishing_dataset.csv --n 400 --threshold 0.45 --url-only
```

On the synthetic URL-only run, mean robustness is ~0.85: the model is strong
against homoglyph and TLD tricks (the evasion-resistant features doing their
job, ~0.91 retention) but weaker against path-lengthening and benign-padding
(~0.53–0.70), which is an honest finding that points to future hardening.
Reporting "X% accurate AND retains Y% under adversarial mutation" is far more
credible to a security reviewer than accuracy alone. The harness is also wired
into `evaluate.py` (skip with `--skip-adversarial`) and writes
`results/adversarial_robustness.{png,csv}`.

---

## How to run everything

```bash
pip install -r requirements.txt

# generate a realistic-difficulty synthetic dataset
python3 data/generate_dataset.py --n 2500 --seed 42 --label-noise 0.12

# train (prints the full metrics block)
python3 train.py --data data/phishing_dataset.csv

# produce all paper figures + tables
python3 evaluate.py --data data/phishing_dataset.csv

# run the web app (three-band verdict + rationale)
python3 run_app.py            # http://localhost:5000

# drift / self-healing demo
python3 -m src.streaming_simulation --steps 400 --drift-at 200
```

## The one thing still needed: a real dataset

Every result here is on synthetic data. Because `train.py` and `evaluate.py`
are dataset-agnostic (any CSV with `url` / `html` / `label`), swapping in
PhishTank + Tranco/UCI is a drop-in:

```bash
python3 evaluate.py --data data/real_phishing.csv
```

That single step converts every figure from proof-of-concept to a real,
publishable result — and is the highest-priority next action.

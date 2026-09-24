# PhishGuard — Competitive Positioning & How to Stand Out

This document maps what comparable phishing-detection projects and papers
actually do, where they stop, and exactly which of those stopping points
PhishGuard can turn into its differentiators. It is written to guide both
the implementation paper and the patent's novelty argument.

---

## 1. What everyone else builds (the crowded baseline)

Surveying open-source phishing-detection projects and the recent literature,
the overwhelming majority converge on the *same* shape:

- **Feature set:** URL lexical features (length, dots, hyphens, IP-as-host,
  `@` symbol, subdomain count, HTTPS flag). Many add WHOIS/domain-age, which
  requires a network call.
- **Model:** a single static classifier — Random Forest, XGBoost, or a
  Decision Tree — trained once on UCI/Kaggle/PhishTank and frozen.
- **Interface:** a Flask app or a Chrome extension that returns a binary
  "phishing / legitimate" verdict.
- **Evaluation:** one accuracy number, sometimes a confusion matrix.

Representative examples: gangeshbaskerr/Phishing-Website-Detection (17
features, PhishTank, RF/XGBoost), Trieuh2/ml-url-phishing-classifier (RF on
the Hannousse 11,430-URL dataset), natanim-kemal/hook (RF on 100k URLs +
Chrome extension), addievo/phishingDetection (Flask + RF). They are
competent, but interchangeable: **static model, binary output, URL-only,
single accuracy score.**

The academic frontier (BiLSTM+attention, GNNs, LLM classifiers) pushes
accuracy higher but inherits the same three structural weaknesses your own
survey named: no drift adaptation, black-box outputs, and API/GPU dependency.

---

## 2. The four gaps nobody closes *together*

From the survey and confirmed against current work, these remain open:

| Gap | State of the field | Who partially addresses it |
|---|---|---|
| **Concept drift** | Almost every project trains once and never adapts. Drift-adaptive work exists in *fraud/intrusion* streaming (ROSFD with ADWIN/DDM, adaptive random forests) but is rarely applied to phishing URLs specifically. | Ejaz et al. (continual learning) — the lone phishing-specific example |
| **Prediction-level explainability** | Global feature importance is common; *per-URL* explanations are rare, and analyst-readable ones rarer still. | XAI papers use SHAP but don't deploy it live |
| **API independence** | Many rely on WHOIS/Safe Browsing/Alexa, adding seconds of latency. | A minority go URL-only, but then lose HTML signal |
| **Real-time lightweight deployment** | Deep models need GPUs; few ship an actual operational artifact. | natanim-kemal/hook (extension), Gowda et al. (browser) |

**The whitespace is the intersection.** No project in the survey — and none
found in the open-source scan — combines *all four* in one system. That
intersection is PhishGuard's thesis and should be stated verbatim as the
novelty claim in the patent.

---

## 3. PhishGuard's five concrete differentiators

These are things the project now *does* that the baseline projects do not.
Each is defensible in the paper and demonstrable in a demo.

### D1. Self-healing via live drift detection (the headline)
A DDM monitor watches live prediction error and triggers retraining only on
confirmed distribution shift — not on a schedule. Almost no phishing project
has this, and the **drift-recovery plot** (`results/drift_recovery.png`) is a
figure the reference XSS paper and the surveyed papers simply don't have.
*This is the single strongest thing to lead with.*

### D2. API-independent hybrid features (URL + HTML, zero network calls)
40+ features spanning URL lexical *and* HTML structural signals, computed with
no WHOIS/Safe Browsing/Alexa lookups. This gives the HTML-side coverage that
URL-only projects lack, while keeping sub-millisecond, browser-speed latency
that WHOIS-dependent projects can't match.

### D3. Evasion-resistant feature engineering (v2)
Beyond the standard lexical features, PhishGuard adds punycode/homograph
detection, brand-in-wrong-position spoofing (`paypal.com.evil.ru`), homoglyph
look-alikes (`amaz0n`, `paypa1`), suspicious-TLD flags, and a
machine-generated-domain heuristic. Most baseline projects miss all of these
and are trivially evaded by them. Each is a concrete methodology bullet.

### D4. Per-prediction explanation in plain English
Numeric ablation attribution (a documented Shapley approximation) feeds a
one-line analyst rationale: *"Flagged (91%) mainly because a form submits to
an external domain; the host is a raw IP; the page collects a password."*
Deterministic and offline by default; upgrades to a live LLM sentence when an
API key is present (survey Direction 3). Baseline projects output a bare label.

### D5. Calibrated, three-band verdicts
Instead of a hard 0.5 cutoff, probabilities are isotonic-calibrated and mapped
to PHISHING / SUSPICIOUS / LEGITIMATE. This is more honest near the boundary
and yields a reliability-curve figure. Baseline projects report raw,
uncalibrated votes.

---

## 4. How to *say* it (framing for paper & patent)

- **Don't** claim highest accuracy — that's a losing race against BiLSTM/GNN
  papers and it's not your contribution.
- **Do** claim the *integration*: "the first phishing detector to unify
  drift-adaptive retraining, API-independent hybrid features, per-prediction
  explainability, and lightweight real-time deployment in a single system."
- **Do** frame the RandomForest choice as deliberate: sub-millisecond, no-GPU
  inference is a *requirement* of real-time browser deployment, not a
  limitation. Switching to a heavy deep model would contradict your own
  survey's Gap 6.
- **Patent novelty hook:** the self-healing loop (D1) + API-independence (D2)
  is the combination most likely to be genuinely non-obvious. Lead the claims
  there.

---

## 5. Roadmap to widen the moat further

Ordered by effort-to-impact for *after* the papers are submitted:

1. **Real dataset** (highest priority, low effort): swap PhishTank + Tranco/UCI
   into `evaluate.py`. Turns every figure from proof-of-concept into a real
   result and populates the paper's numbers. *Blocks nothing else — do first.*
2. **Browser extension** (medium effort, high visibility): a thin extension
   calling `/api/scan` puts PhishGuard in the "1 of 53 with real deployment"
   category your survey highlights.
3. **Live drift on real traffic** (medium): persist scans to SQLite and run the
   DDM monitor on accumulated real predictions, not just the simulation.
4. **Temporal benchmark** (higher): evaluate on time-stamped phishing data to
   directly measure drift over deployment windows — closes survey Gap 4, which
   no proposed direction fully covered.
5. **LLM explanation study** (optional): with an API key, compare template vs
   LLM rationales for analyst usefulness — a small user-study section.

---

## 6. One-paragraph elevator pitch

> Existing phishing detectors are static, opaque, and often network-dependent:
> they train once, output a bare verdict, and degrade silently as attackers
> evolve. PhishGuard is a self-healing detector that monitors its own live
> accuracy and retrains automatically on confirmed concept drift, extracts 40+
> URL and HTML features with zero external API calls for browser-speed
> inference, hardens those features against punycode/homoglyph/brand-spoofing
> evasion, and explains every verdict in one analyst-readable sentence backed
> by calibrated three-band confidence. Its contribution is not a higher
> accuracy number but the integration of drift-adaptation, API-independence,
> explainability, and lightweight deployment that no prior system combines.

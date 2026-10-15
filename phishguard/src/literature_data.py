"""
PhishGuard :: literature-survey content used by the comparison report
-----------------------------------------------------------------------
Everything here is taken from the team's literature survey
("A Systematic Literature Survey on Machine Learning-Based Phishing
Detection: Methods, Datasets, and Research Gaps", 53 reviewed papers).
Reference numbers [n] are the survey's own reference numbers.

Nothing in this file is measured by PhishGuard. It is what the surveyed
papers reported or what the survey concluded about them. The report labels
it that way so it is never confused with PhishGuard's own measured numbers.
"""

SURVEY_TITLE = ("A Systematic Literature Survey on Machine Learning-Based "
                "Phishing Detection: Methods, Datasets, and Research Gaps")

# --- Capability comparison: what each family of prior approaches offers --------
# Values: "Yes", "Partial", "No", "Mixed". Basis = survey Sections III, V, VII.
CAPABILITY_COLUMNS = [
    "Adapts to concept drift",
    "Per-prediction explanation",
    "No third-party lookups",
    "Light real-time deployment",
    "URL + HTML features",
]

CAPABILITY_ROWS = [
    {
        "approach": "Classical ML and ensembles",
        "survey_category": "Category 1, 15 papers, [1]-[15]",
        "cells": ["No", "Partial", "Mixed", "Partial", "Partial"],
        "basis": ("Static datasets, no drift handling, explainability limited to aggregate "
                  "feature importance [38][40][41]. Only [7] demonstrates a deployed browser "
                  "extension. Some systems depend on WHOIS/API lookups [10][13][17][18][21]. [10] combines URL, HTML and domain features; URL-only systems are common [5][6][7][11]."),
    },
    {
        "approach": "Deep learning (CNN, LSTM, GRU, BERT)",
        "survey_category": "Category 2, 13 papers, [16]-[28]",
        "cells": ["No", "No", "Mixed", "No", "Partial"],
        "basis": ("All reviewed models are static and black-box [41]; computational overhead "
                  "(GPU, latency) makes real-time deployment challenging [40][42]. URL-only deep models are common [17][18][25]."),
    },
    {
        "approach": "Feature engineering and XAI (incl. LLM + SHAP)",
        "survey_category": "Category 3, 8 papers, [29]-[36]",
        "cells": ["No", "Yes", "Mixed", "Partial", "Partial"],
        "basis": ("Explainability is the focus (SHAP, LLM rationale), but none of the papers "
                  "combines it with concept-drift detection; LLM pipelines are computationally "
                  "expensive (e.g. [36])."),
    },
    {
        "approach": "Drift, graph (GNN) and LLM-based approaches",
        "survey_category": "Category 4, [38]-[53]",
        "cells": ["Partial", "Partial", "Mixed", "Partial", "Partial"],
        "basis": ("Drift-aware work [43][44][46][47] exists, but only [43] targets phishing "
                  "directly. GNNs need full page loading [48]; multi-agent LLM inference is very "
                  "expensive [53]."),
    },
]

# PhishGuard's own row. The cells describe what the code does; the status
# column keeps the honesty that matters for a research prototype.
PHISHGUARD_ROW = {
    "approach": "PhishGuard (this work)",
    "survey_category": "Proposed system",
    "cells": ["Yes", "Yes", "Yes", "Yes", "Yes"],
    "basis": ("DDM drift monitor with retraining; per-feature ablation attribution plus a "
              "plain-language rationale; no WHOIS/Safe Browsing/rank lookups (the only network "
              "call is one request to the page being checked); RandomForest, no GPU; 48 URL + "
              "HTML features."),
}

CAPABILITY_FOOTNOTE = (
    "Yes = supported, Partial = supported by some papers in the category or only in part, "
    "Mixed = some papers use third-party lookups and some do not, No = not supported by the "
    "reviewed papers. PhishGuard's drift handling has been demonstrated in a streaming "
    "simulation, not on live traffic.")

# --- Research gaps from survey Table XIII (Section V) --------------------------
# Each row: gap, who is affected (with refs), PhishGuard response, honest status.
GAPS = [
    {
        "gap": "Gap 1: No concept-drift detection or adaptation",
        "evidence": ("Static ML/DL models [1][2][3][4][6][7][8][11][16][17][18][19][20][21]"
                     "[22][24][25] are trained once on historical data."),
        "response": "DDM drift monitor watches live errors and triggers retraining.",
        "status": "Implemented; shown in streaming simulation only",
    },
    {
        "gap": "Gap 2: No prediction-level explainability",
        "evidence": ("Ensemble and deep approaches [1][4][11][16][17][22][24][48][49] are "
                     "black-box for individual decisions."),
        "response": "Single-feature ablation attribution for every verdict, plus a one-sentence rationale.",
        "status": "Implemented (ablation estimate, not exact SHAP)",
    },
    {
        "gap": "Gap 3: Third-party API dependence",
        "evidence": ("WHOIS/Alexa/Safe Browsing-dependent systems [10][13][17][18][21][40][42] "
                     "add seconds of latency, cost and availability risk."),
        "response": "No third-party lookups; features come from the URL and the page itself.",
        "status": "Implemented; page fetch adds one request to the target site",
    },
    {
        "gap": "Gap 4: Stale datasets, no temporal validation",
        "evidence": ("UCI/Kaggle/PhishTank snapshots [1][4][6][7][8][11][17][19][24][25] are "
                     "not time-stamped."),
        "response": "Not yet addressed; needs a real, time-stamped dataset.",
        "status": "Open (future work)",
    },
    {
        "gap": "Gap 5: URL-only feature scope",
        "evidence": "URL-only approaches [5][6][7][11][17][18][25] miss innocuous URLs with malicious HTML.",
        "response": "Hybrid URL + HTML features; HTML is now fetched automatically from the URL.",
        "status": "Implemented",
    },
    {
        "gap": "Gap 6: Deep-learning compute cost",
        "evidence": "GPU-dependent models [16][17][19][20][22] are incompatible with browser-speed detection.",
        "response": "RandomForest: CPU-only, no GPU, no transformer.",
        "status": "Implemented",
    },
]

# --- Reported accuracies of best methods per category (survey Section IV) ------
# As reported by each source paper, on that paper's own dataset.
REPORTED_ACCURACY = [
    ("Classical ML", "CNN + Random Forest ensemble, Yang et al.", "[11]", "99.35%"),
    ("Classical ML", "Stacking ensemble (GA-optimized), Al-Mekhlafi et al.", "[1]", "97.16%"),
    ("Classical ML", "SVM, Salahdine et al.", "[9]", "94.5%"),
    ("Deep learning", "RasNet-TCMA-MPNet, Do et al.", "[22]", "99.71%"),
    ("Deep learning", "GloVe + GRU + AFSA, Hilal et al.", "[17]", "99.60%"),
    ("Deep learning", "CNN, Alshingiti et al.", "[16]", "99.2%"),
    ("Deep learning", "CNN-LSTM, Ujah-Ogbuagu et al.", "[19]", "98.9% (11,055-instance dataset)"),
    ("Deep learning", "BERT + CNN, Gupta et al.", "[20]", "97.5%"),
    ("Feature engineering / XAI", "CatBoost + SHAP, Fajar et al.", "[2]", "~100% (235K-URL dataset)"),
    ("GNN", "PhishGNN, Bilot et al.", "[48]", "99.7%"),
    ("GNN", "PhishDet (LRCN + GCN), Ariyadasa et al.", "[49]", "96.42%"),
    ("Concept drift", "MOMBD-CDD Bi-LSTM, Mansour et al.", "[46]", "97.45% (KDD dataset)"),
]

ACCURACY_CAVEAT = (
    "These figures were reported by each source paper on its own dataset, so they cannot be "
    "compared directly with PhishGuard's score on its synthetic evaluation set. Direct, "
    "like-for-like comparison is only valid against the baseline models run on the same data.")

LIMITATIONS = [
    ("Training and evaluation data", "The classifier is trained and evaluated on a synthetic "
     "dataset. Scores on real phishing pages will differ, and real-world accuracy has not yet "
     "been measured. Replacing it with PhishTank, Tranco or UCI data is planned."),
    ("Page fetching", "Many phishing pages are taken down within hours or show a harmless page "
     "to automated visitors. When the page cannot be fetched, PhishGuard scores from the URL alone and "
     "says so (used_html = false)."),
    ("Explanations", "Feature contributions are a single-feature ablation estimate, a simplified "
     "special case of the Shapley idea, not exact SHAP."),
    ("Drift handling", "Drift detection and retraining have been demonstrated in a simulated "
     "stream, not on live traffic."),
]

"""
PhishGuard :: adversarial robustness test harness (Track A.4)
-------------------------------------------------------------
Phishing is an adversarial problem: attackers mutate their URLs/pages until
a detector stops flagging them. A detector that is only measured on a static
test set has never been tested against the thing that actually breaks it in
production. This module does that test.

It takes phishing examples the model ALREADY detects correctly, applies a
family of realistic evasion mutations to each, re-scores the mutated version,
and measures how many are STILL detected. The headline output is a
**robustness score** = fraction of originally-caught phish that survive the
mutation and are still caught.

  robustness(family) = (# still flagged phishing after mutation)
                       / (# flagged phishing before mutation)

A robust detector keeps this near 1.0; a fragile one collapses. Reporting
this alongside accuracy is what separates a credible security detector from a
number on a static benchmark.

Mutations implemented (each mimics a real attacker tactic):
  homoglyph        o->0, l->1, e->3 ... (look-alike domains)
  add_subdomain    prepend a benign-looking label (secure., login., www.)
  url_encode       percent-encode path characters to hide keywords
  benign_padding   inject innocent words into the path
  case_mix         randomize character case in the URL
  add_port         append an explicit :8080-style port
  hyphenate        insert hyphens into the hostname
  tld_swap         move the payload onto a different suspicious TLD
  path_lengthen    append long random path segments

Usage:
  python3 -m src.adversarial --data data/phishing_dataset.csv --n 400
  # writes results/adversarial_robustness.{png,csv}
"""

import argparse
import os
import random
import string
from urllib.parse import quote, urlparse

import pandas as pd

from .features import extract_features, FEATURE_NAMES
from .model import PhishGuardModel, MODEL_PATH, META_PATH

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

_HOMOGLYPH = {"o": "0", "l": "1", "e": "3", "a": "@", "s": "5", "i": "1", "t": "7"}
_BENIGN_WORDS = ["home", "welcome", "index", "main", "content", "page", "info", "help"]
_BENIGN_SUBS = ["secure", "login", "www", "account", "signin", "my", "portal"]
_ALT_TLDS = [".xyz", ".top", ".club", ".online", ".site", ".info"]


def _split_url(url):
    if "://" not in url:
        url = "http://" + url
    p = urlparse(url)
    return p.scheme, (p.hostname or ""), (":" + str(p.port) if p.port else ""), (p.path or ""), \
        ("?" + p.query if p.query else "")


def m_homoglyph(url):
    scheme, host, port, path, query = _split_url(url)
    # substitute in the registered-domain label only
    labels = host.split(".")
    if labels:
        core_idx = max(range(len(labels)), key=lambda i: len(labels[i]))
        labels[core_idx] = "".join(_HOMOGLYPH.get(c, c) if random.random() < 0.5 else c
                                   for c in labels[core_idx])
        host = ".".join(labels)
    return f"{scheme}://{host}{port}{path}{query}"


def m_add_subdomain(url):
    scheme, host, port, path, query = _split_url(url)
    return f"{scheme}://{random.choice(_BENIGN_SUBS)}.{host}{port}{path}{query}"


def m_url_encode(url):
    scheme, host, port, path, query = _split_url(url)
    encoded = quote(path, safe="/")
    return f"{scheme}://{host}{port}{encoded}{query}"


def m_benign_padding(url):
    scheme, host, port, path, query = _split_url(url)
    pad = "/" + "/".join(random.sample(_BENIGN_WORDS, 2))
    return f"{scheme}://{host}{port}{pad}{path}{query}"


def m_case_mix(url):
    scheme, host, port, path, query = _split_url(url)
    host = "".join(c.upper() if random.random() < 0.4 else c for c in host)
    return f"{scheme}://{host}{port}{path}{query}"


def m_add_port(url):
    scheme, host, port, path, query = _split_url(url)
    return f"{scheme}://{host}:{random.choice(['8080','8443','8000'])}{path}{query}"


def m_hyphenate(url):
    scheme, host, port, path, query = _split_url(url)
    labels = host.split(".")
    if labels and len(labels[0]) > 3:
        i = len(labels[0]) // 2
        labels[0] = labels[0][:i] + "-" + labels[0][i:]
        host = ".".join(labels)
    return f"{scheme}://{host}{port}{path}{query}"


def m_tld_swap(url):
    scheme, host, port, path, query = _split_url(url)
    labels = host.split(".")
    if len(labels) >= 2:
        labels[-1] = random.choice(_ALT_TLDS).lstrip(".")
        host = ".".join(labels)
    return f"{scheme}://{host}{port}{path}{query}"


def m_path_lengthen(url):
    scheme, host, port, path, query = _split_url(url)
    extra = "/" + "/".join("".join(random.choice(string.ascii_lowercase) for _ in range(6))
                           for _ in range(3))
    return f"{scheme}://{host}{port}{path}{extra}{query}"


MUTATIONS = {
    "homoglyph": m_homoglyph,
    "add_subdomain": m_add_subdomain,
    "url_encode": m_url_encode,
    "benign_padding": m_benign_padding,
    "case_mix": m_case_mix,
    "add_port": m_add_port,
    "hyphenate": m_hyphenate,
    "tld_swap": m_tld_swap,
    "path_lengthen": m_path_lengthen,
}


def _score(model, url, html):
    feats = extract_features(url, html, base_url=url)
    vec = [feats[n] for n in FEATURE_NAMES]
    return model.predict_proba_vector(vec)


def run(data_path, n=400, threshold=0.5, seed=7, url_only=False):
    random.seed(seed)
    df = pd.read_csv(data_path)
    if "html" not in df.columns:
        df["html"] = ""
    df["html"] = df["html"].fillna("")
    phish = df[df["label"] == 1].sample(min(n, (df["label"] == 1).sum()), random_state=seed)

    model = PhishGuardModel.load(MODEL_PATH, META_PATH)

    # url_only: drop the HTML so the URL surface is the sole signal. This is
    # the realistic attacker setting (they control the URL; a URL scanner may
    # not have rendered HTML) and it is where evasion actually bites.
    def html_of(r):
        return "" if url_only else str(r.html)

    # baseline: which phishing examples does the model catch to begin with?
    caught = []
    for r in phish.itertuples():
        if _score(model, str(r.url), html_of(r)) >= threshold:
            caught.append((str(r.url), html_of(r)))
    base_detect = len(caught) / len(phish) if len(phish) else 0.0

    rows = []
    for name, fn in MUTATIONS.items():
        still = 0
        for url, html in caught:
            try:
                mutated = fn(url)
            except Exception:
                mutated = url
            # keep the HTML the same (attacker changes the URL, page content stays)
            if _score(model, mutated, html) >= threshold:
                still += 1
        retention = still / len(caught) if caught else 0.0
        rows.append({"mutation": name, "retention": round(retention, 4),
                     "evaded": round(1 - retention, 4)})

    table = pd.DataFrame(rows).sort_values("retention")
    overall = round(table["retention"].mean(), 4)
    return base_detect, overall, table, len(caught)


def plot(table, overall, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c53030" if v < 0.7 else "#dd6b20" if v < 0.9 else "#2f855a"
              for v in table["retention"]]
    ax.barh(table["mutation"], table["retention"], color=colors)
    ax.axvline(overall, color="#2b6cb0", linestyle="--",
               label=f"mean robustness = {overall:.2f}")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Detection retention after mutation  (1.0 = fully robust)")
    ax.set_title("Adversarial Robustness: detection retained under evasion mutations")
    for i, v in enumerate(table["retention"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="PhishGuard adversarial robustness test")
    ap.add_argument("--data", default="data/phishing_dataset.csv")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--url-only", action="store_true",
                    help="test URL-surface robustness only (drop HTML) — where evasion bites")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    base_detect, overall, table, n_caught = run(
        args.data, n=args.n, threshold=args.threshold, url_only=args.url_only)

    mode = "URL-only" if args.url_only else "URL+HTML"
    print(f"Mode: {mode}")
    print(f"Baseline phishing detection rate: {base_detect:.3f}  "
          f"({n_caught} caught examples used as the attack set)")
    print(f"Overall adversarial robustness (mean retention): {overall:.3f}\n")
    print(table.to_string(index=False))

    table.to_csv(os.path.join(RESULTS_DIR, "adversarial_robustness.csv"), index=False)
    plot(table, overall, os.path.join(RESULTS_DIR, "adversarial_robustness.png"))
    print(f"\nWrote results/adversarial_robustness.{{png,csv}}")


if __name__ == "__main__":
    main()

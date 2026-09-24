"""
PhishGuard :: LLM-assisted explanation layer (survey Direction 3)
------------------------------------------------------------------
Turns the numeric ablation attribution from model.explain() into a
one-line, analyst-readable rationale.

Design: the numeric attribution stays the source of truth (it is fast,
deterministic, and needs no network). The LLM only *verbalizes* the
already-computed top features -- so if no LLM is configured, a built-in
template produces the same sentence deterministically. This keeps the
system fully functional offline (the environment PhishGuard was built in)
while being ready to call a real model when an API key is present.

To enable the live LLM path, set ANTHROPIC_API_KEY (or OPENAI_API_KEY)
in the environment. Otherwise the template path is used automatically.
"""

import os

# Human-readable phrasing for each feature when it pushes toward "phishing".
_PHISH_PHRASING = {
    "has_ip_address": "the host is a raw IP address instead of a domain",
    "suspicious_tld": "it uses a TLD commonly abused by phishing",
    "brand_in_subdomain": "a known brand name appears in the subdomain (spoofing)",
    "brand_lookalike": "the domain is a look-alike of a known brand",
    "has_punycode": "the domain uses punycode (possible homograph attack)",
    "prefix_suffix_hyphen": "the hostname is padded with hyphens",
    "form_action_external": "a form submits data to an external domain",
    "form_action_empty_or_hash": "a form has an empty or placeholder action",
    "has_password_field": "the page collects a password",
    "iframe_count": "the page hides content in iframes",
    "external_link_ratio": "most links point off-site",
    "suspicious_word_count": "the URL contains credential-related keywords",
    "url_entropy": "the URL string looks randomly generated",
    "vowel_consonant_anomaly": "the domain looks machine-generated",
    "is_https": "the connection is not HTTPS",
    "has_at_symbol": "the URL contains an '@' redirect trick",
    "shortening_service": "the URL uses a link shortener",
    "meta_refresh_present": "the page auto-redirects via meta refresh",
    "right_click_disabled": "right-click is disabled (evasion)",
    "popup_on_load": "the page opens a popup on load",
    "num_subdomains": "the host has an unusually deep subdomain chain",
}


def _template_sentence(verdict, proba, top_features):
    reasons = []
    for f in top_features:
        if f["contribution"] > 0 and f["feature"] in _PHISH_PHRASING:
            reasons.append(_PHISH_PHRASING[f["feature"]])
        if len(reasons) >= 3:
            break
    if verdict == "LEGITIMATE" or not reasons:
        return (f"Assessed as {verdict.lower()} "
                f"(phishing probability {proba:.0%}); no strong phishing signals dominated.")
    joined = "; ".join(reasons)
    return f"Flagged as {verdict.lower()} ({proba:.0%}) mainly because {joined}."


def _llm_sentence(verdict, proba, top_features, url):
    """Call a real LLM if a key is configured. Kept dependency-light: only
    imported/used when a key exists, so offline installs never need the SDK."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        feats = ", ".join(
            f"{f['feature']}={f['value']} (contrib {f['contribution']:+.2f})"
            for f in top_features[:6]
        )
        prompt = (
            "You are a security analyst assistant. In ONE sentence, plainly explain "
            f"why this URL was assessed as {verdict} (phishing probability {proba:.0%}). "
            f"URL: {url}. Top contributing features: {feats}. "
            "Do not invent features not listed. Be concrete and concise."
        )
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return None  # any failure -> caller falls back to template


def explain_in_words(verdict, proba, top_features, url=""):
    """Return a one-line rationale. Uses the LLM when available, else the
    deterministic template. Always returns a string."""
    return _llm_sentence(verdict, proba, top_features, url) or \
        _template_sentence(verdict, proba, top_features)

"""
PhishGuard :: feature extraction
---------------------------------
API-independent hybrid feature architecture (Direction 2 of the survey).

Every feature here is computed from:
  - the raw URL string, and/or
  - HTML content the caller already has in hand (e.g. fetched by the
    browser/agent itself)

No WHOIS, no Google Safe Browsing, no Alexa rank, no third-party lookups.
That keeps extraction latency in the microsecond-to-low-millisecond range,
which is what makes real-time / browser-speed detection possible
(Gap 3 and Gap 6 in the survey).

Two feature families:
  URL_FEATURES  - lexical, always available, computed from the string alone
  HTML_FEATURES - structural, available only when HTML is supplied

If HTML isn't available (e.g. user only pastes a URL), HTML_FEATURES are
filled with neutral placeholder values and a `has_html` flag is set to 0,
so the same trained model can score URL-only or URL+HTML inputs.
"""

import re
import math
from urllib.parse import urlparse

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "account", "update", "secure", "banking",
    "confirm", "password", "webscr", "ebayisapi", "suspend", "urgent",
    "click", "limited", "security", "authenticate", "wallet", "recover",
]

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "shorte.st", "cutt.ly", "rebrand.ly", "tiny.cc",
}

URL_FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_digits",
    "num_special_chars",
    "num_subdomains",
    "has_ip_address",
    "has_at_symbol",
    "has_double_slash_redirect",
    "prefix_suffix_hyphen",
    "https_token_in_hostname",
    "is_https",
    "port_specified",
    "shortening_service",
    "suspicious_word_count",
    "digit_letter_ratio",
    "url_entropy",
    "tld_length",
    "long_url",
    "abnormal_subdomain_count",
]

HTML_FEATURE_NAMES = [
    "has_html",
    "num_forms",
    "form_action_external",
    "form_action_empty_or_hash",
    "has_password_field",
    "iframe_count",
    "external_link_ratio",
    "num_scripts",
    "onmouseover_present",
    "right_click_disabled",
    "popup_on_load",
    "favicon_external",
    "meta_refresh_present",
    "hidden_element_count",
    "num_external_links",
    "num_total_links",
]

FEATURE_NAMES = URL_FEATURE_NAMES + HTML_FEATURE_NAMES


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def _is_ip_address(hostname: str) -> bool:
    if not hostname:
        return False
    return bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", hostname)) or bool(
        re.match(r"^0x[0-9a-fA-F]+$", hostname)
    )


def extract_url_features(url: str) -> dict:
    """Pure lexical features computed from the URL string alone.
    No network access required -- safe to run on every request."""
    url = (url or "").strip()
    if not url:
        url = "http://"
    if "://" not in url:
        url = "http://" + url

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    special_chars = len(re.findall(r"[^\w\-.:/?=&%]", url))

    subdomains = hostname.split(".") if hostname else []
    # subtract domain + tld (rough heuristic, no external TLD list needed)
    num_subdomains = max(len(subdomains) - 2, 0)

    tld = subdomains[-1] if subdomains else ""

    feats = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": digits,
        "num_special_chars": special_chars,
        "num_subdomains": num_subdomains,
        "has_ip_address": int(_is_ip_address(hostname)),
        "has_at_symbol": int("@" in url),
        "has_double_slash_redirect": int(url.rfind("//") > 7),
        "prefix_suffix_hyphen": int("-" in hostname),
        "https_token_in_hostname": int("https" in hostname.lower()),
        "is_https": int(parsed.scheme == "https"),
        "port_specified": int(parsed.port is not None),
        "shortening_service": int(hostname.lower() in SHORTENERS),
        "suspicious_word_count": sum(
            1 for w in SUSPICIOUS_WORDS if w in url.lower()
        ),
        "digit_letter_ratio": round(digits / max(letters, 1), 4),
        "url_entropy": round(_shannon_entropy(url), 4),
        "tld_length": len(tld),
        "long_url": int(len(url) > 75),
        "abnormal_subdomain_count": int(num_subdomains > 3),
    }
    return feats


def extract_html_features(html: str, base_url: str = "") -> dict:
    """Structural features from raw HTML the caller already fetched.
    Falls back to neutral defaults if HTML is missing or bs4 is unavailable."""
    defaults = {name: 0 for name in HTML_FEATURE_NAMES}

    if not html or not _HAS_BS4:
        return defaults

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return defaults

    base_host = urlparse(base_url).hostname or "" if base_url else ""

    forms = soup.find_all("form")
    external_actions = 0
    empty_actions = 0
    for f in forms:
        action = (f.get("action") or "").strip()
        if action in ("", "#"):
            empty_actions += 1
        else:
            action_host = urlparse(action).hostname
            if action_host and base_host and action_host != base_host:
                external_actions += 1

    all_links = soup.find_all("a", href=True)
    external_links = 0
    for a in all_links:
        href = a["href"]
        link_host = urlparse(href).hostname
        if link_host and base_host and link_host != base_host:
            external_links += 1
    total_links = len(all_links)
    ext_ratio = round(external_links / total_links, 4) if total_links else 0.0

    html_lower = html.lower()

    favicon = soup.find("link", rel=lambda x: x and "icon" in x.lower())
    favicon_external = 0
    if favicon and favicon.get("href"):
        fav_host = urlparse(favicon["href"]).hostname
        if fav_host and base_host and fav_host != base_host:
            favicon_external = 1

    feats = {
        "has_html": 1,
        "num_forms": len(forms),
        "form_action_external": int(external_actions > 0),
        "form_action_empty_or_hash": int(empty_actions > 0),
        "has_password_field": int(bool(soup.find("input", {"type": "password"}))),
        "iframe_count": len(soup.find_all("iframe")),
        "external_link_ratio": ext_ratio,
        "num_scripts": len(soup.find_all("script")),
        "onmouseover_present": int("onmouseover" in html_lower),
        "right_click_disabled": int("event.button==2" in html_lower.replace(" ", "")
                                     or "contextmenu" in html_lower),
        "popup_on_load": int("window.open(" in html_lower.replace(" ", "")),
        "favicon_external": favicon_external,
        "meta_refresh_present": int(
            bool(soup.find("meta", attrs={"http-equiv": lambda v: v and v.lower() == "refresh"}))
        ),
        "hidden_element_count": len(
            soup.find_all(style=lambda v: v and "display:none" in v.replace(" ", "").lower())
        ),
        "num_external_links": external_links,
        "num_total_links": total_links,
    }
    return feats


def extract_features(url: str, html: str = "", base_url: str = "") -> dict:
    """Full feature vector (URL + optional HTML) as an ordered dict.
    `base_url` defaults to `url` when HTML was fetched from that same page."""
    url_feats = extract_url_features(url)
    html_feats = extract_html_features(html, base_url=base_url or url)
    combined = {**url_feats, **html_feats}
    # guarantee stable ordering matching FEATURE_NAMES
    return {name: combined.get(name, 0) for name in FEATURE_NAMES}


def features_to_vector(feats: dict) -> list:
    return [feats[name] for name in FEATURE_NAMES]

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

# TLDs disproportionately abused by phishing campaigns (Spamhaus / APWG
# "most abused TLD" reports). Kept as a small static set -- no network lookup.
SUSPICIOUS_TLDS = {
    "zip", "mov", "xyz", "top", "club", "info", "ru", "cn", "gq", "tk", "cc",
    "work", "link", "click", "country", "kim", "science", "party", "gdn",
    "review", "stream", "download", "loan", "men", "date", "racing", "win",
}

# Popular brands most commonly impersonated in phishing (Cofense / APWG
# brand-abuse reports). Used to flag brand-name-in-wrong-position tricks
# like paypal.com.secure-login.ru or amaz0n-account.tk.
TARGET_BRANDS = {
    "paypal", "amazon", "apple", "microsoft", "netflix", "google", "facebook",
    "instagram", "whatsapp", "linkedin", "chase", "wellsfargo", "bankofamerica",
    "citibank", "hsbc", "dhl", "fedex", "ups", "usps", "irs", "coinbase",
    "binance", "metamask", "outlook", "office365", "dropbox", "adobe",
}

# Common homoglyph substitutions used in look-alike domains (o->0, l->1, etc.)
HOMOGLYPH_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "7": "t", "@": "a"})

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
    # --- evasion-resistant features (v2) ---
    "suspicious_tld",
    "has_punycode",
    "brand_in_subdomain",
    "brand_lookalike",
    "vowel_consonant_anomaly",
    "hex_or_encoded_chars",
    "num_query_params",
    "path_depth",
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


def _brand_in_subdomain(hostname: str) -> int:
    """paypal.com.evil.ru -> the brand appears somewhere OTHER than the
    registered domain. We flag when a known brand token appears in the
    subdomain portion (everything before the last two labels)."""
    labels = hostname.lower().split(".")
    if len(labels) <= 2:
        return 0
    subdomain_labels = labels[:-2]
    for lab in subdomain_labels:
        for brand in TARGET_BRANDS:
            if brand in lab:
                return 1
    return 0


def _brand_lookalike(hostname: str) -> int:
    """amaz0n / paypa1 / g00gle: a homoglyph-normalized label matches a
    known brand but the raw label does not (i.e. it is a spoof, not the
    real brand)."""
    for lab in hostname.lower().split("."):
        if not lab:
            continue
        normalized = lab.translate(HOMOGLYPH_MAP)
        for brand in TARGET_BRANDS:
            if normalized == brand and lab != brand:
                return 1
    return 0


def _vowel_consonant_anomaly(hostname: str) -> int:
    """Randomly-generated phishing hostnames (djfkslqwmn.top) have very
    low vowel ratios. Flag registered-domain labels with an unusually low
    vowel fraction and reasonable length."""
    labels = [l for l in hostname.lower().split(".") if l.isalpha()]
    if not labels:
        return 0
    core = max(labels, key=len)
    if len(core) < 7:
        return 0
    vowels = sum(c in "aeiou" for c in core)
    return int(vowels / len(core) < 0.25)


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

    # Correct double-slash-redirect check: look for "//" AFTER the scheme's
    # "://" rather than a naive rfind that trips on long paths.
    after_scheme = url.split("://", 1)[-1]
    double_slash_redirect = int("//" in after_scheme)

    query = parsed.query or ""
    encoded_chars = len(re.findall(r"%[0-9a-fA-F]{2}", url))

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
        "has_double_slash_redirect": double_slash_redirect,
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
        # --- evasion-resistant features (v2) ---
        "suspicious_tld": int(tld.lower() in SUSPICIOUS_TLDS),
        "has_punycode": int("xn--" in hostname.lower()),
        "brand_in_subdomain": _brand_in_subdomain(hostname),
        "brand_lookalike": _brand_lookalike(hostname),
        "vowel_consonant_anomaly": _vowel_consonant_anomaly(hostname),
        "hex_or_encoded_chars": encoded_chars,
        "num_query_params": len([q for q in query.split("&") if q]) if query else 0,
        "path_depth": len([p for p in path.split("/") if p]),
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

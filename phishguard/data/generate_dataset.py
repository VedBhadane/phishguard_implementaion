"""
PhishGuard :: synthetic dataset generator
------------------------------------------
Generates a labeled dataset of (url, html, label) examples for training
and demoing the pipeline WITHOUT needing internet access.

This is NOT a real-world dataset -- it is a rule-based simulator that
constructs URLs/HTML with the statistical patterns the survey's reviewed
literature associates with phishing vs. legitimate sites (IP-based hosts,
excess subdomains, suspicious keywords, external form actions, etc.),
plus random noise so the classes aren't trivially separable.

To swap in a REAL dataset later (recommended before any real use):
  1. Download e.g. the UCI Phishing Websites dataset or a PhishTank/
     Kaggle URL dump.
  2. Produce a CSV with at least columns: url, html (optional), label
     (1 = phishing, 0 = legitimate).
  3. Point train.py at that CSV with --data path/to/real.csv
  4. Re-run training. Everything downstream (features.py, model.py,
     drift.py, app.py) is dataset-agnostic and needs no changes.

See README.md, section "Using a real dataset" for exact steps.
"""

import random
import csv
import argparse
import os

LEGIT_DOMAINS = [
    "amazon.com", "wikipedia.org", "github.com", "nytimes.com", "bbc.com",
    "microsoft.com", "apple.com", "spotify.com", "reddit.com", "stackoverflow.com",
    "linkedin.com", "dropbox.com", "notion.so", "figma.com", "airbnb.com",
    "coursera.org", "khanacademy.org", "who.int", "un.org", "harvard.edu",
]

LEGIT_PATHS = [
    "/", "/about", "/products/1234", "/blog/post-title", "/docs/getting-started",
    "/search?q=phishing+detection", "/user/settings", "/help/faq",
    "/articles/2026/08/17/news-item", "/login",
]

PHISH_BRANDS = [
    "paypal", "amazon", "apple", "microsoft", "netflix", "bankofamerica",
    "wellsfargo", "chase", "instagram", "facebook", "google", "irs",
]

PHISH_TLDS = [".xyz", ".top", ".club", ".info", ".ru", ".cn", ".gq", ".tk", ".cc"]

SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "update", "account", "confirm", "signin",
    "webscr", "suspend", "urgent", "recover", "wallet",
]


def _rand_hex(n):
    return "".join(random.choice("0123456789abcdef") for _ in range(n))


def _rand_alnum(n):
    return "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(n))


def make_legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(LEGIT_PATHS)
    scheme = "https"
    if random.random() < 0.08:  # occasional edge case, still legit
        path += f"?ref={_rand_alnum(6)}"
    return f"{scheme}://www.{domain}{path}"


def make_legit_html(url):
    domain = url.split("//")[1].split("/")[0]
    # occasionally a legitimate page also has a login form (e.g. the site's own
    # login page) so the model can't use "has a password field" in isolation
    has_login = random.random() < 0.25
    login_block = ""
    if has_login:
        login_block = f'<form action="https://{domain}/login" method="post"><input type="text" name="user"><input type="password" name="pass"></form>'
    num_scripts = random.randint(0, 3)
    scripts = "\n".join(f'<script src="/static/app{i}.js"></script>' for i in range(num_scripts))
    return f"""<html><head><title>Welcome</title>
<link rel="icon" href="https://{domain}/favicon.ico"></head>
<body>
<nav><a href="https://{domain}/">Home</a><a href="https://{domain}/about">About</a></nav>
{login_block}
<form action="/search" method="get"><input type="text" name="q"></form>
<p>Some legitimate content about our product.</p>
<a href="https://{domain}/terms">Terms</a>
<a href="https://en.wikipedia.org/wiki/Reference">External reference</a>
{scripts}
</body></html>"""


def make_phish_url():
    brand = random.choice(PHISH_BRANDS)
    style = random.choice([
        "ip", "subdomain_spoof", "typo_tld", "hyphen_spoof", "long_path", "clean_https"
    ])
    scheme = "https" if random.random() < 0.3 else "http"  # phishers increasingly use HTTPS too

    if style == "ip":
        ip = ".".join(str(random.randint(1, 254)) for _ in range(4))
        return f"{scheme}://{ip}/{brand}/{random.choice(SUSPICIOUS_WORDS)}.php"

    if style == "subdomain_spoof":
        fake_sub = f"{brand}-{random.choice(SUSPICIOUS_WORDS)}"
        host = f"{fake_sub}.{_rand_alnum(6)}.{_rand_alnum(4)}{random.choice(PHISH_TLDS)}"
        return f"{scheme}://{host}/account/{_rand_hex(8)}"

    if style == "typo_tld":
        return f"{scheme}://{brand}{random.choice(PHISH_TLDS)}/{random.choice(SUSPICIOUS_WORDS)}"

    if style == "hyphen_spoof":
        host = f"{brand}-{random.choice(SUSPICIOUS_WORDS)}-{_rand_alnum(4)}.com"
        return f"{scheme}://{host}/login.html?id={_rand_hex(10)}"

    if style == "clean_https":
        # no obvious lexical red flags -- relies on HTML-side tricks instead,
        # forces the model to weigh HTML features too, not just the URL
        host = f"{_rand_alnum(7)}-{_rand_alnum(5)}.{random.choice(['com', 'net', 'org'])}"
        return f"https://{host}/{random.choice(['account', 'session', 'portal'])}"

    # long_path
    host = f"{_rand_alnum(5)}-{_rand_alnum(5)}.{random.choice(PHISH_TLDS)}"
    path = "/".join(_rand_alnum(4) for _ in range(random.randint(3, 6)))
    return f"{scheme}://{host}/{path}/{random.choice(SUSPICIOUS_WORDS)}.php?session={_rand_hex(16)}"


def make_phish_html(url):
    """Randomize which suspicious HTML tricks appear (real phishing kits
    vary in sophistication) so the model can't rely on one fixed bundle
    of co-occurring signals -- it has to weigh each one independently."""
    ext_host = f"{_rand_alnum(6)}.{random.choice(PHISH_TLDS)}"

    use_iframe = random.random() < 0.5
    use_onmouseover = random.random() < 0.4
    use_right_click_block = random.random() < 0.4
    use_meta_refresh = random.random() < 0.35
    use_external_favicon = random.random() < 0.45
    num_scripts = random.randint(0, 2)

    body_attr = ' onmouseover="window.status=\'\';return true;"' if use_onmouseover else ""
    iframe_tag = f'<iframe src="http://{ext_host}/track" style="display:none"></iframe>' if use_iframe else ""
    rc_script = "<script>document.oncontextmenu = function(){return false;}</script>" if use_right_click_block else ""
    meta_tag = f'<meta http-equiv="refresh" content="3;url=http://{ext_host}/redirect">' if use_meta_refresh else ""
    favicon_host = ext_host if use_external_favicon else url.split("//")[-1].split("/")[0]
    scripts = "\n".join(f'<script src="http://{ext_host}/s{i}.js"></script>' for i in range(num_scripts))

    return f"""<html><head><title>{random.choice(PHISH_BRANDS).title()} Secure Login</title>
<link rel="icon" href="https://{favicon_host}/favicon.ico">{meta_tag}</head>
<body{body_attr}>
<form action="http://{ext_host}/collect.php" method="post">
<input type="text" name="user"><input type="password" name="pass">
</form>
{iframe_tag}
{rc_script}
{scripts}
<a href="http://{ext_host}/terms">Terms</a>
</body></html>"""


def generate(n_per_class=1500, seed=42, label_noise=0.0):
    """label_noise: fraction of examples whose label is flipped, to create
    realistic class overlap. Real phishing data is NOT linearly separable;
    a small amount of noise makes the benchmark discriminate between models
    and feature sets instead of everything scoring 100%. Default 0.0 keeps
    the original clean behavior; use e.g. 0.12 for a harder, more realistic
    benchmark. This is a stand-in for a real dataset, not a replacement."""
    random.seed(seed)
    rows = []
    for _ in range(n_per_class):
        url = make_legit_url()
        html = make_legit_html(url)
        rows.append((url, html, 0))
    for _ in range(n_per_class):
        url = make_phish_url()
        html = make_phish_html(url)
        rows.append((url, html, 1))
    if label_noise > 0:
        k = int(len(rows) * label_noise)
        for idx in random.sample(range(len(rows)), k):
            u, h, lbl = rows[idx]
            rows[idx] = (u, h, 1 - lbl)
    random.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic PhishGuard dataset")
    ap.add_argument("--n", type=int, default=1500, help="examples per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-noise", type=float, default=0.0,
                    help="fraction of labels to flip for realistic overlap (e.g. 0.12)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "phishing_dataset.csv"))
    args = ap.parse_args()

    rows = generate(args.n, args.seed, label_noise=args.label_noise)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "html", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

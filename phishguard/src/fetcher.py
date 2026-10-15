"""
PhishGuard :: safe page fetcher (v3)
-------------------------------------
Given only a URL, retrieve the raw HTML behind it so the HTML-structural
features can be computed automatically (the user no longer pastes
"view source").

This is the only step that touches the network, and it visits URLs that may
be malicious, so it is deliberately locked down:

  * Only http/https, and only ports 80, 443, 8080, 8443.
  * Every hostname is resolved and EVERY returned address must be a public
    (globally routable) IP. Loopback, private (10/8, 172.16/12, 192.168/16),
    link-local (incl. the 169.254.169.254 cloud-metadata address), CGNAT,
    multicast and reserved ranges are refused (SSRF protection).
  * Redirects are followed manually (max 3) and each hop is re-validated
    with the same rules, so a public URL cannot bounce us to an internal one.
  * Short timeout (default 4 s total), response capped at 1.5 MB.
  * Only HTML content types are accepted.
  * The page is only read as text and handed to the HTML parser. No
    JavaScript is executed, nothing is rendered, no cookies are kept and no
    credentials are sent.
  * TLS certificates are NOT verified, on purpose: many phishing sites use
    invalid/self-signed certificates and we only read the page, never send
    anything secret to it.

Failure never raises: fetch_html() always returns a dict, and the caller
falls back to URL-only scoring (used_html=False) when ok is False.

Known limitation (documented honestly): a hostile DNS server could answer
differently between our validation lookup and the actual connection
("DNS rebinding"). The app never returns fetched content to the caller (only
a verdict and feature values), which limits the impact to a blind GET. For a
public deployment run the fetcher in an isolated network namespace/egress
proxy as well.
"""

import ipaddress
import socket
import time
from urllib.parse import urlsplit, urljoin

import requests

try:  # silence the warning caused by verify=False (intentional, see above)
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:  # pragma: no cover
    pass

USER_AGENT = "PhishGuard-Scanner/3.0 (research; static HTML fetch only)"
ALLOWED_SCHEMES = ("http", "https")
ALLOWED_PORTS = (80, 443, 8080, 8443)
DEFAULT_TIMEOUT = 4.0
DEFAULT_MAX_BYTES = 1_500_000
DEFAULT_MAX_REDIRECTS = 3
HTML_TYPES = ("text/html", "application/xhtml+xml")


def _result(ok=False, reason="", html="", final_url="", status=None,
            nbytes=0, truncated=False, started=None, redirects=0):
    return {
        "ok": ok,
        "reason": reason,          # "ok" or a short machine-readable failure code
        "html": html,
        "final_url": final_url,
        "status_code": status,
        "bytes": nbytes,
        "truncated": truncated,
        "redirects": redirects,
        "elapsed_ms": int((time.monotonic() - started) * 1000) if started else 0,
    }


def _is_public_ip(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text.split("%")[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(ip.is_global) and not ip.is_multicast


def validate_target(url: str, allow_private: bool = False):
    """Return (ok, reason, normalized_url). `allow_private` exists only so the
    unit tests can talk to a local test server; the app never sets it."""
    url = (url or "").strip()
    if not url:
        return False, "invalid_url", url
    if "://" not in url:
        url = "http://" + url
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return False, "invalid_url", url
    if parts.scheme not in ALLOWED_SCHEMES or not host:
        return False, "invalid_url", url
    if parts.username or parts.password:
        return False, "invalid_url", url      # no credentials in URLs
    if port not in ALLOWED_PORTS and not allow_private:
        return False, "blocked_port", url
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False, "dns_failure", url
    except Exception:
        return False, "dns_failure", url
    addrs = {i[4][0] for i in infos}
    if not addrs:
        return False, "dns_failure", url
    if not allow_private and not all(_is_public_ip(a) for a in addrs):
        return False, "blocked_address", url
    return True, "ok", url


def fetch_html(url: str,
               timeout: float = DEFAULT_TIMEOUT,
               max_bytes: int = DEFAULT_MAX_BYTES,
               max_redirects: int = DEFAULT_MAX_REDIRECTS,
               allow_private: bool = False) -> dict:
    """Fetch the HTML at `url` safely. Never raises."""
    started = time.monotonic()
    deadline = started + timeout
    current = url
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
    session.trust_env = False   # ignore proxy env vars / .netrc for scanned URLs
    hops = 0
    try:
        while True:
            ok, reason, current = validate_target(current, allow_private=allow_private)
            if not ok:
                return _result(False, reason, final_url=current, started=started, redirects=hops)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _result(False, "timeout", final_url=current, started=started, redirects=hops)
            resp = session.get(current, timeout=(min(remaining, 3.0), min(remaining, timeout)),
                               allow_redirects=False, stream=True, verify=False)
            try:
                if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("Location")
                    if not loc:
                        return _result(False, "http_error", final_url=current,
                                       status=resp.status_code, started=started, redirects=hops)
                    hops += 1
                    if hops > max_redirects:
                        return _result(False, "too_many_redirects", final_url=current,
                                       status=resp.status_code, started=started, redirects=hops)
                    current = urljoin(current, loc)
                    continue
                if resp.status_code >= 400:
                    return _result(False, f"http_{resp.status_code}", final_url=current,
                                   status=resp.status_code, started=started, redirects=hops)
                ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if ctype and ctype not in HTML_TYPES:
                    return _result(False, "non_html", final_url=current,
                                   status=resp.status_code, started=started, redirects=hops)
                chunks, total, truncated = [], 0, False
                for chunk in resp.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    total += len(chunk)
                    chunks.append(chunk)
                    if total >= max_bytes:
                        truncated = True
                        break
                    if time.monotonic() > deadline:
                        truncated = True
                        break
                raw = b"".join(chunks)[:max_bytes]
                encoding = resp.encoding if resp.encoding and ctype else None
                try:
                    html = raw.decode(encoding or "utf-8", errors="replace")
                except LookupError:
                    html = raw.decode("utf-8", errors="replace")
                if not html.strip():
                    return _result(False, "empty_response", final_url=current,
                                   status=resp.status_code, started=started, redirects=hops)
                return _result(True, "ok", html=html, final_url=current,
                               status=resp.status_code, nbytes=len(raw),
                               truncated=truncated, started=started, redirects=hops)
            finally:
                resp.close()
    except requests.exceptions.Timeout:
        return _result(False, "timeout", final_url=current, started=started, redirects=hops)
    except requests.exceptions.SSLError:
        return _result(False, "connection_error", final_url=current, started=started, redirects=hops)
    except requests.exceptions.RequestException:
        return _result(False, "connection_error", final_url=current, started=started, redirects=hops)
    except Exception:
        return _result(False, "connection_error", final_url=current, started=started, redirects=hops)
    finally:
        session.close()


REASON_TEXT = {
    "ok": "Page fetched.",
    "invalid_url": "The URL could not be parsed.",
    "blocked_port": "The URL uses a port the fetcher does not visit.",
    "blocked_address": "The host resolves to a private or internal address, so it was not visited (SSRF protection).",
    "dns_failure": "The host name could not be resolved.",
    "timeout": "The site did not respond in time.",
    "too_many_redirects": "The URL redirected too many times.",
    "non_html": "The URL did not return an HTML page.",
    "empty_response": "The site returned an empty page.",
    "connection_error": "The site could not be reached.",
    "disabled": "Automatic page fetching is turned off.",
    "provided": "HTML was supplied by the caller.",
}


def reason_text(code: str) -> str:
    if code.startswith("http_"):
        return f"The site answered with HTTP {code[5:]} (it may block automated visitors)."
    return REASON_TEXT.get(code, "Page could not be fetched.")

"""Webpage + pasted-text extraction for the Library.

Fetch (in order): curl_cffi with Chrome TLS impersonation + browserforge
headers (beats most bot-filters — Reuters/Investopedia return 200), then
the shared aiohttp session as fallback. No Playwright: ~150MB + system
libs for marginal gain; stealth defeats bot-filters, never login paywalls.

Extraction keeps the LONGEST across strategies (trafilatura txt default,
trafilatura favor_recall for news layouts, bs4 blocks) — hub pages and
edge-cache shells otherwise win with a few nav lines.

Two outputs per save: plain text (→ sentences for the TTS reader, same
shape as book_service.extract()) and sanitized article HTML (→ the
human "Article" view with formatting + pictures). Hub pages return their
article links for the picker; fully blocked pages degrade to metadata
link cards instead of a bare 422.
"""
import asyncio
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

try:
    import trafilatura  # optional — bs4 fallback covers extraction without it
    HAS_TRAFILATURA = True
except ImportError:
    trafilatura = None  # type: ignore[assignment]
    HAS_TRAFILATURA = False

try:
    from curl_cffi import requests as _curl_requests
    HAS_STEALTH = True
except ImportError:
    _curl_requests = None  # type: ignore[assignment]
    HAS_STEALTH = False

_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

_BINARY_EXTS = {
    "xlsx", "xls", "docx", "doc", "pptx", "ppt", "zip", "gz", "tar",
    "rar", "7z", "png", "jpg", "jpeg", "gif", "webp", "mp4", "mp3",
    "wav", "ogg", "mov", "avi", "exe", "dmg",
}

# Anchors that are never article links (auth, commerce, page furniture).
_NON_ARTICLE_RE = re.compile(
    r"log[\s_-]?in|sign[\s_-]?in|sign[\s_-]?up|subscribe|newsletters?|"
    r"cart|checkout|account|settings|privacy|terms|cookies?|sitemap|"
    r"contact|about[\s_-]?us|careers?|advertis|donate|help|faq|search|"
    r"facebook|twitter|x\.com|instagram|linkedin|youtube|rss|mailto:",
    re.IGNORECASE,
)

MAX_FETCH_BYTES = 5 * 1024 * 1024

_session = None


async def _reset_session():
    """Drop the shared session so the next fetch opens a fresh connection
    (possibly a different edge node — stale stub pages are per-node)."""
    global _session
    if _session is not None:
        try:
            if not _session.closed:
                await _session.close()
        except Exception:
            pass
        _session = None


async def _get_session():
    import aiohttp

    global _session
    if _session is None or _session.closed:
        # Cache-Control: no-cache asks edge caches to revalidate rather
        # than serve a stale/nav-shell variant (observed intermittently on
        # Vary: User-Agent edges). Explicit user saves are low-frequency,
        # so origin revalidation cost is negligible.
        _session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": _BROWSER_UA, "Cache-Control": "no-cache"},
        )
    return _session


def normalize_url(url: str) -> str:
    """Trim + ensure a scheme so dedupe and fetch agree on one form."""
    url = (url or "").strip()
    if url and not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def title_from_url(url: str) -> str:
    """Human-readable fallback title from a URL path (link cards)."""
    try:
        path = urlparse(url).path.strip("/")
        slug = path.rsplit("/", 1)[-1] if path else ""
        slug = re.sub(r"\.(html?|php|aspx?)$", "", slug, flags=re.IGNORECASE)
        slug = re.sub(r"[-_+]+", " ", slug).strip()
        if slug:
            return slug[:120]
        return domain_of(url) or url
    except Exception:
        return url


def _split(text: str) -> List[str]:
    from tools import book_service as _books
    return _books._split_into_sentences(text)


def _clean_blocks(blocks: List[str]) -> str:
    seen: set[str] = set()
    out: List[str] = []
    for b in blocks:
        b = re.sub(r"\s+", " ", b or "").strip()
        if len(b) < 40 or b in seen:
            continue
        seen.add(b)
        out.append(b)
    return "\n\n".join(out)


def _trafilatura_text(html: str, url: str, recall: bool = False) -> str:
    if not HAS_TRAFILATURA:
        return ""
    try:
        text = trafilatura.extract(
            html, url=url, include_tables=True, include_comments=False,
            favor_precision=not recall, favor_recall=recall,
            include_links=False, output_format="txt",
        )
        return (text or "").strip()
    except Exception:
        return ""


def _bs4_blocks(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "button", "noscript", "svg", "iframe"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        blocks = [p.get_text(" ", strip=True) for p in main.find_all(["h1", "h2", "h3", "p", "li"])]
        if not blocks:
            whole = main.get_text(" ", strip=True)
            blocks = [whole] if whole else []
        return _clean_blocks(blocks)
    except Exception:
        return ""


def _extract_main_content(html: str, url: str) -> str:
    """Longest wins across trafilatura (default + recall) and bs4 blocks."""
    candidates = [
        _trafilatura_text(html, url),
        _trafilatura_text(html, url, recall=True),
        _bs4_blocks(html),
    ]
    return max(candidates, key=len, default="")


def _meta_content(soup, *names: str) -> str:
    """First non-empty meta content for property/name keys."""
    for name in names:
        for attr in ("property", "name"):
            try:
                tag = soup.find("meta", attrs={attr: name})
            except Exception:
                tag = None
            if tag and (tag.get("content") or "").strip():
                return tag["content"].strip()
    return ""


def _title_from_html(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        og = _meta_content(soup, "og:title", "twitter:title")
        if og:
            return og
        if soup.title and soup.title.string:
            return re.sub(r"\s+", " ", soup.title.string).strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(" ", strip=True)[:200]
    except Exception:
        pass
    return ""


def page_metadata(html: str, url: str) -> Dict[str, str]:
    """Title + description + top image for link cards and article headers."""
    meta = {"title": "", "description": "", "image": ""}
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        meta["title"] = _title_from_html(html)
        meta["description"] = _meta_content(soup, "og:description", "twitter:description", "description")[:500]
        img = _meta_content(soup, "og:image", "twitter:image")
        if img:
            meta["image"] = urljoin(url, img)
        else:
            first = soup.find("img", src=True)
            if first:
                meta["image"] = urljoin(url, first["src"])
    except Exception:
        pass
    return meta


def extract_hub_links(html: str, url: str, limit: int = 30) -> List[Dict[str, str]]:
    """Same-domain article links for hub/section-front pages (link picker)."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
        base_domain = domain_of(url)
        seen: set[str] = set()
        links: List[Dict[str, str]] = []
        for a in soup.find_all("a", href=True):
            title = re.sub(r"\s+", " ", a.get_text(" ", strip=True))
            if len(title) < 20:
                continue
            abs_url = urljoin(url, a["href"].strip())
            if not abs_url.startswith(("http://", "https://")):
                continue
            if urlparse(abs_url).fragment and abs_url.split("#")[0] == url.split("#")[0]:
                continue
            if domain_of(abs_url) != base_domain:
                continue
            if _NON_ARTICLE_RE.search(title) or _NON_ARTICLE_RE.search(abs_url):
                continue
            if abs_url in seen:
                continue
            seen.add(abs_url)
            links.append({"title": title[:200], "url": abs_url})
            if len(links) >= limit:
                break
        return links
    except Exception:
        return []


# Tags kept in the human article view. Everything else is unwrapped
# (text kept) or decomposed (dropped with contents).
_HTML_KEEP = {"p", "h1", "h2", "h3", "h4", "ul", "ol", "li", "blockquote",
              "figure", "figcaption", "img", "a", "strong", "em", "b", "i",
              "code", "pre", "table", "thead", "tbody", "tr", "th", "td",
              "hr", "br"}
_HTML_DROP = {"script", "style", "nav", "header", "footer", "aside", "form",
              "button", "input", "select", "textarea", "noscript", "iframe",
              "canvas", "svg", "video", "audio", "source", "link", "meta"}
_HTML_ATTRS = {
    "img": {"src", "alt", "title"},
    "a": {"href", "title"},
    "th": {"colspan", "rowspan"},
    "td": {"colspan", "rowspan"},
}

# Reader stylesheet for saved snapshots. `color-scheme: light dark` lets
# the sandboxed iframe follow the OS theme without any script.
ARTICLE_CSS = """
html { color-scheme: light dark; }
body { font: 16px/1.65 system-ui, -apple-system, sans-serif; margin: 0; }
main { max-width: 42rem; margin: 0 auto; padding: 1.5rem 1.25rem 3rem; }
img { max-width: 100%; height: auto; border-radius: 8px; }
figure { margin: 1.5em 0; }
figcaption { font-size: 0.85em; opacity: 0.7; }
pre { overflow-x: auto; padding: 1em; border-radius: 8px; background: rgba(127,127,127,0.12); }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid rgba(127,127,127,0.35); padding: 0.5em; text-align: left; }
blockquote { border-left: 3px solid rgba(127,127,127,0.5); margin-left: 0; padding-left: 1em; opacity: 0.85; }
"""


def article_document(title: str, body_html: str) -> str:
    """Wrap a sanitized body fragment (or escaped paste paragraphs) in a
    minimal standalone document for the sandboxed reader iframe."""
    import html as _html_mod
    safe_title = _html_mod.escape((title or "Saved page")[:200])
    return ("<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{safe_title}</title><style>{ARTICLE_CSS}</style></head>"
            f"<body><main>{body_html}</main></body></html>")


def clean_article_html(html: str, url: str) -> str:
    """Article HTML for the human reading view: structure + pictures kept,
    chrome/scripts/trackers stripped, URLs absolutized. XSS-safe by
    allowlist (rendered additionally in a sandboxed iframe)."""
    try:
        from bs4 import BeautifulSoup, Comment
    except ImportError:
        return ""
    try:
        base = ""
        if HAS_TRAFILATURA:
            try:
                base = trafilatura.extract(
                    html, url=url, include_tables=True, include_comments=False,
                    include_images=True, include_links=True, output_format="html",
                ) or ""
            except Exception:
                base = ""
        soup = BeautifulSoup(base or html, "html.parser")
        # Lead image from the ORIGINAL page head (trafilatura fragments
        # carry no meta tags) — used as hero when no inline picture survives.
        lead_image = ""
        try:
            head_soup = BeautifulSoup(html, "html.parser")
            lead_image = _meta_content(head_soup, "og:image", "twitter:image")
            if lead_image:
                lead_image = urljoin(url, lead_image)
        except Exception:
            pass
        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            comment.extract()
        for tag in soup.find_all(True):
            name = tag.name.lower()
            if name in _HTML_DROP:
                tag.decompose()
                continue
            if name not in _HTML_KEEP:
                tag.unwrap()  # drop tag, keep text (div/span/section/article/...)
                continue
            # Attribute allowlist (+ URL absolutization)
            allowed = _HTML_ATTRS.get(name, set())
            attrs: dict = {}
            if name == "img":
                src = tag.get("src") or tag.get("data-src") or ""
                if not src and tag.get("srcset"):
                    src = tag["srcset"].split(",")[0].strip().split(" ")[0]
                if not src or src.startswith("data:"):
                    tag.decompose()
                    continue
                attrs["src"] = urljoin(url, src)
                attrs["loading"] = "lazy"
                attrs["referrerpolicy"] = "no-referrer"
                if tag.get("alt"):
                    attrs["alt"] = tag["alt"][:300]
                if tag.get("title"):
                    attrs["title"] = tag["title"][:300]
            elif name == "a":
                href = (tag.get("href") or "").strip()
                if not href or href.startswith(("javascript:", "mailto:", "#")):
                    tag.unwrap()
                    continue
                attrs["href"] = urljoin(url, href)
                attrs["target"] = "_blank"
                attrs["rel"] = "noopener"
                if tag.get("title"):
                    attrs["title"] = tag["title"][:300]
            else:
                for k in allowed:
                    if tag.get(k):
                        attrs[k] = tag[k]
            tag.attrs = attrs
        # Drop empty leftovers and chrome remnants
        for tag in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote"]):
            if not tag.get_text(strip=True) and not tag.find("img"):
                tag.decompose()
        # Extractors often drop inline pictures (lazy-loaded, outside the
        # detected container) — fall back to the page's lead image so the
        # human view still has a picture.
        if not soup.find("img"):
            if lead_image:
                hero = soup.new_tag("figure")
                img = soup.new_tag("img", src=lead_image,
                                   loading="lazy", referrerpolicy="no-referrer",
                                   alt="Article image")
                hero.append(img)
                soup.insert(0, hero)
        body = soup.find("body")
        inner = body.decode_contents() if body else str(soup)
        return re.sub(r"\n{3,}", "\n\n", inner).strip()
    except Exception:
        return ""


def _fetch_stealth_sync(url: str, timeout: int) -> tuple:
    """Blocking curl_cffi fetch (Chrome TLS impersonation). Run in a thread."""
    from browserforge.headers import HeaderGenerator

    headers = HeaderGenerator().generate()
    resp = _curl_requests.get(url, impersonate="chrome124", headers=headers,
                              timeout=timeout, allow_redirects=True)
    return resp.content, (resp.headers.get("Content-Type") or "")


async def fetch_url_text(url: str, timeout: int = 30) -> Dict[str, Any]:
    """Fetch a URL and return {title, text, html, links, image, description}.

    `links` is populated for hub/section-front pages (thin article text but
    many same-domain article links) so the caller can offer a link picker.
    Raises ValueError on failure (message is user-facing via the endpoint).
    """
    import aiohttp

    url = normalize_url(url)
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    ext = urlparse(url).path.rsplit("/", 1)[-1].rsplit(".", 1)
    ext = ext[-1].lower() if len(ext) > 1 else ""
    if ext in _BINARY_EXTS:
        raise ValueError(f"URL points to a binary file (.{ext}), not a readable page")

    async def _get_aiohttp() -> tuple:
        session = await _get_session()
        async with session.get(url, allow_redirects=True,
                               timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                raise ValueError(f"Fetch failed: HTTP {resp.status}")
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if any(x in ctype for x in ("image/", "video/", "audio/", "octet-stream", "application/zip")):
                raise ValueError(f"URL is a binary file ({ctype}), not a readable page")
            raw = await resp.content.read(MAX_FETCH_BYTES + 1)
            return raw, ctype

    raw: Optional[bytes] = None
    ctype = ""
    errors: List[str] = []
    # Attempt 1: stealth (curl_cffi, Chrome impersonation) — beats bot-filters
    if HAS_STEALTH:
        try:
            raw, ctype = await asyncio.to_thread(_fetch_stealth_sync, url, timeout)
            ctype = (ctype or "").lower()
            if raw and len(raw) > MAX_FETCH_BYTES:
                raise ValueError("Page too large (>5MB)")
        except ValueError:
            raise
        except Exception as e:
            errors.append(f"stealth: {e}")
            raw = None
    # Attempt 2: shared aiohttp session
    if raw is None:
        try:
            raw, ctype = await _get_aiohttp()
        except ValueError:
            raise
        except Exception as e:
            errors.append(f"fetch: {e}")
    if raw is None:
        raise ValueError(f"Could not fetch URL: {'; '.join(errors) or 'unknown error'}")
    if len(raw) > MAX_FETCH_BYTES:
        raise ValueError("Page too large (>5MB)")
    if not raw.strip():
        raise ValueError("Empty response from URL")

    def _clean(raw_bytes: bytes) -> tuple:
        html = raw_bytes.decode("utf-8", errors="replace")
        if "text/html" not in ctype and "<html" not in html[:2000].lower():
            text = html.strip()  # plain-text endpoint (robots.txt, .md served raw, ...)
            title = urlparse(url).path.rsplit("/", 1)[-1] or url
            return html, text, title, ""
        text = _extract_main_content(html, url)
        title = _title_from_html(html) or urlparse(url).path.rsplit("/", 1)[-1] or url
        article_html = clean_article_html(html, url)
        return html, text, title, article_html

    html, text, title, article_html = _clean(raw)
    # Edges intermittently serve a nav-shell stub of the page (large HTML,
    # ~2K chars of chrome, zero article blocks). Re-fetch a couple of times
    # on fresh connections (stale stubs are per edge node) and keep longest.
    for _ in range(2):
        if len(text.strip()) >= 2000 or len(raw) <= 20000:
            break
        try:
            await _reset_session()
            raw2, ctype2 = await _get_aiohttp()
            _, text_b, _, html_b = _clean(raw2)
            if len(text_b) > len(text):
                raw, text, article_html = raw2, text_b, html_b
        except Exception:
            break
    meta = page_metadata(html, url)
    links = extract_hub_links(html, url) if len(text.strip()) < 2000 else []
    if _is_challenge(title, text):
        raise ValueError("Site shows a bot-check page (Cloudflare challenge)")
    if not text or len(text.strip()) < 20:
        if links:
            # Hub/section front: nothing to read, but plenty to choose from
            raise _HubResult(title or meta["title"] or title_from_url(url), links, meta)
        raise ValueError("No readable content extracted (JS-heavy page or paywall?)")
    return {"title": title[:300], "text": text, "html": article_html,
            "links": links, "image": meta["image"], "description": meta["description"]}


class _HubResult(Exception):
    """Not an error: the URL is a hub page — caller offers the link picker."""
    def __init__(self, title: str, links: List[Dict[str, str]], meta: Dict[str, str]):
        super().__init__("hub")
        self.title = title
        self.links = links
        self.meta = meta


# Bot-check / challenge pages (Cloudflare "Just a moment...", captchas).
# Saving these as articles yields 1-sentence junk — reject so the caller
# falls back to a link card instead.
_CHALLENGE_RE = re.compile(
    r"just a moment|attention required|verify you are (a )?human|"
    r"verify you're (a )?human|cloudflare|access denied|^forbidden$|"
    r"are you a robot|captcha|please verify|pardon our interruption",
    re.IGNORECASE,
)


def _is_challenge(title: str, text: str) -> bool:
    if len(text.strip()) >= 500:
        return False
    return bool(_CHALLENGE_RE.search(title or "") or _CHALLENGE_RE.search(text[:500]))


def sentences_from_text(text: str, title: str = "") -> Dict[str, Any]:
    """Clean text → book_service-shaped sentences, one section = page 1."""
    from tools import book_service as _books

    text = re.sub(r"[ \t]+", " ", (text or "")).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) < 20:
        raise ValueError("Nothing to save — text is empty")
    sentences, page_map = [], []
    offset = 0
    for sent in _split(text):
        sent = _books._strip_citations(sent)
        if not sent or _books._is_citation_only(sent) or _books._is_broken_chunk(sent):
            continue
        idx = text.find(sent[:40], offset)
        sentences.append({"text": sent, "page": 1, "char_start": idx if idx >= 0 else offset})
        page_map.append(1)
        offset = (idx if idx >= 0 else offset) + len(sent)
    if not sentences:
        raise ValueError("No readable sentences found in text")
    return {"sentences": sentences, "page_map": page_map,
            "title": (title or "").strip() or text[:60].split("\n")[0][:120]}


if __name__ == "__main__":
    assert normalize_url("example.com/a") == "https://example.com/a"
    assert normalize_url(" http://x.com ") == "http://x.com"
    assert domain_of("https://www.economist.com/business/2024") == "economist.com"
    assert title_from_url("https://example.com/finance-and-economics") == "finance and economics"
    r = sentences_from_text("Hello world. Second sentence here!", title="T")
    assert r["title"] == "T" and len(r["sentences"]) == 2 and r["page_map"] == [1, 1]
    r = sentences_from_text("  Line one here.\n\n\nLine two here.  ")
    assert len(r["sentences"]) == 2
    try:
        sentences_from_text("   ")
        assert False, "should raise"
    except ValueError:
        pass
    html = "<html><head><title>Biz News</title></head><body><nav>menu</nav><article><h1>Markets rally today</h1><p>" + \
        "Global markets rallied on Tuesday as investors welcomed strong earnings reports from major banks. " * 6 + \
        "</p><img src='/img/market.png' alt='chart'/><script>alert(1)</script></article></body></html>"
    assert _title_from_html(html) == "Biz News"
    body = _extract_main_content(html, "https://example.com/x")
    assert "Global markets rallied" in body and "menu" not in body, body[:100]
    # Sanitizer: structure + pictures kept, chrome/scripts stripped, URLs absolutized
    clean = clean_article_html(html, "https://example.com/x")
    assert "https://example.com/img/market.png" in clean, clean
    assert "<script" not in clean and "alert(1)" not in clean, clean
    assert 'loading="lazy"' in clean, clean
    # Full document wrapper for the reader iframe
    doc = article_document("Markets", "<p>Hello</p>")
    assert "<style>" in doc and "<main><p>Hello</p></main>" in doc, doc[:120]
    assert "<script" not in article_document("<b>x</b>", "<p>y</p>")
    # Hub links: same-domain article anchors kept, nav/auth/external dropped
    hub = ("<html><body><nav><a href='/login'>Log in</a></nav><main>"
           "<a href='https://example.com/markets-stocks-rally-today'>Markets and stocks rally today on earnings</a>"
           "<a href='https://other.com/x'>External long enough anchor text here</a>"
           "<a href='/a'>short</a></main></body></html>")
    links = extract_hub_links(hub, "https://example.com/business/")
    assert len(links) == 1 and links[0]["url"] == "https://example.com/markets-stocks-rally-today", links
    # Challenge pages rejected (→ link card instead of 1-sentence junk)
    assert _is_challenge("Just a moment...", "Just a moment...") is True
    assert _is_challenge("Economics - Wikipedia", "Economics " * 200) is False
    print("OK: web_extract self-check passed (trafilatura=%s stealth=%s)" % (HAS_TRAFILATURA, HAS_STEALTH))

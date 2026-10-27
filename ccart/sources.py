"""Load study material from files (PDF, PPTX, DOCX, TXT) and links (web pages, Wikipedia, PDF links, YouTube).

Everything returns a `Loaded` object: a title, a list of text pages and the kind of source.
Headings (slide titles, Word headings, HTML <h1>-<h6>) are written as numbered lines ("2.1 Title") so that the
pipeline can rebuild the section structure.  Only the Python standard library is needed, except YouTube
transcripts which use the optional `youtube-transcript-api` package.
"""
from __future__ import annotations

import io
import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable, List, Optional, Tuple

from .preprocess import extract_pdf_pages, pages_from_text


@dataclass
class Loaded:
    title: str
    pages: List[str]
    kind: str    # pdf | pptx | docx | txt | web | pdf-link | youtube
    source: str  # file name or URL
    note: str = ""  # shown to the user, e.g. "captions were translated"


KIND_ICON = {"pdf": "📄", "pdf-link": "📄", "pptx": "📊", "docx": "📝", "txt": "📃", "web": "🌐",
             "youtube": "▶️", "paste": "✏️", "sample": "🧪"}

_TRIVIAL_TITLES = {"agenda", "contents", "outline", "overview", "thank you", "thanks", "questions", "q&a",
                   "references", "any questions", "the end", "title", "welcome"}
_SKIP_SECTIONS = {"references", "external links", "notes", "further reading", "bibliography", "see also",
                  "footnotes", "sources", "citations", "navigation menu"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if not text or text[-1] in ".!?:" else text + "."


def blocks_to_pages(blocks: List[tuple]) -> List[str]:
    """[('h', level, text) | ('p', text)] -> pages of text with numbered headings ('1.' / '1.1')."""
    levels = [b[1] for b in blocks if b[0] == "h"]
    lowest = min(levels) if levels else 1
    counters = [0] * 7
    lines: List[str] = []
    for b in blocks:
        if b[0] == "h":
            lvl = max(1, min(6, b[1] - lowest + 1))
            counters[lvl - 1] += 1
            for j in range(lvl, 7):
                counters[j] = 0
            num = ".".join(str(max(c, 1)) for c in counters[:lvl])
            lines.append(f"{num}. {b[2]}" if lvl == 1 else f"{num} {b[2]}")
        else:
            lines.append(b[1])
    return pages_from_text("\n\n".join(lines))


def _clean_inline(text: str) -> str:
    text = re.sub(r"\[\s*(?:\d+|edit|citation needed|note \d+|[a-z])\s*\]", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# PPTX (PowerPoint)
# ---------------------------------------------------------------------------

_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_SKIP_PH = {"dt", "ftr", "sldNum", "hdr", "sldImg"}


def _shapes(elem):
    for ch in elem:
        if ch.tag == _P + "sp" or ch.tag == _P + "graphicFrame":
            yield ch
        elif ch.tag == _P + "grpSp":
            yield from _shapes(ch)


def _paragraphs(node) -> List[str]:
    out = []
    for p in node.iter(_A + "p"):
        t = "".join(x.text or "" for x in p.iter(_A + "t")).strip()
        if t:
            out.append(t)
    return out


def _slide_content(xml: bytes) -> Tuple[str, List[str]]:
    root = ET.fromstring(xml)
    tree = root.find(f"{_P}cSld/{_P}spTree")
    title, body = "", []
    if tree is None:
        return title, body
    for sh in _shapes(tree):
        ph = sh.find(f"{_P}nvSpPr/{_P}nvPr/{_P}ph")
        ptype = ph.get("type", "body") if ph is not None else ""
        if ptype in _SKIP_PH:
            continue
        paras = _paragraphs(sh)
        if not paras:
            continue
        if ptype in ("title", "ctrTitle") and not title:
            title = " ".join(paras)
        else:
            body.extend(paras)
    return title, body


def _slide_paths(z: zipfile.ZipFile) -> List[str]:
    names = set(z.namelist())
    try:
        pres = ET.fromstring(z.read("ppt/presentation.xml"))
        rels = ET.fromstring(z.read("ppt/_rels/presentation.xml.rels"))
        target = {r.get("Id"): r.get("Target") for r in rels.iter(_REL + "Relationship")}
        ordered = []
        for sid in pres.iter(_P + "sldId"):
            t = target.get(sid.get(_R + "id"))
            if t:
                path = "ppt/" + t.lstrip("/").replace("ppt/", "", 1) if not t.startswith("/") else t.lstrip("/")
                if path in names:
                    ordered.append(path)
        if ordered:
            return ordered
    except Exception:  # noqa: BLE001 - fall back to file-name order
        pass
    found = [n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
    return sorted(found, key=lambda n: int(re.findall(r"\d+", n)[-1]))


def _notes_text(z: zipfile.ZipFile, slide_path: str) -> List[str]:
    rels_path = slide_path.replace("slides/", "slides/_rels/") + ".rels"
    if rels_path not in z.namelist():
        return []
    for r in ET.fromstring(z.read(rels_path)).iter(_REL + "Relationship"):
        if r.get("Type", "").endswith("/notesSlide"):
            tgt = os.path.normpath(os.path.join(os.path.dirname(slide_path), r.get("Target", ""))).replace("\\", "/")
            if tgt in z.namelist():
                _, notes = _slide_content(z.read(tgt))
                return notes
    return []


def load_pptx(data: bytes, name: str = "slides.pptx") -> Loaded:
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("This does not look like a .pptx file (old .ppt files must be saved as .pptx first).") from exc
    pages: List[str] = []
    deck_title, n = "", 0
    for path in _slide_paths(z):
        title, body = _slide_content(z.read(path))
        body = [_sentence(_clean_inline(b)) for b in body + _notes_text(z, path) if _clean_inline(b)]
        lines: List[str] = []
        if title and title.strip().lower().rstrip(".") not in _TRIVIAL_TITLES:
            n += 1
            lines.append(f"{n}. {_clean_inline(title)}")
            deck_title = deck_title or _clean_inline(title)
        lines.extend(body)
        if lines:
            pages.append("\n\n".join(lines))
    if not pages:
        raise ValueError("No text was found in this presentation (slides with only pictures cannot be read).")
    stem = os.path.splitext(os.path.basename(name))[0]
    return Loaded(stem or deck_title or "Presentation", pages, "pptx", name)


# ---------------------------------------------------------------------------
# DOCX (Word)
# ---------------------------------------------------------------------------

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def load_docx(data: bytes, name: str = "notes.docx") -> Loaded:
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        root = ET.fromstring(z.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("This does not look like a .docx file (old .doc files must be saved as .docx first).") from exc
    blocks: List[tuple] = []
    for p in root.iter(_W + "p"):
        text = "".join((t.text or "") for t in p.iter(_W + "t")).strip()
        if not text:
            continue
        style = p.find(f"{_W}pPr/{_W}pStyle")
        sval = style.get(_W + "val", "") if style is not None else ""
        m = re.match(r"(?i)^heading\s*(\d)$", sval)
        if m:
            blocks.append(("h", int(m.group(1)), _clean_inline(text)))
        elif sval.lower() == "title":
            blocks.append(("h", 1, _clean_inline(text)))
        elif len(text.split()) >= 4:
            blocks.append(("p", _sentence(_clean_inline(text))))
    pages = blocks_to_pages(blocks)
    if not pages:
        raise ValueError("No text was found in this Word document.")
    return Loaded(os.path.splitext(os.path.basename(name))[0] or "Document", pages, "docx", name)


# ---------------------------------------------------------------------------
# Web pages
# ---------------------------------------------------------------------------

_SKIP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "button", "svg",
              "iframe", "figure", "figcaption", "table", "select", "dialog", "template", "sup", "sub"}
_SKIP_CLASS = {"navbox", "infobox", "sidebar", "toc", "reflist", "references", "reference", "mw-editsection",
               "hatnote", "metadata", "catlinks", "printfooter", "noprint", "cookie", "cookies", "breadcrumb",
               "menu", "navigation", "shortdescription", "mw-empty-elt", "thumb", "gallery", "mbox", "ambox",
               "sistersitebox", "authority-control", "vector-toc", "sidebar-list", "ad", "advert", "promo"}
_CONTAINER_ID = {"mw-content-text", "content", "bodycontent", "main-content", "article-body", "main"}
_VOID = {"br", "img", "hr", "meta", "link", "input", "area", "base", "col", "embed", "source", "track", "wbr"}
_BLOCK = {"p", "li", "dd", "dt", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}


class _Extract(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: List[Tuple[str, bool, bool]] = []  # (tag, skip, container)
        self.title, self._in_title, self.in_body = "", False, False
        self.cur: Optional[str] = None
        self.buf: List[str] = []
        self.blocks: List[tuple] = []  # (kind, level, text, in_container)

    def _skipping(self) -> bool:
        return any(s for _, s, _ in self.stack)

    def _in_container(self) -> bool:
        return any(c for _, _, c in self.stack)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        if tag == "body":
            self.in_body = True
        if tag in _VOID:
            return
        tokens = set(re.split(r"[\s_]+", (a.get("class", "") + " " + a.get("id", "")).lower()))
        skip = tag in _SKIP_TAGS or bool(tokens & _SKIP_CLASS) or a.get("aria-hidden") == "true" \
            or "hidden" in a or a.get("role") in ("navigation", "banner", "complementary", "contentinfo")
        container = tag in ("main", "article") or (a.get("id", "").lower() in _CONTAINER_ID)
        self.stack.append((tag, skip, container))
        if tag in _BLOCK and not self._skipping():
            self._finish()
            self.cur, self.buf = tag, []

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in _VOID:
            return
        if tag in _BLOCK and self.cur == tag:
            self._finish()
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self.cur and not self._skipping():
            self.buf.append(data)

    def _finish(self):
        if self.cur is None:
            return
        text = _clean_inline(" ".join(self.buf))
        inside = self._in_container()
        if text:
            if self.cur[0] == "h" and len(self.cur) == 2:
                self.blocks.append(("h", int(self.cur[1]), text, inside))
            elif len(text.split()) >= 4:
                self.blocks.append(("p", _sentence(text), inside))
        self.cur, self.buf = None, []


def html_to_pages(html: str) -> Tuple[str, List[str]]:
    parser = _Extract()
    parser.feed(html)
    parser._finish()
    blocks = [b for b in parser.blocks]
    if any(b[-1] for b in blocks):  # prefer the main article area when the page marks one
        blocks = [b for b in blocks if b[-1]]
    cut = []
    for b in blocks:  # drop "References", "External links", ... and everything after
        if b[0] == "h" and b[2].strip().lower() in _SKIP_SECTIONS:
            break
        cut.append(b)
    h1 = next((b[2] for b in cut if b[0] == "h" and b[1] == 1), "")
    title = _clean_inline(parser.title) or h1
    title = re.split(r"\s[-|–—]\s", title)[0].strip() or title
    if h1 and sum(1 for b in cut if b[0] == "h" and b[1] == 1) == 1:  # the page title is not a section
        cut = [b for b in cut if not (b[0] == "h" and b[1] == 1)]
    simple = [(b[0], b[1], b[2]) if b[0] == "h" else ("p", b[1]) for b in cut]
    pages = blocks_to_pages(simple)
    if sum(len(p.split()) for p in pages) < 40:
        raise ValueError("Not enough readable text on this page (it may need a login or load its text with "
                         "JavaScript). Try copying the text and using 'Paste text' instead.")
    return title or "Web page", pages


# ---------------------------------------------------------------------------
# Fetching links safely
# ---------------------------------------------------------------------------

def _check_public(url: str) -> None:
    p = urllib.parse.urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("Only http and https links are supported.")
    if os.environ.get("CC_ALLOW_PRIVATE_URLS") == "1":
        return
    try:
        infos = socket.getaddrinfo(p.hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not find the website '{p.hostname}'. Check the link.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise ValueError("Links to private or local addresses are blocked for safety.")


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_public(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_url(url: str, timeout: int = 25, max_bytes: int = 12_000_000) -> Tuple[bytes, str, str]:
    """Download a public link -> (bytes, content-type, final url)."""
    _check_public(url)
    opener = urllib.request.build_opener(_SafeRedirect)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; ConceptCartographer/1.0; study tool)",
        "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.5",
        "Accept-Language": "en"})
    try:
        with opener.open(req, timeout=timeout) as resp:
            data = resp.read(max_bytes + 1)
            ctype = resp.headers.get("Content-Type", "")
            final = resp.geturl()
    except urllib.error.HTTPError as exc:
        raise ValueError(f"The website answered with an error ({exc.code}). It may block automatic readers.") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Could not open the link: {exc.reason}") from exc
    if len(data) > max_bytes:
        raise ValueError("That page is too large to read (over 12 MB).")
    return data, ctype, final


def _decode(data: bytes, ctype: str) -> str:
    m = re.search(r"charset=([\w-]+)", ctype, re.I)
    enc = m.group(1) if m else None
    if not enc:
        m2 = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", data[:3000], re.I)
        enc = m2.group(1).decode("ascii", "ignore") if m2 else "utf-8"
    try:
        return data.decode(enc, errors="replace")
    except LookupError:
        return data.decode("utf-8", errors="replace")


_NO_TEXT_SITES = [
    (("canva.com", "canva.link"),
     "Canva shows its slides with JavaScript, so the text cannot be read from a link. In Canva choose "
     "Share -> Download -> PPTX (or PDF), then use 'Upload file'."),
    (("books.google.", "play.google.com"),
     "Google Books pages cannot be read: the book text is protected and not given to programs. Use a PDF or "
     "notes you are allowed to use ('Upload file'), or a free chapter published by the author or publisher."),
    (("onedrive.live.com", "1drv.ms", "sharepoint.com"),
     "OneDrive / SharePoint links cannot be opened automatically. Download the file and use 'Upload file'."),
    (("scribd.com", "slideshare.net", "coursehero.com", "chegg.com"),
     "This site hides its text behind a login or JavaScript. Download the file if you may, then 'Upload file'."),
]


def rewrite_known_link(url: str) -> Tuple[str, str]:
    """Google Slides / Docs / Drive share links -> their direct download link. Returns (url, note)."""
    p = urllib.parse.urlparse(url)
    host, path = (p.hostname or "").lower(), p.path
    if host == "docs.google.com":
        m = re.match(r"^/(presentation|document)/d/([\w-]+)", path)
        if m:
            kind, fid = m.groups()
            fmt = "pptx" if kind == "presentation" else "docx"
            tail = "export/pptx" if kind == "presentation" else "export?format=docx"
            return f"https://docs.google.com/{kind}/d/{fid}/{tail}", f"Google {'Slides' if kind == 'presentation' else 'Docs'} as {fmt}"
    if host == "drive.google.com":
        m = re.match(r"^/file/d/([\w-]+)", path) or None
        fid = m.group(1) if m else (urllib.parse.parse_qs(p.query).get("id") or [""])[0]
        if fid:
            return f"https://drive.google.com/uc?export=download&id={fid}", "Google Drive file"
    return url, ""


def _office_kind(data: bytes) -> str:
    """'pptx' / 'docx' / '' by looking inside a zip (links often have no file extension)."""
    if data[:2] != b"PK":
        return ""
    try:
        names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    except zipfile.BadZipFile:
        return ""
    if "ppt/presentation.xml" in names:
        return "pptx"
    if "word/document.xml" in names:
        return "docx"
    return ""


def load_url(url: str, fetch: Callable = fetch_url) -> Loaded:
    url = url.strip()
    if not re.match(r"^[a-z][a-z0-9+.-]*://", url, re.I):
        url = "https://" + url
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host in _YT_HOSTS:
        return load_youtube(url)
    for hosts, message in _NO_TEXT_SITES:
        if any(h in host for h in hosts):
            raise ValueError(message)
    url, note = rewrite_known_link(url)
    try:
        data, ctype, final = fetch(url)
    except ValueError as exc:
        if note:
            raise ValueError(f"Could not download the {note}. Make sure its sharing is set to 'Anyone with the "
                             f"link can view'. ({exc})") from exc
        raise
    lower = final.lower().split("?")[0]
    name = os.path.basename(urllib.parse.urlparse(final).path) or host
    office = _office_kind(data)
    if "pdf" in ctype.lower() or lower.endswith(".pdf") or data[:4] == b"%PDF":
        pages = extract_pdf_pages(data)
        if not any(p.strip() for p in pages):
            raise ValueError("No text could be read from that PDF link (it may be a scanned document).")
        return Loaded(os.path.splitext(name)[0] or host, pages, "pdf-link", url)
    if office == "pptx" or lower.endswith(".pptx"):
        ld = load_pptx(data, name if name.endswith(".pptx") else (note or "slides") + ".pptx")
        return Loaded(ld.title, ld.pages, "pptx", url, note)
    if office == "docx" or lower.endswith(".docx"):
        ld = load_docx(data, name if name.endswith(".docx") else (note or "document") + ".docx")
        return Loaded(ld.title, ld.pages, "docx", url, note)
    text = _decode(data, ctype)
    if "html" in ctype.lower() or re.match(r"\s*<(!doctype|html|head|body)", text[:200], re.I):
        if note:  # a share link that returned a web page = a login page, not the file
            raise ValueError(f"The {note} is not shared publicly. Set sharing to 'Anyone with the link can view' "
                             "and try again, or download the file and use 'Upload file'.")
        title, pages = html_to_pages(text)
        return Loaded(title, pages, "web", url)
    if ctype.lower().startswith("text/") or not ctype:
        pages = pages_from_text(text)
        if not any(p.strip() for p in pages):
            raise ValueError("That link contains no text.")
        return Loaded(name or host, pages, "web", url)
    raise ValueError(f"Unsupported content type: {ctype.split(';')[0] or 'unknown'}.")


# ---------------------------------------------------------------------------
# YouTube (captions / transcript)
# ---------------------------------------------------------------------------

_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be",
             "youtube-nocookie.com", "www.youtube-nocookie.com"}


def youtube_id(url: str) -> Optional[str]:
    p = urllib.parse.urlparse(url)
    host = (p.hostname or "").lower()
    if host.endswith("youtu.be"):
        vid = p.path.strip("/").split("/")[0]
    else:
        qs = urllib.parse.parse_qs(p.query)
        if "v" in qs:
            vid = qs["v"][0]
        else:
            m = re.match(r"^/(?:shorts|embed|live|v)/([\w-]{6,})", p.path)
            vid = m.group(1) if m else ""
    return vid if re.fullmatch(r"[\w-]{6,20}", vid or "") else None


def _snippet(sn) -> dict:
    if isinstance(sn, dict):
        return {"text": sn.get("text", ""), "start": sn.get("start", 0), "duration": sn.get("duration", 0)}
    return {"text": getattr(sn, "text", ""), "start": getattr(sn, "start", 0), "duration": getattr(sn, "duration", 0)}


def _pick_transcript(items: list):
    """Best English captions: hand-made, then auto-generated, then any language translated to English."""
    lang = lambda t: (getattr(t, "language_code", "") or "").lower()  # noqa: E731
    for want_generated in (False, True):
        for t in items:
            if lang(t).startswith("en") and bool(getattr(t, "is_generated", False)) == want_generated:
                return t, ""
    for t in items:
        if getattr(t, "is_translatable", False):
            try:
                return t.translate("en"), f"The video's captions are in '{lang(t)}'; YouTube translated them to English."
            except Exception:  # noqa: BLE001
                continue
    return None, ", ".join(f"{lang(t)}{' (auto)' if getattr(t, 'is_generated', False) else ''}" for t in items)


_YT_ERRORS = {
    "TranscriptsDisabled": "Captions are turned off for this video.",
    "VideoUnavailable": "This video is unavailable or private.",
    "VideoUnplayable": "This video cannot be played, so its captions cannot be read.",
    "AgeRestricted": "This video is age-restricted, so its captions cannot be read.",
    "RequestBlocked": "YouTube blocked the request (it sometimes does this on shared or cloud networks). "
                      "Try again later or on another network.",
    "IpBlocked": "YouTube blocked this network's request. Try again later or on another network.",
    "NoTranscriptFound": "This video has no captions the app can use.",
}


def _fetch_transcript(video_id: str):
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise RuntimeError("YouTube links need one extra package. Run:  pip install youtube-transcript-api") from exc
    try:
        try:
            api = YouTubeTranscriptApi()
            tlist = api.list(video_id) if hasattr(api, "list") else YouTubeTranscriptApi.list_transcripts(video_id)
        except TypeError:  # very old versions have only static methods
            tlist = YouTubeTranscriptApi.list_transcripts(video_id)
        items = list(tlist)
        chosen, info = _pick_transcript(items)
        if chosen is None:
            raise RuntimeError("The video only has captions in a language that cannot be translated to English "
                               f"(available: {info or 'none'}). This app reads English text.")
        return [_snippet(x) for x in chosen.fetch()], info
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        raise RuntimeError(_YT_ERRORS.get(name, f"Could not get captions for this video ({name}). It may have no "
                                                 "captions, or YouTube blocked the request.")) from exc


def _oembed_title(url: str) -> str:
    try:
        data, _, _ = fetch_url("https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(url, safe=""),
                               timeout=10, max_bytes=200_000)
        return json.loads(data.decode("utf-8")).get("title", "")
    except Exception:  # noqa: BLE001
        return ""


def transcript_to_text(segments: List[dict]) -> str:
    """Captions have no punctuation: group them into sentences by pauses and length, then paragraphs."""
    sentences: List[str] = []
    cur: List[str] = []
    words, last_end = 0, None

    def flush():
        nonlocal cur, words
        if cur:
            s = re.sub(r"\s+", " ", " ".join(cur)).strip()
            if s:
                s = s[0].upper() + s[1:]
                sentences.append(s if s[-1] in ".!?" else s + ".")
        cur, words = [], 0

    for seg in segments:
        t = re.sub(r"\[[^\]]*\]|\([^)]*\)|[\u266a\u266b]", " ", str(seg.get("text", "")).replace("\n", " "))
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            continue
        start = float(seg.get("start", 0) or 0)
        if cur and last_end is not None and (start - last_end > 1.5 or words >= 28):
            flush()
        cur.append(t)
        words += len(t.split())
        last_end = start + float(seg.get("duration", 0) or 0)
        if t[-1] in ".!?":
            flush()
    flush()
    paras = [" ".join(sentences[i:i + 6]) for i in range(0, len(sentences), 6)]
    return "\n\n".join(paras)


def load_youtube(url: str, fetcher: Callable = _fetch_transcript, titler: Callable = _oembed_title) -> Loaded:
    vid = youtube_id(url)
    if not vid:
        raise ValueError("That does not look like a YouTube video link.")
    got = fetcher(vid)
    segments, note = got if isinstance(got, tuple) else (got, "")
    text = transcript_to_text(segments)
    pages = pages_from_text(text)
    if sum(len(p.split()) for p in pages) < 40:
        raise ValueError("This video's transcript is too short to build a map.")
    return Loaded(titler(url) or f"YouTube video {vid}", pages, "youtube", url, note)

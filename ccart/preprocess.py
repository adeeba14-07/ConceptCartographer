"""PDF/text extraction, cleaning, sentence splitting, tokenization, lemmatization.

NLP topics covered here: text normalisation, sentence segmentation, tokenization,
(rule-based) lemmatization / plural stemming, stop-word handling.
"""
from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .config import SMALL_WORDS

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Mention:
    start_tok: int
    end_tok: int  # exclusive
    cid: str
    char_start: int
    char_end: int


@dataclass
class Sentence:
    sid: int
    text: str
    page: int
    para: int
    tokens: List[Tuple[str, int, int]] = field(default_factory=list)  # (text, start, end)
    lemmas: List[str] = field(default_factory=list)
    mentions: List[Mention] = field(default_factory=list)
    heading: bool = False
    num: Optional[str] = None


# ---------------------------------------------------------------------------
# Tokenization & lemmatization
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[^\W\d_]\w*(?:[-'\u2019]\w+)*|\d+(?:[.,]\d+)*|[^\w\s]", re.UNICODE)
WORD_START_RE = re.compile(r"[^\W\d_]")

_IRREGULAR = {
    "corpora": "corpus", "matrices": "matrix", "indices": "index", "vertices": "vertex",
    "analyses": "analysis", "hypotheses": "hypothesis", "criteria": "criterion",
    "phenomena": "phenomenon", "bias": "bias", "lens": "lens", "series": "series",
    "species": "species", "news": "news", "does": "does", "goes": "go", "data": "data",
    "media": "media", "bases": "base", "axes": "axis", "children": "child", "people": "person",
    "men": "man", "women": "woman", "feet": "foot", "mice": "mouse",
}


_KEEP_S = {"statistics", "mathematics", "physics", "semantics", "pragmatics", "linguistics", "analytics",
           "dynamics", "economics", "logistics", "genetics", "robotics", "phonetics", "informatics"}


def simple_lemma(word: str) -> str:
    """Rule-based plural/possessive normaliser (no external model needed)."""
    w = word.lower().replace("\u2019", "'")
    if w.endswith("'s"):
        w = w[:-2]
    elif w.endswith("s'"):
        w = w[:-1]
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) < 4 or not w.isascii():
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "xes", "ches", "shes", "zzes")):
        return w[:-2]
    if w in _KEEP_S:
        return w
    if w.endswith(("ss", "us", "is", "ous", "ness")):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def is_word(tok: str) -> bool:
    return bool(WORD_START_RE.match(tok))


def key_of(phrase: str) -> Tuple[str, ...]:
    """Canonical (lemma) key of a phrase, used for concept matching."""
    return tuple(simple_lemma(t) for t, _, _ in tokenize(phrase) if is_word(t))


def _cap_piece(p: str, first: bool) -> str:
    if not p:
        return p
    if len(p) > 1 and p.isupper():
        return p
    if any(c.isupper() for c in p[1:]):
        return p
    if not first and p.lower() in SMALL_WORDS:
        return p.lower()
    return p[0].upper() + p[1:]


def pretty_name(text: str) -> str:
    """Title-case a concept name, preserving acronyms (LDA) and hyphen parts."""
    out = []
    for i, word in enumerate(text.split()):
        if "-" in word:
            parts = word.split("-")
            out.append("-".join(_cap_piece(p, first=(i == 0 and j == 0)) for j, p in enumerate(parts)))
        else:
            out.append(_cap_piece(word, first=(i == 0)))
    return " ".join(out)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "concept"


# ---------------------------------------------------------------------------
# PDF / text extraction
# ---------------------------------------------------------------------------


def extract_pdf_pages(data: bytes) -> List[str]:
    """Return a list with one text string per PDF page (pdfplumber, pypdf fallback)."""
    pages: List[str] = []
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        if any(p.strip() for p in pages):
            return pages
    except Exception:  # noqa: BLE001 - fall through to pypdf
        pages = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return [(p.extract_text() or "") for p in reader.pages]
    except Exception:  # noqa: BLE001
        return pages


def pages_from_text(text: str, chars_per_page: int = 2400) -> List[str]:
    """Split raw text into pages: on form-feeds if present, else synthetic pages."""
    if "\f" in text:
        return [p for p in text.split("\f")]
    paras = re.split(r"\n\s*\n", text.strip())
    pages, cur, size = [], [], 0
    for para in paras:
        cur.append(para)
        size += len(para)
        if size >= chars_per_page:
            pages.append("\n\n".join(cur))
            cur, size = [], 0
    if cur:
        pages.append("\n\n".join(cur))
    return pages


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "")
    s = re.sub(r"[\u2010\u2011\u2012]", "-", s)
    s = re.sub(r"[ \t\r\v]+", " ", s)
    return s


_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s*)?[-\u2013]?\s*\d{1,4}\s*[-\u2013]?\s*$", re.I)


def _norm_line_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def clean_pages(pages: List[str]) -> List[str]:
    """Normalise unicode, drop running headers/footers and bare page numbers."""
    pages = [normalize_text(p) for p in pages]
    n = len(pages)
    drop = set()
    if n >= 4:
        counts: dict = {}
        for p in pages:
            lines = [l.strip() for l in p.splitlines() if l.strip()]
            for l in set(lines[:2] + lines[-2:]):
                counts[_norm_line_key(l)] = counts.get(_norm_line_key(l), 0) + 1
        thr = max(3, int(0.5 * n))
        drop = {k for k, c in counts.items() if c >= thr}
    cleaned = []
    for p in pages:
        out = []
        for l in p.splitlines():
            if _PAGE_NUM_RE.match(l):
                continue
            if l.strip() and _norm_line_key(l) in drop:
                continue
            out.append(l)
        txt = "\n".join(out)
        txt = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", txt)  # de-hyphenate line breaks
        cleaned.append(txt)
    return cleaned


def _is_heading(line: str, nxt: Optional[str]) -> bool:
    words = line.split()
    if not (1 <= len(words) <= 10):
        return False
    if line.rstrip()[-1] in ".,;:!?)":
        return False
    if not (line[0].isupper() or line[0].isdigit()):
        return False
    if words[-1].lower() in SMALL_WORDS:
        return False
    return nxt is None or nxt == "" or nxt[0].isupper() or nxt[0].isdigit()


_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+")


def page_to_blocks(text: str) -> List[Tuple[str, bool, Optional[str]]]:
    """Join wrapped lines into paragraphs; headings become their own block -> [(text, is_heading)]."""
    lines = [l.strip() for l in text.splitlines()]
    paras: List[Tuple[str, bool, Optional[str]]] = []
    cur: List[str] = []

    def flush():
        if cur:
            paras.append((" ".join(cur), False, None))
            cur.clear()

    for i, line in enumerate(lines):
        if not line:
            flush()
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if _is_heading(line, nxt):
            flush()
            mnum = _NUM_RE.match(line)
            head = _NUM_RE.sub("", line, count=1)
            if head.isupper() and len(head.split()) >= 2:
                head = head.title()
            paras.append((head + ".", True, mnum.group(1) if mnum else None))
            continue
        cur.append(line)
    flush()
    return paras


def page_to_paragraphs(text: str) -> List[str]:
    return [t for t, _, _ in page_to_blocks(text)]


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

ABBREV = {
    "e.g", "i.e", "etc", "fig", "figs", "eq", "eqs", "al", "vs", "dr", "mr", "mrs", "ms", "prof",
    "sec", "no", "approx", "cf", "inc", "ltd", "st", "vol", "pp", "p", "ch", "ed", "eds", "resp",
}
_SPLIT_RE = re.compile(r"([.!?]+[\"')\]\u201d\u2019]*)(\s+)")


def split_sentences(paragraph: str) -> List[str]:
    text = re.sub(r"\s+", " ", paragraph).strip()
    if not text:
        return []
    out: List[str] = []
    start = 0
    for m in _SPLIT_RE.finditer(text):
        end = m.end(1)
        nxt = text[m.end(): m.end() + 1]
        if not nxt:
            continue
        cand = text[start:end]
        parts = cand.split()
        last = parts[-1] if parts else ""
        core = last.rstrip(".!?\"')]\u201d\u2019")
        if m.group(1).startswith(".") and len(m.group(1)) == 1:
            if core.lower() in ABBREV or re.fullmatch(r"[A-Z]", core):
                continue
            if re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", core):
                continue
        if not (nxt.isupper() or nxt.isdigit() or nxt in "\"'([\u201c\u2018"):
            continue
        out.append(cand.strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


_EX_START = re.compile(
    r"^[\(\[\s]*(?:for\s+(?:example|instance)|e\.g\.?|i\.e\.?|example|examples|consider|suppose|imagine|"
    r"let\s+us\s+(?:consider|take)|let's|say,|figure|fig\.|table|exercise|question|q\d)\b", re.I)
_EX_ANY = re.compile(r"\b(?:for\s+example|for\s+instance|e\.g\.)", re.I)
_QUOTED = re.compile(r"[\"\u201c][^\"\u201c\u201d]{3,}[\"\u201d]")


def is_example_sentence(text: str) -> bool:
    """True for textbook illustrations ('For example, the boy ate an apple.', quoted sentences, ...)."""
    t = text.strip()
    if _EX_START.search(t) or _EX_ANY.search(t):
        return True
    quoted = sum(len(m.group(0)) for m in _QUOTED.finditer(t))
    return quoted >= 0.5 * len(t)


def build_sentences(pages: List[str], first_page: int = 1, max_words: int = 120,
                    drop_examples: bool = True) -> List[Sentence]:
    """Pages -> list of Sentence objects with page/paragraph ids, tokens and lemmas."""
    sentences: List[Sentence] = []
    para_id = 0
    for pi, page in enumerate(pages):
        for para, is_head, head_num in page_to_blocks(page):
            para_id += 1
            for sent in split_sentences(para):
                toks = tokenize(sent)
                words = [t for t in toks if is_word(t[0])]
                if len(words) < (1 if is_head else 2) or len(words) > max_words:
                    continue  # one-word headings such as '1.1 Mitochondria' are kept
                if drop_examples and is_example_sentence(sent):
                    continue
                lemmas = [simple_lemma(t[0]) if is_word(t[0]) else t[0] for t in toks]
                sentences.append(
                    Sentence(sid=len(sentences), text=sent, page=first_page + pi, para=para_id,
                             tokens=toks, lemmas=lemmas, heading=is_head, num=head_num)
                )
    return sentences

"""Explanations for every concept: a simple definition, how it connects, and how it works.

Rule: the definition comes from the document whenever the document defines the concept.
Only when it does not, a definition is written - by an AI model when an API key is available,
otherwise built offline from the concept's connections in the document.
"""
from __future__ import annotations

import json
import os
import re
import textwrap
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx

from .concepts import ConceptStore
from .outline import heading_concept
from .preprocess import Sentence, is_word

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

SOURCE_TEXT = {
    "document": "From your document",
    "document_section": "From the start of its section in your document",
    "ai": "Written by AI in simple words - this is not in your document",
    "auto": "Built from how it connects in your document (the document gives no definition)",
}


def _short(text: str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _n_words(s: Sentence) -> int:
    return sum(1 for t, _, _ in s.tokens if is_word(t))


# ---------------------------------------------------------------------------
# How a concept connects (built only from real edges of the graph)
# ---------------------------------------------------------------------------

_OUT = {
    "subtopic": "Includes", "is_a": "Is a kind of", "requires": "Uses", "used_for": "Used for",
    "has_part": "Made of", "part_of": "Part of", "evaluates": "Evaluates", "improves": "Improves",
    "produces": "Produces", "operates_on": "Works on", "defined_as": "Also called",
}
_IN = {
    "subtopic": "Part of", "is_a": "Kinds of it", "requires": "Used by", "used_for": "Helped by",
    "has_part": "Part of", "part_of": "Made of", "evaluates": "Evaluated by", "improves": "Improved by",
    "produces": "Produced by", "operates_on": "Processed by", "defined_as": "Also called",
}
_ORDER = ["Part of", "Is a kind of", "Includes", "Made of", "Kinds of it", "Uses", "Used by", "Used for",
          "Helped by", "Works on", "Processed by", "Evaluates", "Evaluated by", "Improves", "Improved by",
          "Produces", "Produced by", "Also called", "Explained in the same section", "Often appears with"]


def connection_groups(G: nx.DiGraph, cid: str, names: Dict[str, str], limit: int = 5) -> List[Tuple[str, List[str]]]:
    groups: Dict[str, List[str]] = {}

    def add(label: str, other: str):
        lst = groups.setdefault(label, [])
        if names[other] not in lst:
            lst.append(names[other])

    for _, v, d in G.out_edges(cid, data=True):
        if d["kind"] in ("triple", "structure"):
            add(_OUT.get(d["rel_type"], "Related to"), v)
        elif d["kind"] == "section":
            add("Explained in the same section", v)
        else:
            add("Often appears with", v)
    for u, _, d in G.in_edges(cid, data=True):
        if d["kind"] in ("triple", "structure"):
            add(_IN.get(d["rel_type"], "Related to"), u)
        elif d["kind"] == "section":
            add("Part of" if G.nodes[u].get("heading", False) else "Explained in the same section", u)
        else:
            add("Often appears with", u)
    ordered = sorted(groups, key=lambda k: _ORDER.index(k) if k in _ORDER else 50)
    return [(k, groups[k][:limit]) for k in ordered]


def _join(items: List[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------------------
# Offline explanation
# ---------------------------------------------------------------------------

def offline_definition(name: str, groups: List[Tuple[str, List[str]]], page: Optional[int]) -> str:
    g = dict(groups)
    parts: List[str] = []
    if g.get("Is a kind of"):
        parts.append(f"{name} is a kind of {_join(g['Is a kind of'][:2])}.")
    elif g.get("Part of"):
        parts.append(f"{name} is part of {_join(g['Part of'][:2])}.")
    elif g.get("Includes"):
        parts.append(f"{name} is a topic that includes {_join(g['Includes'][:4])}.")
    if g.get("Uses"):
        parts.append(f"It uses {_join(g['Uses'][:3])}.")
    if g.get("Used for"):
        parts.append(f"It is used for {_join(g['Used for'][:3])}.")
    if g.get("Works on"):
        parts.append(f"It works on {_join(g['Works on'][:3])}.")
    if g.get("Used by"):
        parts.append(f"It is used by {_join(g['Used by'][:3])}.")
    if not parts:
        near = (g.get("Explained in the same section") or g.get("Often appears with") or [])[:3]
        where = f" on page {page}" if page else ""
        if near:
            parts.append(f"{name} is a topic discussed{where} together with {_join(near)}.")
        else:
            parts.append(f"{name} is a topic discussed{where} of the document.")
    return " ".join(parts)


def _context_sentence(store: ConceptStore, sentences: List[Sentence], cid: str, avoid: str = "") -> Optional[Sentence]:
    """Best explaining sentence: the concept near the start, a reasonable length, not a heading."""
    best, best_key = None, None
    for sid in store.concepts[cid].sentence_ids:
        s = sentences[sid]
        if s.heading or s.text == avoid:
            continue
        n = _n_words(s)
        if not 8 <= n <= 40:
            continue
        m = next((m for m in s.mentions if m.cid == cid), None)
        pos = m.start_tok if m else 99
        key = (0 if pos <= 2 else 1 if pos <= 6 else 2, abs(n - 18), s.sid)  # concept as the subject first
        if best_key is None or key < best_key:
            best, best_key = s, key
    return best


def build_explanations(sentences: List[Sentence], store: ConceptStore, G: nx.DiGraph) -> Dict[str, dict]:
    """Definition (document first), how it connects, how it works - for every concept."""
    names = {cid: c.name for cid, c in store.concepts.items()}
    section_def: Dict[str, Sentence] = {}
    cur: Optional[str] = None
    for s in sentences:  # the first sentences under a section heading explain that section's concept
        if s.heading:
            cur = heading_concept(s)
            continue
        if cur and cur not in section_def and 6 <= _n_words(s) <= 55:
            section_def[cur] = s
            cur = None
    out: Dict[str, dict] = {}
    for cid, c in store.concepts.items():
        groups = connection_groups(G, cid, names)
        page = c.pages[0] if c.pages else None
        if c.def_rank == 0 and c.definition:
            definition, source, def_text = c.definition, "document", c.definition
        elif cid in section_def:
            s = section_def[cid]
            definition, source, def_text = _short(s.text), "document_section", s.text
        else:
            definition, source, def_text = offline_definition(c.name, groups, page), "auto", ""
        ctx = _context_sentence(store, sentences, cid, avoid=def_text)
        out[cid] = {
            "name": c.name, "page": page, "definition": definition, "def_source": source,
            "how": _short(ctx.text, 260) if ctx else "", "how_source": "document" if ctx else "",
            "how_page": ctx.page if ctx else None, "connects": groups,
        }
    return out


# ---------------------------------------------------------------------------
# AI definitions (only for concepts the document does not define)
# ---------------------------------------------------------------------------

def make_anthropic_completer(api_key: str, model: Optional[str] = None, timeout: int = 90) -> Callable[[str], str]:
    model = model or os.environ.get("CC_MODEL") or DEFAULT_MODEL

    def complete(prompt: str) -> str:
        body = json.dumps({"model": model, "max_tokens": 4000, "temperature": 0.2,
                           "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body, method="POST",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.load(exc).get("error", {}).get("message", "")
            except Exception:  # noqa: BLE001
                detail = ""
            raise RuntimeError(f"AI service error {exc.code}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach the AI service: {exc.reason}") from exc
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    return complete


_PROMPT = """You are helping a student understand a document. For each concept below write two short parts:
- "definition": ONE simple sentence (at most 25 words) saying what the concept is, in plain everyday words.
- "how": ONE simple sentence (at most 30 words) saying how it works or why it matters.
Rules: use the document context when it helps, otherwise use well-established general knowledge. Be accurate;
if you are not sure what a concept means in this document, say it in a general, safe way. Do not copy the
context sentences word for word. No jargon without a quick explanation.

Subject of the document (its main headings): {subject}

Return ONLY a JSON object that maps each id to {{"definition": "...", "how": "..."}}.

Concepts:
{items}
"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the AI reply")
    return json.loads(m.group(0))


def fill_with_ai(explain: Dict[str, dict], store: ConceptStore, sentences: List[Sentence], G: nx.DiGraph,
                 complete: Callable[[str], str], batch_size: int = 20) -> Dict:
    """Write definitions ONLY for concepts whose definition source is 'auto' (not in the document)."""
    targets = [cid for cid, e in explain.items() if e["def_source"] == "auto"]
    headings = [s.text.rstrip(".") for s in sentences if s.heading][:12]
    subject = "; ".join(headings) or "(unknown)"
    report = {"requested": len(targets), "written": 0, "errors": []}
    for i in range(0, len(targets), batch_size):
        batch = targets[i:i + batch_size]
        lines = []
        for cid in batch:
            e = explain[cid]
            ctx = _context_sentence(store, sentences, cid)
            rel = "; ".join(f"{k.lower()} {', '.join(v[:3])}" for k, v in e["connects"][:4])
            lines.append(f"- id: {cid} | name: {e['name']} | connections: {rel or 'none'} | "
                         f"context: {_short(ctx.text, 220) if ctx else 'none'}")
        try:
            data = _parse_json(complete(_PROMPT.format(subject=subject, items="\n".join(lines))))
        except Exception as exc:  # noqa: BLE001 - keep the offline text for this batch
            report["errors"].append(str(exc))
            continue
        for cid in batch:
            item = data.get(cid)
            if not isinstance(item, dict):
                continue
            d, h = str(item.get("definition", "")).strip(), str(item.get("how", "")).strip()
            if 5 <= len(d) <= 400:
                explain[cid]["definition"], explain[cid]["def_source"] = d, "ai"
                report["written"] += 1
                if 5 <= len(h) <= 400 and explain[cid]["how_source"] != "document":
                    explain[cid]["how"], explain[cid]["how_source"] = h, "ai"
    return report


# ---------------------------------------------------------------------------
# Tooltips for the map
# ---------------------------------------------------------------------------

def build_tips(G: nx.DiGraph, explain: Dict[str, dict], topic_names: Dict[int, str]) -> Dict[str, dict]:
    tips = {}
    for n, d in G.nodes(data=True):
        e = explain.get(n)
        if not e:
            continue
        tid = d.get("topic")
        tips[n] = {
            "name": e["name"], "topic": topic_names.get(tid, "") if tid is not None else "",
            "page": e["page"], "definition": e["definition"], "source": SOURCE_TEXT[e["def_source"]],
            "src_kind": e["def_source"], "how": e["how"], "how_kind": e["how_source"],
            "connects": [[k, v] for k, v in e["connects"]], "mentions": d.get("freq", 0),
        }
    return tips


def tip_as_text(t: dict) -> str:
    lines = [t["name"] + (f"   [{t['topic']}]" if t["topic"] else ""), "",
             textwrap.fill(t["definition"], 62), f"({t['source']})"]
    if t["how"]:
        lines += ["", "How it works:", textwrap.fill(t["how"], 62)]
    if t["connects"]:
        lines += ["", "How it connects:"] + [f" - {k}: {', '.join(v)}" for k, v in t["connects"][:6]]
    return "\n".join(lines)

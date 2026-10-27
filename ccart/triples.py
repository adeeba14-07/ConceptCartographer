"""Predicate-Argument Structure extraction (Unit IV).

Two complementary extractors produce (subject concept, predicate, object concept) triples:

* dependency-based (spaCy): subject/object/prepositional-object/agent links of every verb,
  including copular "X is a type of Y", passives ("X is computed by Y" -> Y computes X),
  relative clauses and coordination;
* pattern-based (works without a parser): relation phrases found between consecutive
  concept mentions, with support for coordination ("A and B use C") and passives.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .config import PART_NOUNS, TYPE_NOUNS, classify_relation
from .preprocess import Sentence


@dataclass
class Triple:
    subj: str
    rel: str
    obj: str
    rel_type: str
    prereq_dir: Optional[str]
    source: str  # "dependency" | "pattern"
    sid: int
    page: int
    sentence: str
    count: int = 1
    pages: Set[int] = field(default_factory=set)

    def key(self):
        return (self.subj, self.obj, self.rel_type, self.rel if self.rel_type == "related" else "")


# ---------------------------------------------------------------------------
# Dependency-based
# ---------------------------------------------------------------------------


def _np_tokens(tok):
    """Tokens of the noun phrase headed by `tok` (do not descend into clauses/preps)."""
    stack, out = [tok], []
    while stack:
        t = stack.pop()
        out.append(t)
        for c in t.children:
            if c.dep_ in ("relcl", "acl", "advcl", "ccomp", "xcomp", "conj", "cc", "punct", "prep", "agent"):
                continue
            stack.append(c)
    return out


def dependency_triples(doc, sent: Sentence) -> List[Tuple[str, str, str]]:
    if not sent.mentions:
        return []
    tok_cid: Dict[int, str] = {}
    for m in sent.mentions:
        for t in doc:
            if m.char_start <= t.idx < m.char_end:
                tok_cid[t.i] = m.cid

    def concept_of(tok) -> Optional[str]:
        if tok.i in tok_cid:
            return tok_cid[tok.i]
        best, bd = None, 99
        for t in _np_tokens(tok):
            d = abs(t.i - tok.i)
            if t.i in tok_cid and d <= 3 and d < bd:
                best, bd = tok_cid[t.i], d
        return best

    def expand(tokens):
        out = []
        for t in tokens:
            out.append(t)
            out.extend(list(t.conjuncts))
        seen, res = set(), []
        for t in out:
            if t.i not in seen:
                seen.add(t.i)
                res.append(t)
        return res

    def pobjs(prep):
        return [g for g in prep.children if g.dep_ in ("pobj",)]

    results: List[Tuple[str, str, str]] = []
    for v in doc:
        if v.pos_ not in ("VERB", "AUX"):
            continue
        lemma = v.lemma_.lower()
        if v.pos_ == "AUX" and lemma != "be":
            continue
        children = list(v.children)
        subj = [c for c in children if c.dep_ in ("nsubj", "nsubjpass", "csubj")]
        passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in children)
        if not subj and v.dep_ == "conj" and v.head.pos_ in ("VERB", "AUX"):
            subj = [c for c in v.head.children if c.dep_ in ("nsubj", "nsubjpass")]
        if v.dep_ == "relcl":
            subj = [v.head] if (not subj or subj[0].lower_ in ("that", "which", "who")) else subj
        if not subj:
            continue
        subj_all = expand(subj)

        objs: List[Tuple] = []
        agents: List = []
        if lemma == "be":
            for c in children:
                if c.dep_ == "attr":
                    if c.lemma_.lower() in TYPE_NOUNS | PART_NOUNS:
                        had = False
                        for p in c.children:
                            if p.dep_ == "prep" and p.lower_ == "of":
                                for po in pobjs(p):
                                    objs.append((po, f"is a {c.lemma_.lower()} of"))
                                    had = True
                        if not had:
                            objs.append((c, "is a"))
                    else:
                        objs.append((c, "is a"))
                elif c.dep_ == "acomp":
                    for p in c.children:
                        if p.dep_ == "prep":
                            for po in pobjs(p):
                                objs.append((po, f"is {c.lower_} {p.lower_}"))
        else:
            prt = [c.lower_ for c in children if c.dep_ == "prt"]
            base = (f"is {v.lower_}" if passive else lemma) + (f" {prt[0]}" if prt else "")
            for c in children:
                if c.dep_ in ("dobj", "obj", "dative", "oprd"):
                    objs.append((c, base))
                elif c.dep_ == "prep":
                    for po in pobjs(c):
                        objs.append((po, f"{base} {c.lower_}"))
                elif c.dep_ == "agent":
                    agents.extend(pobjs(c))
                elif c.dep_ == "xcomp":
                    for g in c.children:
                        if g.dep_ in ("dobj", "obj"):
                            objs.append((g, f"{base} to {c.lemma_.lower()}" if passive else f"{base} {c.lemma_.lower()}"))
                        elif g.dep_ == "prep":
                            for po in pobjs(g):
                                objs.append((po, f"{base} {c.lemma_.lower()} {g.lower_}"))
            if passive and agents:
                # "X is computed by Y"  ->  (Y, compute, X)
                for ag in expand(agents):
                    ac = concept_of(ag)
                    for s in subj_all:
                        sc = concept_of(s)
                        if ac and sc and ac != sc:
                            results.append((ac, lemma, sc))
                continue

        for s in subj_all:
            sc = concept_of(s)
            if not sc:
                continue
            for o, rel in objs:
                for oo in expand([o]):
                    oc = concept_of(oo)
                    if oc and oc != sc:
                        results.append((sc, rel, oc))
    return results


# ---------------------------------------------------------------------------
# Pattern-based
# ---------------------------------------------------------------------------

_COORD = {",", "and", "or", "&", "plus", "as", "well"}
_FILLER = {"also", "then", "can", "may", "will", "often", "typically", "usually", "generally", "simply",
           "commonly", "just", "only", "which", "that", "who", "now", "thus", "hence", "first", "next",
           ","}
_NEG = {"not", "no", "never", "cannot", "n't", "without", "neither", "nor"}
_STOP_PUNCT = {".", "?", "!", ";", ":", "(", ")"}
_PASSIVE_BY = re.compile(r"\b(?:is|are|was|were|be|been|being)\s+(\w+)\s+by$")
_AUX = {"is", "are", "be", "was", "were", "been", "being"}
_PREP = {"on", "of", "to", "from", "by", "as", "for", "in", "with", "into", "over"}
_DET_TRAIL = {"the", "each", "every", "all", "any", "some", "very", "more", "most", "its", "their", "this",
              "that", "these", "those", "one", "other", "same", "how", "well", "whether", "whose"}
_NOUN_SUFFIX = ("tion", "sion", "ment", "ness", "ity", "al")


def _is_key(w: str) -> bool:
    if w in _AUX:
        return True
    if w.endswith(_NOUN_SUFFIX):
        return False
    return classify_relation(w)[0] != "related"


def clean_label(words: List[str]) -> Optional[str]:
    """Reduce the text between two concepts to its relation phrase ('is a type of', 'relies on')."""
    start = next((i for i, w in enumerate(words) if _is_key(w)), None)
    if start is None:
        return None
    j = start + 1
    while j < len(words) and (_is_key(words[j]) or words[j] in _PREP or words[j] in ("a", "an")
                              or words[j] in TYPE_NOUNS or words[j] in PART_NOUNS):
        j += 1
    label = words[start:j]
    while label and (label[-1] in _DET_TRAIL
                     or (label[-1] in ("a", "an") and not (len(label) > 1 and label[-2] in _AUX))):
        label.pop()
    if not label:
        return None
    text = " ".join(label)
    return re.sub(r"^be\b", "is", text)


def pattern_triples(sent: Sentence) -> List[Tuple[str, str, str]]:
    ms = sorted(sent.mentions, key=lambda m: m.start_tok)
    if len(ms) < 2:
        return []
    groups = [[ms[0]]]
    for prev, cur in zip(ms, ms[1:]):
        gap = [t[0].lower() for t in sent.tokens[prev.end_tok:cur.start_tok]]
        if gap and all(g in _COORD for g in gap):
            groups[-1].append(cur)
        else:
            groups.append([cur])
    out: List[Tuple[str, str, str]] = []
    for g1, g2 in zip(groups, groups[1:]):
        a, b = g1[-1], g2[0]
        gap = [t[0].lower() for t in sent.tokens[a.end_tok:b.start_tok]]
        if not gap or len(gap) > 8 or any(t in _STOP_PUNCT for t in gap) or any(t in _NEG for t in gap):
            continue
        words = [t for t in gap if t not in _FILLER]
        if not words:
            continue
        swap = False
        m = _PASSIVE_BY.search(" ".join(words))
        if m:
            swap, label = True, m.group(1)
        else:
            label = clean_label(words)
        if not label:
            continue
        rtype, _ = classify_relation(label)
        if rtype == "related":
            continue
        if rtype == "is_a" and len(words) - len(label.split()) > 1:
            continue
        if len(words) > 7:
            continue
        for x in g1:
            for y in g2:
                if x.cid == y.cid:
                    continue
                out.append((y.cid, label, x.cid) if swap else (x.cid, label, y.cid))
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def extract_triples(sentences: List[Sentence], docs=None, strict: bool = False) -> List[Triple]:
    """Run both extractors, classify relations, deduplicate and count evidence."""
    agg: Dict[tuple, Triple] = {}

    def add(s: Sentence, subj: str, rel: str, obj: str, source: str):
        rtype, pdir = classify_relation(rel)
        if strict and rtype == "related":
            return
        if subj == obj:
            return
        t = Triple(subj, rel.strip(), obj, rtype, pdir, source, s.sid, s.page, s.text, 1, {s.page})
        k = t.key()
        if k in agg:
            agg[k].count += 1
            agg[k].pages.add(s.page)
        else:
            agg[k] = t

    for idx, s in enumerate(sentences):
        if len(s.mentions) < 2:
            continue
        if docs is not None:
            try:
                for subj, rel, obj in dependency_triples(docs[idx], s):
                    add(s, subj, rel, obj, "dependency")
            except Exception:  # noqa: BLE001 - never let one odd parse break the run
                pass
        for subj, rel, obj in pattern_triples(s):
            add(s, subj, rel, obj, "pattern")
    return sorted(agg.values(), key=lambda t: (-t.count, t.subj, t.obj))

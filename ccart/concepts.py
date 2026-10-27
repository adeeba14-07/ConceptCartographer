"""Concept (key-term) extraction, acronym merging, mention matching, definitions.

Candidates come from spaCy noun chunks when a parser is available; otherwise from a
stop-word / verb-breaker n-gram term extractor (statistical fallback).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import GENERIC_SINGLE, LEADING_MODIFIERS, STOPWORDS, VERBISH
from .preprocess import (Mention, Sentence, is_word, key_of, pretty_name, simple_lemma,
                         slugify, tokenize)

Key = Tuple[str, ...]


@dataclass
class Concept:
    cid: str
    key: Key
    name: str
    aliases: List[str] = field(default_factory=list)
    freq: int = 0
    pages: List[int] = field(default_factory=list)
    sentence_ids: List[int] = field(default_factory=list)
    definition: str = ""
    definition_page: Optional[int] = None
    score: float = 0.0
    heading: bool = False
    def_rank: int = 9  # 0 = the document defines it, 1 = it is the subject of a sentence, 2 = only mentioned


class ConceptMatcher:
    """Greedy longest-match lookup of lemma sequences -> concept id."""

    def __init__(self, key_map: Dict[Key, str]):
        self.key_map = dict(key_map)
        self.max_len = max((len(k) for k in self.key_map), default=1)

    _LIGHT = {":", ",", "-", "\u2013", "\u2014", "(", ")", "/", "&", "'", '"', "\u201c", "\u201d", "\u2019"}

    def find(self, lemmas: List[str]) -> List[Tuple[int, int, str]]:
        """Greedy longest match over the WORDS of a sentence; light punctuation (: , - ( ) /) inside a
        phrase is allowed, so headings like 'Morphology: Concept and Types' still match."""
        widx = [i for i, l in enumerate(lemmas) if l[:1].isalpha()]
        ws = [lemmas[i] for i in widx]
        out: List[Tuple[int, int, str]] = []
        p, n = 0, len(ws)
        while p < n:
            hit = None
            for L in range(min(self.max_len, n - p), 0, -1):
                cid = self.key_map.get(tuple(ws[p:p + L]))
                if cid is None:
                    continue
                if L > 1 and any(not all(t in self._LIGHT for t in lemmas[widx[p + j] + 1:widx[p + j + 1]])
                                 for j in range(L - 1)):
                    continue  # a full stop / semicolon between the words: not one phrase
                hit = (widx[p], widx[p + L - 1] + 1, cid, L)
                break
            if hit:
                out.append(hit[:3])
                p += hit[3]
            else:
                p += 1
        return out


@dataclass
class ConceptStore:
    concepts: Dict[str, Concept]
    matcher: ConceptMatcher
    acronyms: Dict[str, str] = field(default_factory=dict)  # acronym -> expansion

    def name(self, cid: str) -> str:
        c = self.concepts.get(cid)
        return c.name if c else cid


# ---------------------------------------------------------------------------
# Acronyms
# ---------------------------------------------------------------------------

_ACRO_RE = re.compile(r"\(([A-Z][A-Za-z0-9\-]{1,12})\)")


def find_acronyms(sentences: List[Sentence]) -> Dict[str, str]:
    """'Latent Dirichlet Allocation (LDA)' -> {'lda': 'Latent Dirichlet Allocation'}."""
    found: Dict[str, str] = {}
    small = {"of", "and", "the", "for", "a", "an", "in", "on", "to"}
    for s in sentences:
        for m in _ACRO_RE.finditer(s.text):
            acr = m.group(1)
            letters = re.sub(r"[^A-Za-z]", "", acr).upper()
            if len(letters) < 2:
                continue
            before = s.text[: m.start()]
            words = list(re.finditer(r"[A-Za-z]+", before))
            for k in range(len(letters), min(len(letters) + 3, len(words)) + 1):
                if k > len(words):
                    break
                ws = words[-k:]
                all_init = "".join(w.group(0)[0].upper() for w in ws)
                content_init = "".join(w.group(0)[0].upper() for w in ws if w.group(0).lower() not in small)
                if letters in (all_init, content_init) and ws[0].group(0).lower() not in small:
                    phrase = before[ws[0].start():].strip()
                    found.setdefault(acr.lower(), phrase)
                    break
    return found


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def _valid_key(key: Key, allow_inner_stop: bool = False) -> bool:
    if not key or len(key) > 4 and not allow_inner_stop:
        return False
    if key[0] in STOPWORDS or key[-1] in STOPWORDS:
        return False
    if not allow_inner_stop and any(k in STOPWORDS for k in key):
        return False
    if all(k in STOPWORDS or k in GENERIC_SINGLE for k in key):
        return False
    if len(key) == 1:
        k = key[0]
        if len(k) < 2 or k in GENERIC_SINGLE or k in VERBISH or k in LEADING_MODIFIERS:
            return False
    if any(not re.search(r"[a-z]", k) for k in key):
        return False
    return True


def _spacy_candidates(sentences: List[Sentence], docs) -> Dict[Key, dict]:
    cands: Dict[Key, dict] = {}
    bad_left = {"DET", "PRON", "NUM", "PUNCT", "ADV", "CCONJ", "ADP", "SCONJ", "PART", "SYM"}
    for doc in docs:
        for chunk in doc.noun_chunks:
            toks = list(chunk)
            while toks and (toks[0].pos_ in bad_left or toks[0].lower_ in LEADING_MODIFIERS
                            or toks[0].lower_ in STOPWORDS):
                toks.pop(0)
            while toks and toks[-1].pos_ in {"PUNCT", "PART", "DET", "SYM"}:
                toks.pop()
            if not toks or toks[-1].pos_ not in ("NOUN", "PROPN"):
                continue
            if len(toks) > 4:
                toks = toks[-4:]
            raw = doc[toks[0].i: toks[-1].i + 1].text
            key = key_of(raw)
            if not _valid_key(key):
                continue
            c = cands.setdefault(key, {"count": 0, "raws": Counter(), "force": False})
            c["count"] += 1
            c["raws"][raw] += 1
    return cands


_ADJ_SUFFIX = ("ical", "ful", "ous", "ive", "able", "ible", "less")


def _is_content_token(text: str, lemma: str) -> bool:
    if not is_word(text):
        return False
    low = text.lower()
    if lemma in STOPWORDS or low in STOPWORDS or lemma in VERBISH or low in VERBISH:
        return False
    if lemma in LEADING_MODIFIERS or low in LEADING_MODIFIERS:
        return False
    if len(low) > 5 and low.endswith("ly"):
        return False
    if len(low) > 5 and low.endswith(_ADJ_SUFFIX):
        return False
    if len(low) > 4 and low.endswith("ed"):  # past-tense / participle verbs ("surrounded")
        return False
    return True


def _ngram_candidates(sentences: List[Sentence], min_freq: int, max_n: int = 4) -> Dict[Key, dict]:
    counts: Counter = Counter()
    raws: Dict[Key, Counter] = defaultdict(Counter)
    for s in sentences:
        run: List[Tuple[str, str]] = []

        def flush():
            L = len(run)
            for n in range(1, max_n + 1):
                for i in range(L - n + 1):
                    seg = run[i:i + n]
                    key = tuple(l for l, _ in seg)
                    if _valid_key(key):
                        counts[key] += 1
                        raws[key][" ".join(r for _, r in seg)] += 1
            run.clear()

        for (text, _, _), lem in zip(s.tokens, s.lemmas):
            if _is_content_token(text, lem):
                run.append((lem, text))
            else:
                flush()
        flush()
    keep = {k for k, c in counts.items() if c >= min_freq}
    pruned = set()
    for h in keep:
        L = len(h)
        if L < 2:
            continue
        for n in range(1, L):
            for i in range(L - n + 1):
                g = h[i:i + n]
                if g in keep and g != h and counts[h] >= 0.6 * counts[g]:
                    pruned.add(g)
    return {k: {"count": counts[k], "raws": raws[k], "force": False} for k in keep - pruned}


# ---------------------------------------------------------------------------
# Store construction
# ---------------------------------------------------------------------------


def annotate_mentions(sentences: List[Sentence], matcher: ConceptMatcher) -> None:
    for s in sentences:
        s.mentions = []
        for i, j, cid in matcher.find(s.lemmas):
            s.mentions.append(Mention(i, j, cid, s.tokens[i][1], s.tokens[j - 1][2]))


def _count_mentions(sentences: List[Sentence], matcher: ConceptMatcher):
    c: Counter = Counter()
    paras: Dict[str, set] = defaultdict(set)
    for s in sentences:
        for _, _, cid in matcher.find(s.lemmas):
            c[cid] += 1
            paras[cid].add(s.para)
    return c, paras


def _display_name(key: Key, raws: Counter) -> str:
    """Name from the most common surface form per token; restores acronym / mixed-case casing."""
    tokens = list(key)
    votes: List[Counter] = [Counter() for _ in tokens]
    for raw, n in raws.items():
        rt = [t for t, _, _ in tokenize(raw) if is_word(t)]
        if len(rt) == len(tokens):
            for i, t in enumerate(rt):
                votes[i][t] += n
    out = []
    for i, tok in enumerate(tokens):
        best = tok
        if votes[i]:
            total_votes = sum(votes[i].values())
            upper = [r for r in votes[i] if 1 < len(r) <= 6 and r.isupper() and votes[i][r] >= 0.5 * total_votes]
            mixed = [r for r in votes[i] if any(ch.isupper() for ch in r[1:]) and not r.isupper()]
            if upper:
                best = max(upper, key=lambda r: votes[i][r])
            elif mixed and simple_lemma(mixed[0]) == tok:
                best = max(mixed, key=lambda r: votes[i][r])
            else:
                same = Counter()
                for r, n in votes[i].items():
                    if simple_lemma(r) == tok:
                        same[r.lower()] += n
                if same:
                    best = same.most_common(1)[0][0]
        out.append(best)
    return pretty_name(" ".join(out))


def build_concept_store(sentences: List[Sentence], docs=None, max_concepts: int = 60,
                        min_freq: int = 2, extra_terms: Optional[List[str]] = None,
                        exclude_terms: Optional[List[str]] = None, min_paragraphs: int = 2) -> ConceptStore:
    acr = find_acronyms(sentences)
    cands = _spacy_candidates(sentences, docs) if docs else _ngram_candidates(sentences, min_freq)
    # section headings are what the author considers the topics of the chapter
    skip_heading = {"unit", "part", "chapter", "contents", "summary", "introduction", "conclusion",
                    "references", "overview", "objectives", "outcomes", "prerequisites", "syllabus",
                    "questions", "exercises", "review", "agenda"}
    heading_keys = set()
    for s_ in sentences:
        if not s_.heading:
            continue
        words = [t for t, _, _ in s_.tokens if is_word(t)]
        if not 1 <= len(words) <= 5:
            continue
        phrase = " ".join(words)
        k = key_of(phrase)
        if len(k) == 1 and len(k[0]) <= 3 and not phrase.isupper():
            continue  # suffix fragments such as "-ed" / "-ing" are examples, not concepts (NLP, FSM stay)
        if k and k[0] not in skip_heading and _valid_key(k, allow_inner_stop=True):
            c = cands.setdefault(k, {"count": 0, "raws": Counter(), "force": True})
            c["force"] = True
            c["raws"][phrase] += 1
            heading_keys.add(k)
    for term in extra_terms or []:
        k = key_of(term)
        if k and len(k) <= 6 and not (len(k) == 1 and k[0] in STOPWORDS):
            c = cands.setdefault(k, {"count": 0, "raws": Counter(), "force": True})
            c["force"] = True
            c["raws"][term.strip()] += 1
    for term in exclude_terms or []:
        k = key_of(term)
        cands.pop(k, None)
        acr = {a: ph for a, ph in acr.items() if key_of(ph) != k and (a,) != k}

    alias_map: Dict[Key, Key] = {}
    for acr_l, phrase in acr.items():
        full = key_of(phrase)
        if not full or len(full) > 6 or not _valid_key(full, allow_inner_stop=True):
            continue
        ak = (acr_l,)
        alias_map[ak] = full
        c = cands.setdefault(full, {"count": 0, "raws": Counter(), "force": True})
        c["force"] = True
        c["raws"][phrase] += 1
        if ak in cands:
            c["count"] += cands[ak]["count"]
            del cands[ak]

    def pick(cs: Dict[Key, dict], limit: int):
        """Forced terms (headings, your searches, acronym expansions) are never dropped by the limit."""
        forced = [(k, v) for k, v in cs.items() if v["force"]]
        rest = [(k, v) for k, v in cs.items() if not v["force"] and v["count"] >= min_freq]
        rest.sort(key=lambda kv: kv[1]["count"] * (1 + 0.35 * (len(kv[0]) - 1)), reverse=True)
        return forced + rest[:limit]

    # pass 1: provisional set (over-generate), then count true mentions via longest match
    prov = pick(cands, max_concepts * 3)
    tmp_map: Dict[Key, str] = {}
    for idx, (k, _) in enumerate(prov):
        tmp_map[k] = f"t{idx}"
    for ak, fk in alias_map.items():
        if fk in tmp_map:
            tmp_map[ak] = tmp_map[fk]
    counts, para_sets = _count_mentions(sentences, ConceptMatcher(tmp_map))
    many_paras = len({s.para for s in sentences}) >= 12
    scored = []
    for k, v in prov:
        n = counts.get(tmp_map[k], 0)
        spread = len(para_sets.get(tmp_map[k], ()))
        # a real chapter concept recurs across the text; one-off illustration nouns (boy, apple) do not
        if many_paras and not v["force"] and n < 6 and spread < min_paragraphs:
            continue
        if (not v["force"] and len(k) == 1 and len(k[0]) <= 3
                and not any(r.strip().isupper() for r in v["raws"])):
            continue  # short lowercase words ("ed", "ing", "cat") are examples, not concepts (NLP, LDA stay)
        if n >= min_freq or (v["force"] and n >= 1):
            sc = n * (1 + 0.35 * (len(k) - 1)) * (0.7 if len(k) == 1 else 1.0) + (3 if v["force"] else 0)
            scored.append((k, v, n, sc))
    scored.sort(key=lambda t: t[3], reverse=True)
    forced_scored = [t for t in scored if t[1]["force"]]
    normal_scored = [t for t in scored if not t[1]["force"]]
    scored = forced_scored + normal_scored[:max(max_concepts - len(forced_scored), max_concepts // 2)]

    # pass 2: final ids / names
    concepts: Dict[str, Concept] = {}
    key_map: Dict[Key, str] = {}
    key_to_cid: Dict[Key, str] = {}
    for k, v, n, sc in scored:
        name = _display_name(k, v["raws"])
        cid = slugify(name)
        base, i = cid, 2
        while cid in concepts:
            cid = f"{base}_{i}"
            i += 1
        concepts[cid] = Concept(cid=cid, key=k, name=name, score=sc, heading=(k in heading_keys))
        key_map[k] = cid
        key_to_cid[k] = cid
    acronyms: Dict[str, str] = {}
    for ak, fk in alias_map.items():
        if fk in key_to_cid:
            cid = key_to_cid[fk]
            key_map[ak] = cid
            concepts[cid].aliases.append(ak[0].upper())
            acronyms[ak[0].upper()] = concepts[cid].name

    matcher = ConceptMatcher(key_map)
    annotate_mentions(sentences, matcher)
    store = ConceptStore(concepts=concepts, matcher=matcher, acronyms=acronyms)
    _fill_stats(store, sentences)
    _extract_definitions(store, sentences)
    return store


def _fill_stats(store: ConceptStore, sentences: List[Sentence]) -> None:
    pages: Dict[str, set] = defaultdict(set)
    sids: Dict[str, List[int]] = defaultdict(list)
    freq: Counter = Counter()
    for s in sentences:
        seen = set()
        for m in s.mentions:
            freq[m.cid] += 1
            pages[m.cid].add(s.page)
            if m.cid not in seen:
                sids[m.cid].append(s.sid)
                seen.add(m.cid)
    for cid, c in store.concepts.items():
        c.freq = freq.get(cid, 0)
        c.pages = sorted(pages.get(cid, []))
        c.sentence_ids = sids.get(cid, [])


_DEF_VERBS = {"is", "are", "refer", "refers", "denote", "denotes", "mean", "means", "represent", "represents"}


_NOT_A_VERB = {",", ".", ";", ":", "and", "or", "of", "in", "on", "for", "to", "with", "as", "that", "which",
               "by", "from", "at", "than", "but", "if", "when", "while", "also"}


def _extract_definitions(store: ConceptStore, sentences: List[Sentence]) -> None:
    """Best explaining sentence per concept.
    rank 0 = the document explains it (formal 'X is ...' definition preferred, else a sentence where X is the
    subject); rank 1 = X starts a sentence only loosely; rank 2 = merely mentioned."""
    best: Dict[str, Tuple[Tuple[int, int], Sentence]] = {}
    for s in sentences:
        n_words = sum(1 for t, _, _ in s.tokens if is_word(t))
        if n_words < 5 or n_words > 55 or s.heading:
            continue
        low = [t[0].lower() for t in s.tokens]
        for m in s.mentions:
            after = low[m.end_tok:]
            if len(after) >= 3 and after[0] == "(" and after[2] == ")":
                after = after[3:]  # "Natural Language Processing (NLP) turns ..." -> look past the acronym
            nxt = after[:3]
            prev = low[m.start_tok - 1] if m.start_tok > 0 else ""
            subject_like = m.start_tok == 0 or prev in {"the", "a", "an", ",", ":", "-", "\u2013", "\u2014", "in", "for"}
            rank, formal = 2, 1
            if m.start_tok <= 6 and nxt and subject_like:
                if nxt[0] in _DEF_VERBS or nxt[:3] == ["can", "be", "defined"] or nxt[:2] == ["also", "called"]:
                    rank, formal = 0, 0
                elif m.start_tok <= 2 and nxt[0].isalpha() and nxt[0] not in _NOT_A_VERB:
                    rank, formal = 0, 1  # "X splits text into tokens": the document explains X
            if rank == 2 and m.start_tok <= 6:
                rank = 1
            cur = best.get(m.cid)
            if cur is None or (rank, formal) < cur[0]:
                best[m.cid] = ((rank, formal), s)
    for cid, ((rank, _), s) in best.items():
        c = store.concepts[cid]
        text = s.text if len(s.text) <= 320 else s.text[:317].rstrip() + "..."
        c.definition = text
        c.definition_page = s.page
        c.def_rank = rank

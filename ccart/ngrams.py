"""N-gram based co-occurrence (Unit V).

Concepts that appear within a sliding window of N tokens in the same paragraph get a
weak edge.  Also provides word n-gram frequency tables and PMI collocations.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import GENERIC_SINGLE, STOPWORDS, VERBISH
from .preprocess import Sentence, is_word


@dataclass
class CoocEdge:
    a: str
    b: str
    count: int
    strength: float  # count / sqrt(freq_a * freq_b)


def cooccurrence_edges(sentences: List[Sentence], freqs: Dict[str, int], window: int = 12,
                       min_count: int = 2, max_per_node: int = 3) -> List[CoocEdge]:
    """Sliding N-token window over each paragraph; keep the strongest edges per concept."""
    by_para: Dict[int, List[Sentence]] = defaultdict(list)
    for s in sentences:
        by_para[s.para].append(s)
    counts: Counter = Counter()
    for sents in by_para.values():
        positions: List[Tuple[int, str]] = []
        offset = 0
        for s in sents:
            for m in s.mentions:
                positions.append((offset + m.start_tok, m.cid))
            offset += len(s.tokens)
        positions.sort()
        for i, (pi, ci) in enumerate(positions):
            for pj, cj in positions[i + 1:]:
                if pj - pi > window:
                    break
                if ci != cj:
                    counts[tuple(sorted((ci, cj)))] += 1
    edges = []
    for (a, b), c in counts.items():
        if c < min_count:
            continue
        denom = math.sqrt(max(freqs.get(a, 1), 1) * max(freqs.get(b, 1), 1))
        edges.append(CoocEdge(a, b, c, c / denom))
    edges.sort(key=lambda e: (-e.strength, -e.count))
    per_node: Counter = Counter()
    kept = []
    for e in edges:
        if per_node[e.a] >= max_per_node and per_node[e.b] >= max_per_node:
            continue
        kept.append(e)
        per_node[e.a] += 1
        per_node[e.b] += 1
    return kept


def _content_words(s: Sentence) -> List[str]:
    return [l.lower() for (t, _, _), l in zip(s.tokens, s.lemmas) if is_word(t)]


def ngram_tables(sentences: List[Sentence], top_k: int = 25) -> Dict[str, List[Tuple[str, int]]]:
    """Frequent unigrams/bigrams/trigrams (stop-word bounded n-grams removed)."""
    tables: Dict[str, Counter] = {"unigram": Counter(), "bigram": Counter(), "trigram": Counter()}
    for s in sentences:
        w = _content_words(s)
        for n, name in ((1, "unigram"), (2, "bigram"), (3, "trigram")):
            for i in range(len(w) - n + 1):
                g = w[i:i + n]
                if g[0] in STOPWORDS or g[-1] in STOPWORDS:
                    continue
                if n == 1 and (g[0] in GENERIC_SINGLE or g[0] in VERBISH or len(g[0]) < 3):
                    continue
                if n > 1 and (g[0] in VERBISH or g[-1] in VERBISH):
                    continue
                tables[name][" ".join(g)] += 1
    return {k: v.most_common(top_k) for k, v in tables.items()}


def pmi_collocations(sentences: List[Sentence], min_count: int = 2, top_k: int = 20) -> List[Tuple[str, int, float]]:
    """Bigram collocations ranked by pointwise mutual information."""
    uni, bi = Counter(), Counter()
    total = 0
    for s in sentences:
        w = _content_words(s)
        uni.update(w)
        total += len(w)
        for a, b in zip(w, w[1:]):
            bi[(a, b)] += 1
    n_bi = max(sum(bi.values()), 1)
    out = []
    for (a, b), c in bi.items():
        if c < min_count or a in STOPWORDS or b in STOPWORDS or a in VERBISH or b in VERBISH:
            continue
        pmi = math.log2((c / n_bi) / ((uni[a] / total) * (uni[b] / total)))
        out.append((f"{a} {b}", c, pmi))
    out.sort(key=lambda t: (-t[2], -t[1]))
    return out[:top_k]

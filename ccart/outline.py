"""Document outline: heading hierarchy (6 -> 6.1 -> 6.2) and what each section covers.

This is how the map reflects the structure of the chapter: a section heading is linked to its
sub-sections, and to the concepts that are explained inside it.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from .concepts import ConceptStore
from .preprocess import Sentence, is_word


def heading_concept(s: Sentence) -> Optional[str]:
    """Concept that (almost) equals the heading text, if any."""
    nwords = sum(1 for t, _, _ in s.tokens if is_word(t))
    best, best_len = None, 0
    for m in s.mentions:
        n = sum(1 for i in range(m.start_tok, m.end_tok) if is_word(s.tokens[i][0]))
        if n > best_len:
            best, best_len = m.cid, n
    return best if best and best_len >= 0.6 * nwords else None


def build_outline(sentences: List[Sentence], store: ConceptStore, per_section: int = 8,
                  min_mentions: int = 2) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], int]]:
    """Return (subtopic edges parent->child, section-coverage counts {(heading, concept): n})."""
    current: Dict[str, str] = {}
    subtopics: List[Tuple[str, str]] = []
    cur_h: Optional[str] = None
    counts: Dict[str, Counter] = defaultdict(Counter)
    for s in sentences:
        if s.heading:
            cid = heading_concept(s)
            cur_h = cid
            if s.num and cid:
                parts = s.num.split(".")
                for k in range(len(parts) - 1, 0, -1):
                    parent = current.get(".".join(parts[:k]))
                    if parent and parent != cid:
                        subtopics.append((parent, cid))
                        break
                current[s.num] = cid
                for key in [n for n in current if n.startswith(s.num + ".")]:
                    del current[key]  # a new section closes its old sub-sections
            continue
        if cur_h:
            for m in s.mentions:
                if m.cid != cur_h:
                    counts[cur_h][m.cid] += 1
    section = {(h, c): n for h, cn in counts.items() for c, n in cn.most_common(per_section) if n >= min_mentions}
    return list(dict.fromkeys(subtopics)), section

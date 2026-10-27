"""N-gram language models with add-k smoothing and perplexity (Unit V: LM evaluation)."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from .preprocess import Sentence, is_word

BOS, EOS, UNK = "<s>", "</s>", "<UNK>"


def lm_sentences(sentences: List[Sentence]) -> List[List[str]]:
    out = []
    for s in sentences:
        toks = []
        for (t, _, _), lem in zip(s.tokens, s.lemmas):
            if is_word(t):
                toks.append(lem.lower())
            elif t[:1].isdigit():
                toks.append("<num>")
        if len(toks) >= 2:
            out.append(toks)
    return out


class NGramLM:
    def __init__(self, n: int = 2, k: float = 1.0):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n, self.k = n, k
        self.counts: Counter = Counter()
        self.ctx_counts: Counter = Counter()
        self.vocab: set = set()

    def _pad(self, sent: Sequence[str]) -> List[str]:
        return [BOS] * (self.n - 1) + list(sent) + [EOS]

    def fit(self, sents: List[List[str]]) -> "NGramLM":
        self.counts, self.ctx_counts, self.vocab = Counter(), Counter(), {EOS, UNK}
        for s in sents:
            self.vocab.update(s)
            p = self._pad(s)
            for i in range(self.n - 1, len(p)):
                ctx = tuple(p[i - self.n + 1:i])
                self.counts[ctx + (p[i],)] += 1
                self.ctx_counts[ctx] += 1
        return self

    @property
    def V(self) -> int:
        return len(self.vocab)

    def prob(self, word: str, ctx: Tuple[str, ...]) -> float:
        if word not in self.vocab:
            word = UNK
        ctx = tuple(c if (c in self.vocab or c == BOS) else UNK for c in ctx)
        return (self.counts[ctx + (word,)] + self.k) / (self.ctx_counts[ctx] + self.k * self.V)

    def perplexity(self, sents: List[List[str]]) -> float:
        log_sum, n = 0.0, 0
        for s in sents:
            p = self._pad(s)
            for i in range(self.n - 1, len(p)):
                ctx = tuple(p[i - self.n + 1:i])
                log_sum += math.log(self.prob(p[i], ctx))
                n += 1
        return math.exp(-log_sum / n) if n else float("nan")

    def predict_next(self, context: Sequence[str], topn: int = 5) -> List[Tuple[str, float]]:
        """Most likely next words (relative frequency), backing off to shorter contexts."""
        context = [c.lower() for c in context]
        for order in range(self.n - 1, -1, -1):
            ctx = tuple(([BOS] * order + context)[-order:]) if order else ()
            cands = _backoff_counts(self, ctx)
            total = sum(c for _, c in cands)
            if total:
                cands.sort(key=lambda t: -t[1])
                return [(w, c / total) for w, c in cands[:topn]]
        return []


def _backoff_counts(lm: NGramLM, ctx: Tuple[str, ...]) -> List[Tuple[str, int]]:
    agg: Counter = Counter()
    L = len(ctx)
    for gram, c in lm.counts.items():
        if gram[-1] == EOS:
            continue
        if L == 0 or gram[-L - 1:-1] == ctx:
            agg[gram[-1]] += c
    return list(agg.items())


def split_train_test(sents: List[List[str]], every: int = 5) -> Tuple[List[List[str]], List[List[str]]]:
    if len(sents) < 10:
        return sents, sents
    train = [s for i, s in enumerate(sents) if i % every != 0]
    test = [s for i, s in enumerate(sents) if i % every == 0]
    return train, test


def evaluate_models(sentences: List[Sentence], max_n: int = 3,
                    smoothings: Tuple[float, ...] = (1.0, 0.1, 0.01)) -> Dict:
    """Train unigram..N-gram LMs for several add-k values and report train/test perplexity."""
    sents = lm_sentences(sentences)
    train, test = split_train_test(sents)
    rows = []
    for n in range(1, max_n + 1):
        for k in smoothings:
            lm = NGramLM(n, k).fit(train)
            rows.append({"n": n, "model": {1: "Unigram", 2: "Bigram", 3: "Trigram"}.get(n, f"{n}-gram"),
                         "add-k": k, "train perplexity": lm.perplexity(train),
                         "test perplexity": lm.perplexity(test)})
    best = min(rows, key=lambda r: r["test perplexity"]) if rows else None
    return {"rows": rows, "best": best, "n_train": len(train), "n_test": len(test),
            "held_out": train is not test, "vocab": len({w for s in train for w in s})}


def build_predictor(sentences: List[Sentence], n: int = 3, k: float = 0.1) -> NGramLM:
    return NGramLM(n, k).fit(lm_sentences(sentences))

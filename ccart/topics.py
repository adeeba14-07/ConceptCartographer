"""LDA topic modelling (Gensim) + coherence-score evaluation (Unit V).

Documents are small windows of consecutive sentences.  Multi-word concepts are merged into
single tokens so topics are expressed in concepts.  Number of topics is chosen by sweeping k
and maximising coherence (c_v when Gensim is available, otherwise NPMI).  If Gensim cannot be
imported the module falls back to scikit-learn's LDA so the app still works.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from .concepts import ConceptStore
from .config import GENERIC_SINGLE, LEADING_MODIFIERS, STOPWORDS, VERBISH
from .preprocess import Sentence, is_word


def gensim_available() -> bool:
    try:
        import gensim  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@dataclass
class TopicModelResult:
    engine: str
    num_topics: int
    topics: List[dict]
    concept_topic: Dict[str, dict]
    coherence_overall: Dict[str, Optional[float]]
    sweep: List[dict]
    n_docs: int
    vocab_size: int
    primary_metric: str = "npmi"
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def build_documents(sentences: List[Sentence], store: ConceptStore, window: int = 3,
                    concepts_only: bool = False):
    word_counts: Counter = Counter(
        l.lower() for s in sentences for (t, _, _), l in zip(s.tokens, s.lemmas) if is_word(t))
    docs: List[List[str]] = []
    doc_cids: List[Counter] = []
    for start in range(0, len(sentences), window):
        chunk = sentences[start:start + window]
        toks: List[str] = []
        cc: Counter = Counter()
        for s in chunk:
            by_start = {m.start_tok: m for m in s.mentions}
            i = 0
            while i < len(s.tokens):
                if i in by_start:
                    m = by_start[i]
                    toks.append(m.cid)
                    cc[m.cid] += 1
                    i = m.end_tok
                    continue
                text, lem = s.tokens[i][0], s.lemmas[i].lower()
                if concepts_only:
                    i += 1
                    continue
                if (is_word(text) and len(lem) > 2 and lem not in STOPWORDS and lem not in VERBISH
                        and lem not in GENERIC_SINGLE and lem not in LEADING_MODIFIERS
                        and word_counts[lem] >= 2 and not text.lower().endswith("ly")):
                    toks.append(lem)
                i += 1
        if len(toks) >= (2 if concepts_only else 3):
            docs.append(toks)
            doc_cids.append(cc)
    if concepts_only:
        return docs, doc_cids
    # drop words that occur in a large share of documents: they blur every topic together
    if len(docs) >= 10:
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        limit = 0.35 * len(docs)
        too_common = {w for w, c in df.items() if c > limit}
        if too_common:
            new_docs, new_cids = [], []
            for d, cc in zip(docs, doc_cids):
                nd = [w for w in d if w not in too_common]
                if len(nd) >= 3:
                    new_docs.append(nd)
                    new_cids.append(Counter({k: v for k, v in cc.items() if k not in too_common}))
            if len(new_docs) >= 5:
                docs, doc_cids = new_docs, new_cids
    return docs, doc_cids


# ---------------------------------------------------------------------------
# Coherence (own implementations: UMass + NPMI over document co-occurrence)
# ---------------------------------------------------------------------------


def _doc_freqs(docs: List[List[str]]):
    sets = [set(d) for d in docs]
    df: Counter = Counter()
    for s in sets:
        df.update(s)
    return sets, df


def coherence_scores(topic_words: List[List[str]], docs: List[List[str]]) -> Dict[str, List[float]]:
    sets, df = _doc_freqs(docs)
    N = len(sets)
    cache: Dict[Tuple[str, str], int] = {}

    def D2(a: str, b: str) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in cache:
            cache[key] = sum(1 for s in sets if a in s and b in s)
        return cache[key]

    umass, npmi = [], []
    for words in topic_words:
        ws = [w for w in words if df.get(w, 0) > 0][:10]
        um, nm = [], []
        for i in range(1, len(ws)):
            for j in range(i):
                d_ij = D2(ws[i], ws[j])
                um.append(math.log((d_ij + 1) / df[ws[j]]))
        for a, b in combinations(ws, 2):
            d_ab = D2(a, b)
            if d_ab == 0:
                nm.append(-1.0)
            else:
                p_ab = d_ab / N
                if p_ab >= 1.0:
                    nm.append(1.0)
                else:
                    pmi = math.log(p_ab / ((df[a] / N) * (df[b] / N)))
                    nm.append(pmi / (-math.log(p_ab)))
        umass.append(float(np.mean(um)) if um else float("nan"))
        npmi.append(float(np.mean(nm)) if nm else float("nan"))
    return {"u_mass": umass, "npmi": npmi}


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


@dataclass
class _Fit:
    topic_words: List[List[Tuple[str, float]]]
    doc_topic: np.ndarray
    extra: tuple = ()


def _fit_gensim(docs: List[List[str]], k: int, seed: int) -> _Fit:
    from gensim import corpora, models

    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=1, no_above=0.85)
    if len(dictionary) < 5:
        dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(d) for d in docs]
    passes = 20 if len(docs) < 800 else 8
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, random_state=seed,
                          passes=passes, iterations=100, chunksize=max(100, len(docs)), alpha="auto")
    tw = [[(w, float(p)) for w, p in lda.show_topic(t, topn=10)] for t in range(k)]
    dt = np.zeros((len(docs), k))
    for i, bow in enumerate(corpus):
        if not bow:
            dt[i, :] = 1.0 / k
            continue
        for t, p in lda.get_document_topics(bow, minimum_probability=0.0):
            dt[i, t] = p
    return _Fit(tw, dt, (lda, dictionary, corpus))


def _fit_sklearn(docs: List[List[str]], k: int, seed: int) -> _Fit:
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(analyzer=lambda x: x)
    X = vec.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=k, random_state=seed, max_iter=40, learning_method="batch")
    dt = lda.fit_transform(X)
    comps = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
    vocab = vec.get_feature_names_out()
    tw = []
    for t in range(k):
        idx = np.argsort(-comps[t])[:10]
        tw.append([(str(vocab[i]), float(comps[t, i])) for i in idx])
    return _Fit(tw, dt)


def _gensim_cv(fit: _Fit, docs: List[List[str]]) -> Tuple[Optional[float], Optional[List[float]]]:
    try:
        from gensim.models.coherencemodel import CoherenceModel

        dictionary = fit.extra[1]
        topics = [[w for w, _ in tw if w in dictionary.token2id] for tw in fit.topic_words]
        cm = CoherenceModel(topics=topics, texts=docs, dictionary=dictionary, coherence="c_v", processes=1)
        per = [float(x) for x in cm.get_coherence_per_topic()]
        overall = float(cm.get_coherence())
        if math.isnan(overall):
            return None, None
        return overall, per
    except Exception:  # noqa: BLE001
        return None, None


def _evaluate(fit: _Fit, docs: List[List[str]], use_gensim: bool) -> dict:
    words = [[w for w, _ in tw] for tw in fit.topic_words]
    own = coherence_scores(words, docs)
    res = {"npmi": float(np.nanmean(own["npmi"])), "u_mass": float(np.nanmean(own["u_mass"])),
           "per_npmi": own["npmi"], "per_umass": own["u_mass"], "c_v": None, "per_cv": None}
    if use_gensim:
        cv, per = _gensim_cv(fit, docs)
        res["c_v"], res["per_cv"] = cv, per
    return res


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_topic_modeling(sentences: List[Sentence], store: ConceptStore, num_topics: Optional[int] = None,
                       k_min: int = 2, k_max: int = 8, window: int = 3, seed: int = 42,
                       concepts_only: bool = True) -> TopicModelResult:
    docs, doc_cids = [], []
    if concepts_only:  # topics expressed purely in concepts are far easier to read
        docs, doc_cids = build_documents(sentences, store, window, concepts_only=True)
        if len(docs) < 8 or len({w for d in docs for w in d}) < 6:
            docs, doc_cids = [], []
    if not docs:
        docs, doc_cids = build_documents(sentences, store, window)
    notes: List[str] = []
    use_gensim = gensim_available()
    engine = "gensim" if use_gensim else "scikit-learn (Gensim not installed)"
    vocab = {w for d in docs for w in d}

    if len(docs) < 3 or len(vocab) < 6:
        notes.append("Text too short for topic modelling - all concepts placed in a single topic.")
        words = Counter(w for d in docs for w in d).most_common(10)
        tot = max(sum(c for _, c in words), 1)
        topic = {"id": 0, "words": [(w, c / tot) for w, c in words], "label": "All concepts",
                 "coherence": {}}
        ct = {cid: {"topic": 0, "confidence": 1.0, "distribution": [1.0]} for cid in store.concepts}
        return TopicModelResult(engine, 1, [topic], ct, {}, [], len(docs), len(vocab), notes=notes)

    fit_fn = _fit_gensim if use_gensim else _fit_sklearn
    k_max = max(2, min(k_max, len(docs) - 1, len(vocab) - 1))
    k_min = max(2, min(k_min, k_max))
    sweep, fits = [], {}
    ks = [num_topics] if num_topics else list(range(k_min, k_max + 1))
    for k in ks:
        try:
            fit = fit_fn(docs, k, seed)
        except Exception as exc:  # noqa: BLE001
            if use_gensim:
                notes.append(f"Gensim failed ({exc}); switched to scikit-learn LDA.")
                use_gensim, fit_fn, engine = False, _fit_sklearn, "scikit-learn (Gensim failed)"
                fit = fit_fn(docs, k, seed)
            else:
                raise
        ev = _evaluate(fit, docs, use_gensim)
        fits[k] = (fit, ev)
        sweep.append({"k": k, "c_v": ev["c_v"], "npmi": ev["npmi"], "u_mass": ev["u_mass"]})

    primary = "c_v" if (use_gensim and all(r["c_v"] is not None for r in sweep)) else "npmi"
    if num_topics:
        best_k = ks[0]
    else:  # prefer the smallest k whose coherence is within 0.02 of the best: cleaner, fewer topics
        top = max(r[primary] for r in sweep)
        best_k = min(r["k"] for r in sweep if r[primary] >= top - 0.02)
    fit, ev = fits[best_k]

    # concept -> topic through the document-topic distributions of the docs where it occurs
    ct: Dict[str, dict] = {}
    for cid in store.concepts:
        dist = np.zeros(best_k)
        for i, cc in enumerate(doc_cids):
            if cid in cc:
                dist += cc[cid] * fit.doc_topic[i]
        if dist.sum() <= 0:
            ct[cid] = {"topic": 0, "confidence": 0.0, "distribution": [1.0 / best_k] * best_k}
        else:
            dist = dist / dist.sum()
            ct[cid] = {"topic": int(dist.argmax()), "confidence": float(dist.max()),
                       "distribution": [float(x) for x in dist]}

    def pretty(w: str) -> str:
        return store.concepts[w].name if w in store.concepts else w.replace("_", " ").title()

    topics = []
    for t, tw in enumerate(fit.topic_words):
        concept_words = [w for w, _ in tw if w in store.concepts]
        chosen = concept_words[:3] if len(concept_words) >= 2 else [w for w, _ in tw[:3]]
        label = " / ".join(pretty(w) for w in chosen)
        coh = {"npmi": ev["per_npmi"][t], "u_mass": ev["per_umass"][t]}
        if ev["per_cv"]:
            coh["c_v"] = ev["per_cv"][t]
        topics.append({"id": t, "words": [(pretty(w), p) for w, p in tw], "label": label, "coherence": coh})
    overall = {"c_v": ev["c_v"], "npmi": ev["npmi"], "u_mass": ev["u_mass"]}
    return TopicModelResult(engine, best_k, topics, ct, overall, sweep, len(docs), len(vocab),
                            primary_metric=primary, notes=notes)


def smooth_with_graph(result: TopicModelResult, neighbors: Dict[str, List[Tuple[str, float]]],
                      alpha: float = 0.4) -> None:
    """Pull each concept's topic towards its graph neighbours so linked concepts share colours."""
    if result.num_topics < 2:
        return
    base = {cid: np.array(v["distribution"], dtype=float) for cid, v in result.concept_topic.items()}
    for cid, v in result.concept_topic.items():
        nb = [(n, w) for n, w in neighbors.get(cid, []) if n in base]
        if not nb:
            continue
        tot = sum(w for _, w in nb)
        mix = sum(base[n] * w for n, w in nb) / tot if tot else 0
        dist = (1 - alpha) * base[cid] + alpha * mix
        dist = dist / dist.sum() if dist.sum() else dist
        v["distribution"] = [float(x) for x in dist]
        v["topic"] = int(dist.argmax())
        v["confidence"] = float(dist.max())


def relabel_by_members(result: TopicModelResult, store: ConceptStore, per: int = 3) -> None:
    """Name each topic after its own strongest concepts, so no concept names several topics."""
    if result.num_topics < 2:
        return
    for t in result.topics:
        members = [(cid, v["confidence"] * math.log(1 + store.concepts[cid].freq))
                   for cid, v in result.concept_topic.items() if v["topic"] == t["id"]]
        members.sort(key=lambda x: -x[1])
        if members:
            t["label"] = " / ".join(store.name(c) for c, _ in members[:per])

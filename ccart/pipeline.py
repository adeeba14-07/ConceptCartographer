"""End-to-end ConceptCartographer pipeline.

PDF/text -> clean -> sentences -> (spaCy parse) -> concepts -> triples -> n-gram edges
-> LDA topics + coherence -> NetworkX graph -> meaning representations -> language models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import networkx as nx

from .concepts import ConceptStore, build_concept_store
from .explain import build_explanations
from .graph import build_graph, build_meaning_representations
from .langmodel import NGramLM, build_predictor, evaluate_models
from .outline import build_outline
from .ngrams import CoocEdge, cooccurrence_edges, ngram_tables, pmi_collocations
from .preprocess import Sentence, build_sentences, clean_pages
from .topics import TopicModelResult, relabel_by_members, run_topic_modeling, smooth_with_graph
from .triples import Triple, extract_triples


_LAST_DOCS = {"key": None, "docs": None}  # re-use the spaCy parse when only settings change


@dataclass
class Params:
    max_concepts: int = 60
    min_freq: int = 2
    strict_relations: bool = False
    cooc_window: int = 12
    cooc_min_count: int = 2
    cooc_max_per_node: int = 3
    auto_topics: bool = True
    num_topics: int = 4
    k_min: int = 3
    k_max: int = 6
    lda_window: int = 4
    drop_examples: bool = True
    min_paragraphs: int = 2
    concepts_only_topics: bool = True
    first_page: int = 1
    max_chars: int = 400_000
    extra_terms: List[str] = field(default_factory=list)
    exclude_terms: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    params: Params
    sentences: List[Sentence]
    store: ConceptStore
    triples: List[Triple]
    cooc: List[CoocEdge]
    topics: TopicModelResult
    G: nx.DiGraph
    P: nx.DiGraph
    frames: List[dict]
    ngram_tables: Dict
    collocations: List
    lm_report: Dict
    predictor: NGramLM
    stats: Dict
    warnings: List[str] = field(default_factory=list)
    parser: str = "rule-based (no spaCy model)"
    explain: Dict[str, dict] = field(default_factory=dict)  # definition / how it works / connections per concept
    explain_version: int = 0
    ai_report: Dict = field(default_factory=dict)


def run_pipeline(pages: List[str], params: Params, nlp=None,
                 progress: Optional[Callable[[float, str], None]] = None) -> PipelineResult:
    def tick(frac: float, msg: str):
        if progress:
            progress(frac, msg)

    warnings: List[str] = []
    tick(0.03, "Cleaning text...")
    pages = clean_pages(pages)
    total = 0
    trimmed = []
    for p in pages:
        if total + len(p) > params.max_chars:
            warnings.append(f"Input truncated to {params.max_chars:,} characters for speed.")
            break
        trimmed.append(p)
        total += len(p)
    pages = trimmed

    tick(0.08, "Splitting sentences and tokenizing...")
    sentences = build_sentences(pages, first_page=params.first_page, drop_examples=params.drop_examples)
    if len(sentences) < 5:
        raise ValueError("Not enough readable text was found (need at least 5 sentences). "
                         "If this is a scanned PDF, run OCR first.")

    docs = None
    parser = "rule-based (no spaCy model)"
    if nlp is not None and nlp.has_pipe("parser"):
        tick(0.15, "Parsing sentences with spaCy (dependencies, POS, NER)...")
        dkey = hash(tuple(s.text for s in sentences))
        if _LAST_DOCS["key"] == dkey:
            docs = _LAST_DOCS["docs"]
        else:
            docs = list(nlp.pipe([s.text for s in sentences], batch_size=64))
            _LAST_DOCS["key"], _LAST_DOCS["docs"] = dkey, docs
        parser = f"spaCy {nlp.meta.get('name', 'model')}"
    else:
        warnings.append("spaCy model not available: using the rule-based concept and relation extractors.")

    tick(0.35, "Extracting concepts...")
    max_c = params.max_concepts if params.max_concepts > 0 else max(40, min(150, len(sentences) // 5))
    store = build_concept_store(sentences, docs, max_c, params.min_freq,
                                params.extra_terms, params.exclude_terms, params.min_paragraphs)
    if len(store.concepts) < 3:
        raise ValueError("Fewer than 3 concepts were found. Try lowering 'Minimum concept frequency' "
                         "or provide a longer chapter.")

    tick(0.45, "Extracting predicate-argument triples...")
    triples = extract_triples(sentences, docs, strict=params.strict_relations)
    docs = None  # free memory

    outline = build_outline(sentences, store)

    tick(0.55, "Finding n-gram co-occurrences...")
    freqs = {cid: c.freq for cid, c in store.concepts.items()}
    cooc = cooccurrence_edges(sentences, freqs, params.cooc_window, params.cooc_min_count,
                              params.cooc_max_per_node)

    tick(0.65, "Training LDA topic models and scoring coherence...")
    topics = run_topic_modeling(sentences, store, None if params.auto_topics else params.num_topics,
                                params.k_min, params.k_max, params.lda_window,
                                concepts_only=params.concepts_only_topics)
    nbrs: Dict[str, list] = {}
    for t in triples:
        nbrs.setdefault(t.subj, []).append((t.obj, 2.0 * t.count))
        nbrs.setdefault(t.obj, []).append((t.subj, 2.0 * t.count))
    for e in cooc:
        nbrs.setdefault(e.a, []).append((e.b, e.strength))
        nbrs.setdefault(e.b, []).append((e.a, e.strength))
    for parent, child in outline[0]:
        nbrs.setdefault(parent, []).append((child, 3.0))
        nbrs.setdefault(child, []).append((parent, 3.0))
    for (h, c), n in outline[1].items():
        nbrs.setdefault(h, []).append((c, 1.0))
        nbrs.setdefault(c, []).append((h, 1.0))
    smooth_with_graph(topics, nbrs)
    relabel_by_members(topics, store)
    warnings.extend(topics.notes)

    tick(0.85, "Building the knowledge graph...")
    G, P = build_graph(store, triples, cooc, topics, outline)
    frames = build_meaning_representations(store, triples, G, P, topics)
    explain = build_explanations(sentences, store, G)
    for f in frames:  # the frame carries the final definition and how it works
        e = explain[f["id"]]
        f["definition"], f["definition_source"], f["how_it_works"] = e["definition"], e["def_source"], e["how"]

    tick(0.92, "Evaluating n-gram language models...")
    tables = ngram_tables(sentences)
    colloc = pmi_collocations(sentences)
    lm_report = evaluate_models(sentences)
    predictor = build_predictor(sentences)

    und = G.to_undirected()
    stats = {
        "sentences": len(sentences), "pages": len({s.page for s in sentences}),
        "concepts": G.number_of_nodes(),
        "triples": len(triples), "triple_edges": sum(1 for *_, d in G.edges(data=True) if d["kind"] == "triple"),
        "weak_edges": sum(1 for *_, d in G.edges(data=True) if d["kind"] in ("cooccur", "section")),
        "outline_edges": sum(1 for *_, d in G.edges(data=True) if d["kind"] == "structure"),
        "prereq_edges": P.number_of_edges(),
        "topics": topics.num_topics,
        "components": nx.number_connected_components(und) if len(und) else 0,
        "isolated": sum(1 for n in G if und.degree(n) == 0),
        "density": round(nx.density(und), 4) if len(und) > 1 else 0.0,
    }
    tick(1.0, "Done.")
    return PipelineResult(params, sentences, store, triples, cooc, topics, G, P, frames, tables, colloc,
                          lm_report, predictor, stats, warnings, parser, explain)

"""Knowledge-graph construction (NetworkX) and meaning representation (JSON per concept)."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .concepts import ConceptStore
from .config import REL_COLORS, TOPIC_PALETTE, UNASSIGNED_COLOR
from .ngrams import CoocEdge
from .topics import TopicModelResult
from .triples import Triple

# Priority used to choose the "headline" relation of an edge that has several.
_REL_PRIORITY = ["is_a", "part_of", "has_part", "requires", "used_for", "evaluates", "improves",
                 "produces", "operates_on", "defined_as", "related"]


def topic_color(topic_id: Optional[int]) -> str:
    if topic_id is None:
        return UNASSIGNED_COLOR
    return TOPIC_PALETTE[topic_id % len(TOPIC_PALETTE)]


def build_graph(store: ConceptStore, triples: List[Triple], cooc: List[CoocEdge],
                topics: Optional[TopicModelResult], outline=None) -> Tuple[nx.DiGraph, nx.DiGraph]:
    """Return (G, P): G the knowledge graph, P the prerequisite DAG-ish graph (prereq -> dependent)."""
    G = nx.DiGraph()
    for cid, c in store.concepts.items():
        t = topics.concept_topic.get(cid) if topics else None
        G.add_node(cid, label=c.name, freq=c.freq, page=(c.pages[0] if c.pages else None),
                   definition=c.definition, topic=(t["topic"] if t else None),
                   topic_conf=(t["confidence"] if t else 0.0), aliases=list(c.aliases), heading=c.heading)

    by_pair: Dict[Tuple[str, str], List[Triple]] = defaultdict(list)
    for t in triples:
        if t.subj in G and t.obj in G:
            by_pair[(t.subj, t.obj)].append(t)
    for (u, v), ts in by_pair.items():
        ts_sorted = sorted(ts, key=lambda x: (_REL_PRIORITY.index(x.rel_type), -x.count))
        head = ts_sorted[0]
        G.add_edge(u, v, kind="triple", relation=head.rel, rel_type=head.rel_type,
                   weight=sum(x.count for x in ts), cost=1.0,
                   relations=[{"rel": x.rel, "type": x.rel_type, "count": x.count, "page": x.page,
                               "sentence": x.sentence, "source": x.source} for x in ts_sorted])
    sub_edges, section_counts = outline if outline else ([], {})
    for parent, child in sub_edges:  # document outline: section -> sub-section
        if parent in G and child in G and not G.has_edge(parent, child):
            if G.has_edge(child, parent):
                G.remove_edge(child, parent)
            G.add_edge(parent, child, kind="structure", relation="has sub-topic", rel_type="subtopic",
                       weight=3, cost=0.9, relations=[])
    for (h, c), n in section_counts.items():  # concepts explained inside a section
        if h in G and c in G and not G.has_edge(h, c) and not G.has_edge(c, h):
            G.add_edge(h, c, kind="section", relation="covers", rel_type="covers", weight=n, cost=1.8,
                       relations=[])
    for e in cooc:
        if G.has_edge(e.a, e.b) or G.has_edge(e.b, e.a):
            continue
        G.add_edge(e.a, e.b, kind="cooccur", relation="co-occurs with", rel_type="cooccur",
                   weight=e.count, strength=e.strength, cost=2.5, relations=[])

    P = nx.DiGraph()
    P.add_nodes_from(G.nodes)
    for parent, child in sub_edges:  # overview first, then its sub-topics
        if parent in P and child in P and parent != child and not P.has_edge(child, parent):
            P.add_edge(parent, child, weight=3, relation="has sub-topic", rel_type="subtopic")
    for t in triples:
        if t.prereq_dir == "obj_to_subj":
            pre, dep = t.obj, t.subj
        elif t.prereq_dir == "subj_to_obj":
            pre, dep = t.subj, t.obj
        else:
            continue
        if pre == dep or pre not in P or dep not in P:
            continue
        if P.has_edge(pre, dep):
            P[pre][dep]["weight"] += t.count
        else:
            P.add_edge(pre, dep, weight=t.count, relation=t.rel, rel_type=t.rel_type)

    # importance = blend of PageRank and frequency
    und = G.to_undirected()
    try:
        pr = nx.pagerank(und, weight="weight") if und.number_of_edges() else {n: 1 / max(len(G), 1) for n in G}
    except Exception:  # noqa: BLE001
        pr = {n: und.degree(n) for n in G}
    max_pr = max(pr.values(), default=1) or 1
    max_f = max((d["freq"] for _, d in G.nodes(data=True)), default=1) or 1
    for n in G:
        base_imp = 0.6 * pr[n] / max_pr + 0.4 * G.nodes[n]["freq"] / max_f
        if store.concepts[n].heading:  # section titles are the chapter's real topics
            base_imp += 0.3
        G.nodes[n]["importance"] = round(min(base_imp, 1.0), 4)
        G.nodes[n]["degree"] = und.degree(n)
    return G, P


def make_dag(P: nx.DiGraph) -> nx.DiGraph:
    """Break cycles by removing the weakest edge of each cycle (so topological order exists)."""
    D = P.copy()
    for _ in range(len(D) * 3 + 10):
        try:
            cyc = nx.find_cycle(D)
        except nx.NetworkXNoCycle:
            break
        u, v = min(((a, b) for a, b in cyc), key=lambda e: D[e[0]][e[1]].get("weight", 1))
        D.remove_edge(u, v)
    return D


def _pred(rel: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", rel.lower()).strip("_") or "related_to"


def build_meaning_representations(store: ConceptStore, triples: List[Triple], G: nx.DiGraph,
                                  P: nx.DiGraph, topics: Optional[TopicModelResult],
                                  topic_names: Optional[Dict[int, str]] = None) -> List[dict]:
    """One machine-readable JSON frame per concept (Meaning Representation, Unit IV)."""
    topic_names = topic_names or {}
    out = []
    for cid, c in store.concepts.items():
        t = topics.concept_topic.get(cid) if topics else None
        tid = t["topic"] if t else None
        tlabel = topic_names.get(tid) if tid is not None else None
        if tlabel is None and topics and tid is not None and tid < len(topics.topics):
            tlabel = topics.topics[tid]["label"]
        relations = []
        for tr in triples:
            if tr.subj == cid or tr.obj == cid:
                outgoing = tr.subj == cid
                other = tr.obj if outgoing else tr.subj
                relations.append({
                    "predicate": tr.rel, "type": tr.rel_type,
                    "direction": "out" if outgoing else "in",
                    "argument": {"id": other, "name": store.name(other)},
                    "logical_form": f"{_pred(tr.rel)}({tr.subj}, {tr.obj})",
                    "page": tr.page, "count": tr.count, "extraction": tr.source,
                    "evidence": tr.sentence,
                })
        pre = [{"id": p, "name": store.name(p), "relation": P[p][cid].get("relation")}
               for p in P.predecessors(cid)]
        deps = [{"id": d, "name": store.name(d)} for d in P.successors(cid)]
        related = []
        for nb in set(G.successors(cid)) | set(G.predecessors(cid)):
            d = G.get_edge_data(cid, nb) or G.get_edge_data(nb, cid)
            related.append({"id": nb, "name": store.name(nb),
                            "evidence": {"triple": "triple", "structure": "document outline",
                                         "section": "same section"}.get(d["kind"], "n-gram co-occurrence"),
                            "strength": d.get("weight", 1)})
        related.sort(key=lambda r: (r["evidence"] != "triple", -r["strength"]))
        out.append({
            "id": cid, "name": c.name, "aliases": c.aliases, "definition": c.definition,
            "definition_page": c.definition_page, "page": c.pages[0] if c.pages else None,
            "pages": c.pages, "frequency": c.freq,
            "topic": {"id": tid, "label": tlabel, "confidence": round(t["confidence"], 3) if t else None},
            "importance": G.nodes[cid].get("importance"),
            "prerequisites": pre, "dependents": deps, "related": related, "relations": relations,
        })
    return out


def graph_to_node_link(G: nx.DiGraph) -> dict:
    nodes = [{"id": n, **{k: v for k, v in d.items()}} for n, d in G.nodes(data=True)]
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({"source": u, "target": v, "kind": d["kind"], "relation": d["relation"],
                      "type": d["rel_type"], "weight": d.get("weight", 1)})
    return {"directed": True, "nodes": nodes, "edges": edges}


def graph_for_graphml(G: nx.DiGraph) -> nx.DiGraph:
    """Copy with only GraphML-safe attribute types."""
    H = nx.DiGraph()
    for n, d in G.nodes(data=True):
        H.add_node(n, label=d["label"], freq=int(d["freq"]), page=int(d["page"] or 0),
                   topic=int(d["topic"]) if d["topic"] is not None else -1,
                   importance=float(d.get("importance", 0)), definition=d.get("definition", ""))
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, kind=d["kind"], relation=d["relation"], rel_type=d["rel_type"],
                   weight=float(d.get("weight", 1)))
    return H

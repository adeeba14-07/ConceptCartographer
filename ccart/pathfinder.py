"""Learning-path finder: shortest weighted path between two concepts + study order."""
from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx

from .graph import make_dag


def _direction_graph(G: nx.DiGraph, P: nx.DiGraph, prefer_prereq: bool) -> nx.DiGraph:
    """Bidirectional cost graph; moving along a prerequisite link is cheaper."""
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes)
    for u, v, d in G.edges(data=True):
        base = d.get("cost", 1.0)
        for a, b in ((u, v), (v, u)):
            cost = base
            if prefer_prereq:
                if P.has_edge(a, b):
                    cost = base * 0.7
                elif P.has_edge(b, a):
                    cost = base * 1.15
            if not H.has_edge(a, b) or H[a][b]["cost"] > cost:
                H.add_edge(a, b, cost=cost)
    return H


def _describe_step(G: nx.DiGraph, a: str, b: str) -> Dict:
    if G.has_edge(a, b):
        d = G[a][b]
        forward = True
    else:
        d = G[b][a]
        forward = False
    if d["kind"] == "cooccur":
        text = "appears near (n-gram co-occurrence)"
        arrow = "···"
    elif d["kind"] == "section":
        text = "same section"
        arrow = "—section→" if forward else "←section—"
    else:
        text = d["relation"]
        arrow = f"—{text}→" if forward else f"←{text}—"
    return {"from": a, "to": b, "kind": d["kind"], "relation": text, "forward": forward,
            "arrow": arrow, "weight": d.get("weight", 1),
            "evidence": (d.get("relations") or [{}])[0].get("sentence", "")}


def find_learning_path(G: nx.DiGraph, P: nx.DiGraph, src: str, dst: str,
                       prefer_prereq: bool = True) -> Dict:
    if src not in G or dst not in G:
        return {"found": False, "reason": "Unknown concept."}
    if src == dst:
        return {"found": True, "path": [src], "steps": [], "cost": 0.0, "study_order": [src]}
    H = _direction_graph(G, P, prefer_prereq)
    try:
        path = nx.shortest_path(H, src, dst, weight="cost")
    except nx.NetworkXNoPath:
        comp = nx.node_connected_component(G.to_undirected(), src)
        return {"found": False, "reason": "These concepts are in different, unconnected parts of the graph "
                                          "(try raising the co-occurrence window or lowering the minimum "
                                          "co-occurrence count).", "component_size": len(comp)}
    cost = sum(H[a][b]["cost"] for a, b in zip(path, path[1:]))
    steps = [_describe_step(G, a, b) for a, b in zip(path, path[1:])]
    return {"found": True, "path": path, "steps": steps, "cost": round(cost, 2),
            "study_order": study_order(P, dst, extra=path)}


def study_order(P: nx.DiGraph, target: str, extra: Optional[List[str]] = None, max_depth: int = 3) -> List[str]:
    """Prerequisites of `target` (up to max_depth levels) in topological order, ending with the target."""
    D = make_dag(P)
    keep = {target}
    frontier = {target}
    for _ in range(max_depth):
        nxt = set()
        for n in frontier:
            nxt.update(D.predecessors(n))
        nxt -= keep
        keep |= nxt
        frontier = nxt
        if not frontier:
            break
    if extra:
        keep |= set(extra)
    sub = D.subgraph(keep)
    order = list(nx.lexicographical_topological_sort(sub, key=lambda n: n))
    if target in order:
        order.remove(target)
        order.append(target)
    return order

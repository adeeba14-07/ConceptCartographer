"""Interactive visualisation: PyVis (vis-network) HTML, with a hand-written fallback."""
from __future__ import annotations

import json
import textwrap
from typing import Dict, Iterable, List, Optional

import networkx as nx

from .config import PATH_COLOR, REL_COLORS
from .explain import tip_as_text
from .graph import topic_color

OPTIONS = {
    "physics": {
        "enabled": True,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {"gravitationalConstant": -70, "centralGravity": 0.012,
                             "springLength": 130, "springConstant": 0.07, "damping": 0.5},
        "stabilization": {"iterations": 250},
    },
    "interaction": {"hover": True, "navigationButtons": True, "keyboard": True,
                    "tooltipDelay": 80, "multiselect": True},
    "nodes": {"font": {"size": 15, "color": "#e5e7eb", "strokeWidth": 3, "strokeColor": "#0b1020"}},
    "edges": {"smooth": {"type": "dynamic"},
              "font": {"size": 10, "align": "middle", "color": "#cbd5e1", "strokeWidth": 0},
              "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}}},
}


def _tooltip(G: nx.DiGraph, n: str, topic_names: Dict[int, str]) -> str:
    d = G.nodes[n]
    tid = d.get("topic")
    tname = topic_names.get(tid, f"Topic {tid}") if tid is not None else "-"
    definition = textwrap.fill(d.get("definition") or "(no definition found)", 60)
    return (f"{d['label']}\nTopic: {tname}\nMentions: {d['freq']}   Page: {d.get('page') or '-'}\n"
            f"Connections: {d.get('degree', 0)}\n\n{definition}")


def build_vis_data(G: nx.DiGraph, topic_names: Optional[Dict[int, str]] = None,
                   highlight_path: Optional[List[str]] = None, show_cooc: bool = True,
                   show_edge_labels: bool = True, hide_isolated: bool = True,
                    nodes_filter: Optional[Iterable[str]] = None):
    topic_names = topic_names or {}
    nodes_allowed = set(nodes_filter) if nodes_filter is not None else set(G.nodes)
    path_nodes = set(highlight_path or [])
    path_edges = set()
    for a, b in zip(highlight_path or [], (highlight_path or [])[1:]):
        path_edges.add((a, b))
        path_edges.add((b, a))

    edges = []
    connected = set()
    for u, v, d in G.edges(data=True):
        if u not in nodes_allowed or v not in nodes_allowed:
            continue
        if d["kind"] in ("cooccur", "section") and not show_cooc and (u, v) not in path_edges:
            continue
        connected.update((u, v))
        on_path = (u, v) in path_edges
        if d["kind"] == "section":
            e = {"from": u, "to": v, "arrows": "to", "dashes": True, "width": 3.5 if on_path else 1.2,
                 "color": {"color": PATH_COLOR if on_path else "#a78bfa", "opacity": 1 if on_path else 0.6},
                 "title": f"“{G.nodes[v]['label']}” is explained in the section “{G.nodes[u]['label']}”"}
        elif d["kind"] == "cooccur":
            e = {"from": u, "to": v, "arrows": "", "dashes": True, "width": 3.5 if on_path else 1,
                 "color": {"color": PATH_COLOR if on_path else "#64748b", "opacity": 1 if on_path else 0.45},
                 "title": f"co-occurs within the n-gram window ({d.get('weight', 1)}x)"}
        else:
            col = REL_COLORS.get(d["rel_type"], "#94a3b8")
            ev = (d.get("relations") or [{}])[0].get("sentence", "")
            e = {"from": u, "to": v, "arrows": "to", "width": 4 if on_path else 1.2 + min(d.get("weight", 1), 4) * 0.5,
                 "color": {"color": PATH_COLOR if on_path else col, "opacity": 1},
                 "title": f"{G.nodes[u]['label']} -- {d['relation']} -> {G.nodes[v]['label']}\n\n"
                          + textwrap.fill(ev, 60)}
            if show_edge_labels:
                e["label"] = d["relation"][:28]
        edges.append(e)

    nodes = []
    for n, d in G.nodes(data=True):
        if n not in nodes_allowed:
            continue
        if hide_isolated and n not in connected and n not in path_nodes:
            continue
        on_path = n in path_nodes
        col = topic_color(d.get("topic"))
        nodes.append({
            "id": n, "label": d["label"], "title": _tooltip(G, n, topic_names),
            "size": 14 + 36 * float(d.get("importance", 0.0)),
            "color": {"background": col, "border": PATH_COLOR if on_path else "#0b1020",
                      "highlight": {"background": col, "border": "#ffffff"}},
            "borderWidth": 5 if on_path else 1.5, "shape": "dot",
        })
    return nodes, edges


def _pyvis_html(nodes, edges, height: int) -> str:
    from pyvis.network import Network

    net = Network(height=f"{height}px", width="100%", bgcolor="#0b1020", font_color="#e5e7eb",
                  directed=True, cdn_resources="in_line")
    for n in nodes:
        n = dict(n)
        nid = n.pop("id")
        net.add_node(nid, **n)
    for e in edges:
        e = dict(e)
        src, dst = e.pop("from"), e.pop("to")
        net.add_edge(src, dst, **e)
    net.set_options(json.dumps(OPTIONS, separators=(",", ":")))
    html = net.generate_html()
    if "vis-network" not in html and "vis.Network" not in html and "new vis" not in html:
        raise RuntimeError("PyVis produced no network script")
    return html


_FALLBACK = """<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>html,body{margin:0;background:#0b1020;font-family:sans-serif}#net{width:100%;height:%HEIGHT%px}</style>
</head><body><div id="net"></div><script>
var nodes=new vis.DataSet(%NODES%);var edges=new vis.DataSet(%EDGES%);
var network=new vis.Network(document.getElementById('net'),{nodes:nodes,edges:edges},%OPTIONS%);
</script></body></html>"""


_TIP_JS = """<style>
div.vis-tooltip{max-width:440px;white-space:normal!important;font-family:Segoe UI,Arial,sans-serif;font-size:13px;
line-height:1.45;color:#e5e7eb;background:#111827;border:1px solid #7c3aed;border-radius:10px;padding:10px 13px;
box-shadow:0 8px 28px rgba(0,0,0,.55)}
div.vis-tooltip .t{font-size:15px;font-weight:700;margin-bottom:2px}
div.vis-tooltip .tp{display:inline-block;font-size:11px;padding:1px 8px;border-radius:999px;background:#4c1d95;margin-left:6px}
div.vis-tooltip .src{font-size:11px;font-style:italic;opacity:.75;margin:2px 0 6px}
div.vis-tooltip .ai{color:#fbbf24}div.vis-tooltip .h{font-weight:700;margin-top:6px;color:#c4b5fd}
div.vis-tooltip ul{margin:2px 0 0 16px;padding:0}
</style>
<script>
(function(){
var tips=%TIPS%;
function esc(s){return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
function build(t){
 var h='<div class="t">'+esc(t.name)+(t.topic?'<span class="tp">'+esc(t.topic)+'</span>':'')+'</div>';
 h+='<div>'+esc(t.definition)+'</div><div class="src '+(t.src_kind==='ai'?'ai':'')+'">'+esc(t.source)+(t.page?' · page '+t.page:'')+'</div>';
 if(t.how){h+='<div class="h">How it works</div><div>'+esc(t.how)+(t.how_kind==='ai'?' <span class="ai">(AI)</span>':'')+'</div>';}
 if(t.connects&&t.connects.length){h+='<div class="h">How it connects</div><ul>';
  t.connects.slice(0,6).forEach(function(c){h+='<li><b>'+esc(c[0])+':</b> '+esc(c[1].join(', '))+'</li>';});h+='</ul>';}
 var d=document.createElement('div');d.innerHTML=h;return d;}
function apply(){
 try{ if(typeof nodes==='undefined'||!nodes||!nodes.update) return false;
  nodes.update(nodes.getIds().map(function(id){var t=tips[id];return t?{id:id,title:build(t)}:{id:id};}));
  return true;}catch(e){return false;}}
if(!apply()){window.addEventListener('load',function(){setTimeout(apply,50);});}
})();
</script>"""


def _inject_tips(html: str, tips: Dict[str, dict]) -> str:
    payload = json.dumps(tips, ensure_ascii=False).replace("</", "<\\/")
    block = _TIP_JS.replace("%TIPS%", payload)
    return html.replace("</body>", block + "</body>", 1) if "</body>" in html else html + block


def render_graph_html(G: nx.DiGraph, height: int = 650, tips: Optional[Dict[str, dict]] = None, **kwargs) -> str:
    nodes, edges = build_vis_data(G, **kwargs)
    if tips:  # plain-text tooltip first (always works); rich HTML tooltip is added by script below
        for n in nodes:
            if n["id"] in tips:
                n["title"] = tip_as_text(tips[n["id"]])
    if not nodes:
        return ("<div style='color:#e5e7eb;font-family:sans-serif;padding:24px;background:#0b1020'>"
                "No concepts to display with the current filters.</div>")
    try:
        html = _pyvis_html(nodes, edges, height)
    except Exception:  # noqa: BLE001 - PyVis missing/incompatible -> plain vis-network page
        html = (_FALLBACK.replace("%HEIGHT%", str(height)).replace("%NODES%", json.dumps(nodes))
                .replace("%EDGES%", json.dumps(edges)).replace("%OPTIONS%", json.dumps(OPTIONS)))
    if tips:
        shown = {n["id"] for n in nodes}
        html = _inject_tips(html, {k: v for k, v in tips.items() if k in shown})
    return html

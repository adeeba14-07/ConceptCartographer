"""Headless smoke test: python tests/smoke_test.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccart.pipeline import Params, run_pipeline
from ccart.preprocess import pages_from_text
from ccart.sample_text import SAMPLE_CHAPTER
from ccart.pathfinder import find_learning_path
from ccart.visualize import render_graph_html

res = run_pipeline(pages_from_text(SAMPLE_CHAPTER), Params(extra_terms=["bag of words"], exclude_terms=["word"]))
print("PARSER:", res.parser)
print("STATS:", res.stats)
print("WARN:", res.warnings)
print("\nCONCEPTS:")
for c in sorted(res.store.concepts.values(), key=lambda c: -c.freq):
    print(f"  {c.name:38s} freq={c.freq:2d} pages={c.pages} aliases={c.aliases}")
print("\nTRIPLES:")
for t in res.triples:
    print(f"  ({res.store.name(t.subj)}) -[{t.rel}|{t.rel_type}]-> ({res.store.name(t.obj)}) x{t.count} [{t.source}]")
print("\nTOPICS (k=%d, engine=%s, primary=%s):" % (res.topics.num_topics, res.topics.engine, res.topics.primary_metric))
for t in res.topics.topics:
    print("  ", t["id"], t["label"], "|", ", ".join(w for w, _ in t["words"][:6]))
print("  sweep:", res.topics.sweep)
print("\nLM:", res.lm_report["best"])
print("\nPATH:")
ids = list(res.store.concepts)
def cid(name):
    for k, c in res.store.concepts.items():
        if c.name.lower() == name.lower(): return k
a, b = cid("Tokenization"), cid("Latent Dirichlet Allocation")
print(a, b)
if a and b:
    p = find_learning_path(res.G, res.P, a, b)
    print(p["found"], p.get("path"), p.get("study_order"))
html = render_graph_html(res.G, topic_names={})
print("HTML bytes:", len(html))
assert len(res.store.concepts) >= 10 and len(res.triples) >= 10
print("OK")

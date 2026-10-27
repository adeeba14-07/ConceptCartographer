"""Quality checks on several different documents (no NLP-specific wording) + the AI-definition rule.

Run: python tests/test_documents.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccart.explain import build_tips, fill_with_ai
from ccart.pipeline import Params, run_pipeline
from ccart.preprocess import pages_from_text, key_of
from ccart.visualize import render_graph_html

BIOLOGY = """1. Cell Biology

The cell is the basic unit of life. A cell is a small structure that carries out the functions of a living organism. Every cell is surrounded by a cell membrane. The cell membrane controls what enters and leaves the cell. Inside the cell, the cytoplasm holds the organelles.

1.1 Mitochondria

Mitochondria are organelles that produce energy for the cell. The mitochondria convert glucose into ATP. ATP is the energy currency of the cell. Mitochondria have their own DNA. A cell that needs much energy contains many mitochondria.

1.2 Nucleus

The nucleus stores the genetic material of the cell. The nucleus contains DNA. DNA carries the instructions for building proteins. The nucleus is surrounded by the nuclear envelope. Ribosomes build proteins using instructions copied from DNA.

2. Photosynthesis

Photosynthesis is the process by which plants make glucose from sunlight. Photosynthesis takes place in the chloroplasts. Chloroplasts contain chlorophyll. Chlorophyll absorbs sunlight. Plants use glucose for energy and growth. Photosynthesis releases oxygen.

2.1 Light Reactions

The light reactions capture sunlight in the chloroplast membranes. The light reactions produce ATP. Chlorophyll drives the light reactions. Water is split during the light reactions.

2.2 Calvin Cycle

The Calvin cycle builds glucose from carbon dioxide. The Calvin cycle uses ATP from the light reactions. The Calvin cycle takes place in the stroma of the chloroplast.
"""

HISTORY = """1. The Industrial Revolution

The Industrial Revolution was a period of rapid change in how goods were made. The Industrial Revolution began in Britain in the eighteenth century. Factories replaced hand production in homes. The steam engine powered many factories. Cities grew quickly because workers moved to the factories.

1.1 The Steam Engine

The steam engine converts heat into movement. James Watt improved the steam engine. The steam engine pumped water out of mines. The steam engine later powered trains and ships. Coal fuelled the steam engine.

1.2 Factories and Workers

A factory is a building where many workers produce goods with machines. Factory owners paid workers low wages. Workers faced long hours and dangerous conditions. Trade unions formed to protect workers. Child labour was common in factories until reform laws were passed.

2. Railways

Railways connected cities and carried goods quickly. The railway used the steam engine for power. Railways lowered the cost of transport. Railways helped factories sell goods across the country. Railway companies employed thousands of workers.

2.1 Reform Laws

Reform laws limited working hours in factories. The Factory Act reduced child labour. Reform laws followed protests by trade unions. Governments appointed inspectors to check factories.
"""

SOFTWARE = """1. Databases

A database stores data in an organised way so that programs can find it quickly. A database is managed by a database management system. The database management system controls access to the data.

1.1 Relational Databases

A relational database stores data in tables. Each table has rows and columns. A primary key identifies each row of a table. A foreign key links one table to another table. SQL is the language used to query a relational database.

1.2 Indexes

An index speeds up queries on a table. An index works like the index of a book. A database builds an index on a column. Too many indexes slow down writes to the table.

2. Transactions

A transaction is a group of database operations that succeed or fail together. A transaction keeps the database consistent. A transaction follows the ACID rules. Atomicity means that all steps of a transaction happen or none happen. Isolation means that transactions do not disturb each other.

2.1 Locking

Locking prevents two transactions from changing the same row at once. A lock protects a row until the transaction ends. Locking supports isolation. Deadlock happens when two transactions wait for each other forever.
"""

NO_HEADINGS = """Inflation is a general rise in prices over time. Inflation reduces the buying power of money. Central banks try to control inflation with interest rates. A central bank raises the interest rate to slow inflation. A higher interest rate makes loans more expensive. Expensive loans reduce spending by households and firms. Lower spending slows the rise in prices.

Unemployment measures the share of people without a job who are looking for work. High unemployment means that firms produce less. Governments use public spending to reduce unemployment. Public spending can raise inflation when the economy is already strong. Central banks and governments must balance inflation and unemployment.

A recession is a long fall in economic activity. A recession raises unemployment. A recession lowers inflation. Central banks cut the interest rate during a recession to encourage loans and spending.
"""

JUNK = {"the","a","an","it","this","that","one","two","three","five","example","figure","thing","way","time"}


def check(name, text, expect_outline=True):
    res = run_pipeline(pages_from_text(text), Params(max_concepts=0, min_freq=2))
    store, G = res.store, res.G
    names = [c.name for c in store.concepts.values()]
    deg = dict(G.to_undirected().degree())
    connected = sum(1 for n in G if deg[n] > 0) / max(len(G), 1)
    srcs = {}
    for e in res.explain.values():
        srcs[e["def_source"]] = srcs.get(e["def_source"], 0) + 1
    print(f"\n=== {name}: {len(names)} concepts | outline edges {res.stats['outline_edges']} | "
          f"connected {connected:.0%} | definition sources {srcs}")
    print("   concepts:", ", ".join(sorted(names)))
    assert len(names) >= 6, "too few concepts"
    assert connected >= 0.7, f"too many isolated concepts ({connected:.0%})"
    bad = [n for n in names if n.lower() in JUNK or n.isdigit()]
    assert not bad, f"junk concepts: {bad}"
    if expect_outline:
        assert res.stats["outline_edges"] >= 2, "section structure not found"
    # every concept has a usable explanation and every one can be found by search
    for cid, e in res.explain.items():
        assert len(e["definition"]) >= 15, (cid, e["definition"])
        assert key_of(e["name"]), cid
    # tooltips reach the page for every drawn concept
    tips = build_tips(G, res.explain, {})
    html = render_graph_html(G, height=500, tips=tips, hide_isolated=False)
    assert "How it works" in html and "How it connects" in html
    return res


def test_ai_rule(res):
    """AI may only touch concepts the document does not define."""
    before = {cid: dict(e) for cid, e in res.explain.items()}
    asked = []

    def fake_ai(prompt):
        ids = [l.split("|")[0].replace("- id:", "").strip() for l in prompt.splitlines() if l.startswith("- id:")]
        asked.extend(ids)
        return json.dumps({i: {"definition": f"Simple words for {i}.", "how": f"It works like {i}."} for i in ids})

    rep = fill_with_ai(res.explain, res.store, res.sentences, res.G, fake_ai)
    changed = [cid for cid, e in res.explain.items() if e["def_source"] == "ai"]
    in_doc = [cid for cid, e in before.items() if e["def_source"] in ("document", "document_section")]
    assert set(asked) == {cid for cid, e in before.items() if e["def_source"] == "auto"}, "AI was asked about documented concepts"
    assert all(res.explain[c]["definition"] == before[c]["definition"] for c in in_doc), "document definition overwritten"
    assert len(changed) == len(asked) == rep["written"]
    print(f"   AI rule OK: asked about {len(asked)} undefined concepts, left {len(in_doc)} documented ones untouched")

    # a failing AI must never break anything (use a fresh copy: nothing is left undefined after the run above)
    import copy
    fresh = {cid: dict(e) for cid, e in before.items()}
    snapshot = copy.deepcopy(fresh)
    def broken(prompt): raise RuntimeError("no internet")
    rep2 = fill_with_ai(fresh, res.store, res.sentences, res.G, broken)
    assert rep2["errors"] and fresh == snapshot
    print("   AI failure handled: offline definitions kept")


if __name__ == "__main__":
    r = None
    for nm, tx, outline in [("Biology", BIOLOGY, True), ("History", HISTORY, True),
                            ("Software", SOFTWARE, True), ("No headings (economics)", NO_HEADINGS, False)]:
        r = check(nm, tx, outline)
        if nm == "Biology":
            test_ai_rule(r)
    print("\nALL DOCUMENT CHECKS PASSED")

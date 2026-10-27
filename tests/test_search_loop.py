"""Regression: searching a phrase must never trigger an endless rebuild loop (python tests/test_search_loop.py)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "run_app_with_stub.py")).read().split("APP = sys.argv")[0])
import ccart.sample_text as stx, ccart.pipeline as pl

stx.SAMPLE_CHAPTER = """1. Morphology: Concept and Types

Morphology studies the structure of a word. A word is made of a root, prefixes and suffixes. Each word has a root. Every word can have affixes. Morphology has several types.

1.1 Inflectional Morphology

Inflectional morphology changes the form of a word without changing its meaning. Plural endings are inflectional. The root stays the same.

2. Stemming

Stemming removes prefixes or suffixes from a word using simple rules. A stemmer is fast. Stemming does not always give a real word. A stemmer cuts the word to a stem. Stemming; lemmatization are both used often.
"""
orig, n = pl.run_pipeline, {"builds": 0}
def counting(*a, **k):
    n["builds"] += 1
    return orig(*a, **k)
pl.run_pipeline = counting
code = open(os.path.join(ROOT, "app.py")).read()

def run_once(press, find):
    RUN_BUTTON["value"], FIND["v"] = press, find
    try:
        exec(compile(code, "app.py", "exec"), {"__name__": "__main__"}); return "finished"
    except StopApp:
        return "rerun"

run_once(True, "")
# 1) heading with a colon inside is found directly (no rebuild needed)
run_once(False, "Morphology: Concept and Types")
assert n["builds"] == 1, n
# 2) a phrase that can never become a concept: at most ONE automatic attempt, then it stops
for _ in range(6):
    if run_once(False, "stemming lemmatization") == "finished":
        break
assert n["builds"] <= 2, f"rebuild loop! {n}"
print("search loop regression OK, builds:", n["builds"])

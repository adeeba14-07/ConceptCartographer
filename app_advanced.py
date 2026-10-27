"""ConceptCartographer - turn any textbook into an interactive concept map.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import networkx as nx
import pandas as pd
import streamlit as st

from ccart.concepts import annotate_mentions
from ccart.config import PATH_COLOR, REL_COLORS, REL_LABELS, STOPWORDS
from ccart.graph import (graph_for_graphml, graph_to_node_link, topic_color,
                         build_meaning_representations)
from ccart.pathfinder import find_learning_path
from ccart.pipeline import Params, PipelineResult, run_pipeline
from ccart.preprocess import (Sentence, extract_pdf_pages, is_word, pages_from_text, simple_lemma,
                              tokenize)
from ccart.sample_text import SAMPLE_CHAPTER
from ccart.topics import gensim_available
from ccart.triples import dependency_triples, pattern_triples
from ccart.visualize import render_graph_html

st.set_page_config(page_title="ConceptCartographer", page_icon="🗺️", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1500px;}
.cc-hero {background: linear-gradient(120deg,#1e3a8a 0%,#6d28d9 55%,#be185d 100%); padding: 22px 28px;
          border-radius: 16px; color: #fff; margin-bottom: 14px;}
.cc-hero h1 {margin: 0; font-size: 2.0rem; color: #fff;}
.cc-hero p {margin: 4px 0 0 0; opacity: .92; font-size: 1.02rem;}
.cc-chip {display:inline-block; padding: 2px 10px; margin: 2px 4px 2px 0; border-radius: 999px;
          font-size: .82rem; color:#0b1020; font-weight:600;}
.cc-step {border-left: 4px solid #6d28d9; padding: 6px 14px; margin: 6px 0; background: rgba(109,40,217,.08);
          border-radius: 0 10px 10px 0;}
.cc-small {font-size:.85rem; opacity:.8;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the spaCy language model (first run may download it)...")
def load_nlp():
    try:
        import spacy
    except Exception:  # noqa: BLE001
        return None, "spaCy is not installed (pip install spacy)."
    import importlib
    for attempt in range(2):
        try:
            nlp = spacy.load("en_core_web_sm")
            nlp.max_length = 3_000_000
            return nlp, "ok"
        except OSError:
            if attempt == 0:
                try:
                    spacy.cli.download("en_core_web_sm")
                    importlib.invalidate_caches()
                except BaseException:  # noqa: BLE001 - download() may call sys.exit
                    break
    return None, "Model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm"


@st.cache_data(show_spinner="Reading the PDF...")
def read_pdf(digest: str, data: bytes):
    return extract_pdf_pages(data)


def show_html(html: str, height: int, key: str = ""):
    try:
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)
    except Exception:  # noqa: BLE001
        st.warning("Could not embed the interactive view in this Streamlit version. Download it from the Export tab.")


def render_cached(res_obj, height: int, **kw) -> str:
    """render_graph_html with a small per-session cache (Streamlit reruns the whole script on every click)."""
    cache = st.session_state.setdefault("_render_cache", {})
    key = (id(res_obj), height, json.dumps(kw, sort_keys=True, default=str))
    if key not in cache:
        if len(cache) > 12:
            cache.clear()
        cache[key] = render_graph_html(res_obj.G, height=height, **kw)
    return cache[key]


def howto(title: str, body: str, open_: bool = True):
    with st.expander(f"ℹ️ {title}", expanded=open_):
        st.markdown(body)


def chip(text: str, color: str) -> str:
    return f"<span class='cc-chip' style='background:{color}'>{text}</span>"


def result() -> PipelineResult | None:
    return st.session_state.get("result")


def topic_names() -> dict:
    return st.session_state.get("topic_names", {})


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='cc-hero'><h1>ConceptCartographer</h1>"
    "<p>Turn any textbook into an interactive concept map - semantic parsing + Bayesian topic modelling "
    "+ n-gram analysis, with a learning path between any two concepts.</p></div>",
    unsafe_allow_html=True,
)

nlp, nlp_status = load_nlp()

# ---------------------------------------------------------------------------
# Sidebar: input + parameters
# ---------------------------------------------------------------------------
pages: list = []
first_page = 1
with st.sidebar:
    st.header("1 · Input")
    source = st.radio("Text source", ["Upload PDF / TXT", "Paste text", "Built-in sample chapter"], index=2)
    if source == "Upload PDF / TXT":
        up = st.file_uploader("Textbook chapter (PDF or TXT)", type=["pdf", "txt"])
        if up is not None:
            data = up.getvalue()
            if up.name.lower().endswith(".pdf"):
                all_pages = read_pdf(hashlib.md5(data).hexdigest(), data)
            else:
                all_pages = pages_from_text(data.decode("utf-8", errors="ignore"))
            if not any(p.strip() for p in all_pages):
                st.error("No text could be extracted (is this a scanned PDF? run OCR first).")
            else:
                n = len(all_pages)
                st.caption(f"{n} page(s) detected.")
                c1, c2 = st.columns(2)
                p_from = c1.number_input("From page", 1, n, 1)
                p_to = c2.number_input("To page", 1, n, min(n, 30))
                if p_to < p_from:
                    st.warning("'To page' must be >= 'From page'.")
                else:
                    pages = all_pages[int(p_from) - 1:int(p_to)]
                    first_page = int(p_from)
    elif source == "Paste text":
        txt = st.text_area("Paste a chapter", height=220, placeholder="Paste textbook text here...")
        if txt.strip():
            pages = pages_from_text(txt)
    else:
        pages = pages_from_text(SAMPLE_CHAPTER)
        st.caption("Short NLP chapter (6 pages) so you can try everything instantly.")

    st.header("2 · Settings")
    with st.expander("Concepts & relations", expanded=True):
        max_concepts = st.slider("Max concepts", 10, 150, 60, 5)
        min_freq = st.slider("Minimum concept frequency", 1, 6, 2)
        strict = st.checkbox("Strict relation vocabulary (higher precision)", value=False,
                             help="Keep only relations whose verb is in the built-in relation lexicon.")
        extra = st.text_area("Always include these concepts (one per line)",
                             value="bag of words" if source == "Built-in sample chapter" else "", height=70)
        excl = st.text_area("Exclude these terms (one per line)", value="", height=60)
    with st.expander("N-gram co-occurrence (weak edges)"):
        window = st.slider("Window size N (tokens)", 4, 40, 12)
        cmin = st.slider("Min co-occurrences", 1, 6, 2)
        cmax = st.slider("Max weak edges per concept", 0, 8, 3)
    with st.expander("LDA topic model"):
        auto_k = st.checkbox("Choose number of topics by coherence", value=True)
        k_range = st.slider("Topic search range", 2, 12, (3, 7)) if auto_k else (3, 7)
        k_fixed = st.slider("Number of topics", 2, 12, 4) if not auto_k else 4
        lda_win = st.slider("Sentences per LDA document", 2, 8, 3)
    run = st.button("🚀 Build knowledge graph", type="primary", disabled=not pages)

    st.divider()
    st.caption("**System status**")
    st.caption(f"{'✅' if nlp is not None else '⚠️'} spaCy: {'en_core_web_sm loaded' if nlp is not None else nlp_status}")
    st.caption(f"{'✅' if gensim_available() else '⚠️'} Gensim: {'available' if gensim_available() else 'not installed - using scikit-learn LDA'}")

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run and pages:
    params = Params(
        max_concepts=max_concepts, min_freq=min_freq, strict_relations=strict, cooc_window=window,
        cooc_min_count=cmin, cooc_max_per_node=cmax, auto_topics=auto_k, num_topics=k_fixed,
        k_min=k_range[0], k_max=k_range[1], lda_window=lda_win, first_page=first_page,
        extra_terms=[l.strip() for l in extra.splitlines() if l.strip()],
        exclude_terms=[l.strip() for l in excl.splitlines() if l.strip()],
    )
    bar = st.progress(0.0, text="Starting...")

    def cb(frac: float, msg: str):
        bar.progress(min(max(frac, 0.0), 1.0), text=msg)

    try:
        res = run_pipeline(pages, params, nlp, cb)
    except ValueError as exc:
        bar.empty()
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        bar.empty()
        st.error(f"Unexpected error while building the graph: {exc}")
    else:
        bar.empty()
        for k in [k for k in st.session_state if k.startswith(("tn_", "lp_", "g_", "t_", "c_", "n_", "lm_"))]:
            del st.session_state[k]
        st.session_state["result"] = res
        st.session_state["topic_names"] = {t["id"]: t["label"] for t in res.topics.topics}

res = result()
if res is not None:
    for _t in res.topics.topics:  # keep graph colours/legend in sync with renamed topics
        _k = f"tn_{_t['id']}"
        if _k in st.session_state:
            st.session_state["topic_names"][_t["id"]] = st.session_state[_k]

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
if res is None:
    st.info("Choose a text source in the sidebar and press **Build knowledge graph**. "
            "The built-in sample chapter lets you explore every feature immediately.")
    st.markdown("### How it works")
    st.markdown(
        """
| Stage | NLP concept (syllabus) | What happens |
|---|---|---|
| Text extraction | Text normalisation | `pdfplumber` reads pages; headers/footers and hyphenation are cleaned |
| Sentence splitting & tokenization | Tokenization, lemmatization | Abbreviation-aware splitter, regex tokenizer, rule lemmatizer |
| Semantic parsing | **Predicate-Argument Structure** (Unit IV) | spaCy dependency parse -> `(subject) -predicate-> (object)` triples |
| Concept frames | **Meaning Representation** (Unit IV) | JSON per concept: definition, page, prerequisites, related, logical forms |
| Topic modelling | **LDA** (Unit V) | Gensim LDA groups concepts into topics -> node colours |
| Co-occurrence | **N-gram models** (Unit V) | Sliding N-token window adds weak edges between nearby concepts |
| Validation | **LM evaluation** (Unit V) | Coherence (c_v / NPMI / UMass) picks & validates topics; perplexity for n-gram LMs |
| Graph & path | Graph algorithms | NetworkX graph, PyVis view, shortest learning path |
"""
    )
    st.stop()

G, P, store = res.G, res.P, res.store
names = {cid: c.name for cid, c in store.concepts.items()}

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
m = st.columns(6)
m[0].metric("Concepts", res.stats["concepts"])
m[1].metric("Relations (triples)", res.stats["triple_edges"])
m[2].metric("Weak n-gram edges", res.stats["weak_edges"])
m[3].metric("Topics (LDA)", res.stats["topics"])
m[4].metric("Sentences", res.stats["sentences"])
m[5].metric("Components", res.stats["components"])
for w in res.warnings:
    st.warning(w)

st.info("**Quick tour (3 steps):** ① Open **Knowledge Graph** and hover over a bubble to read its definition. "
        "② Open **Learning Path**, choose two concepts and read the chain of steps. "
        "③ Open **Topics (LDA)** to see which concepts were grouped together. "
        "Every tab has an ℹ️ box explaining what you are looking at.")

tabs = st.tabs(["🕸️ Knowledge Graph", "🧭 Learning Path", "🧩 Concepts (JSON)", "🔗 Triples",
                "🧠 Topics (LDA)", "📊 Language Models", "🔬 NLP Pipeline", "⬇️ Export"])

# ---------------------------------------------------------------------------
# Tab 1: Knowledge graph
# ---------------------------------------------------------------------------
with tabs[0]:
    howto("What is this?", """Each **bubble is a concept** found in the text (for example "Perplexity"). Each **arrow is a relation** found in a sentence, such as *N-Gram Model → uses → Tokenization*. **Dashed grey lines** are weak links: two concepts that simply appear close together in the text.

- **Bubble colour** = the topic the concept belongs to (the legend above the graph shows the colours).
- **Bubble size** = how important the concept is (many mentions and many connections).
- **Hover** a bubble to see its definition and page. **Drag** bubbles around, **scroll** to zoom.
- If the graph is crowded, untick *Show weak edges* or pick fewer topics in *Topics to show*.""", True)
    c1, c2, c3, c4 = st.columns(4)
    hide_iso = c1.checkbox("Hide isolated concepts", True, key="g_iso")
    show_weak = c2.checkbox("Show weak (n-gram) edges", True, key="g_weak")
    show_lab = c3.checkbox("Show relation labels", True, key="g_lab")
    height = c4.slider("Height", 400, 950, 650, 50, key="g_h")
    tids = [t["id"] for t in res.topics.topics]
    sel = st.multiselect("Topics to show", tids, default=tids, key="g_topics",
                         format_func=lambda t: f"{t}: {topic_names().get(t, t)}")
    sel = sel or tids  # an empty selection means "show every topic"
    allowed = [n for n, d in G.nodes(data=True) if d.get("topic") in sel or d.get("topic") is None]
    legend = "".join(chip(topic_names().get(t, f"Topic {t}"), topic_color(t)) for t in tids)
    rel_legend = "".join(chip(REL_LABELS[r], REL_COLORS[r]) for r in
                         ["is_a", "requires", "used_for", "has_part", "evaluates", "improves"])
    st.markdown(f"**Node colour = LDA topic** &nbsp; {legend}", unsafe_allow_html=True)
    st.markdown(f"**Edge colour = relation** &nbsp; {rel_legend} {chip('co-occurrence (dashed)', '#64748b')}",
                unsafe_allow_html=True)
    html = render_cached(res, height, topic_names=topic_names(), show_cooc=show_weak,
                         show_edge_labels=show_lab, hide_isolated=hide_iso, nodes_filter=sorted(allowed))
    show_html(html, height + 10)
    st.caption("Drag nodes, scroll to zoom, hover for definitions. Node size = importance (PageRank + frequency).")

    st.subheader("Most important concepts")
    top = sorted(G.nodes(data=True), key=lambda x: -x[1].get("importance", 0))[:12]
    st.dataframe(pd.DataFrame([{"Concept": d["label"], "Topic": topic_names().get(d["topic"], "-"),
                                "Importance": d.get("importance"), "Mentions": d["freq"],
                                "Connections": d.get("degree", 0), "Page": d.get("page")} for _, d in top]),
                 hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2: Learning path
# ---------------------------------------------------------------------------
with tabs[1]:
    howto("What is this?", """Choose two concepts: **From** = something you already know, **To** = something you want to learn. The app finds the shortest chain of relations connecting them and shows it as numbered steps (with the sentence from the text as proof).

Below the steps is a **study order**: the prerequisites to read first, ending with your target concept. The red line in the graph is the path.""", True)
    st.markdown("Pick any two concepts. The path is the cheapest chain of relations between them "
                "(strong triple edges cost less than weak n-gram edges).")
    ordered = sorted(store.concepts, key=lambda c: (store.concepts[c].pages[:1] or [999], -store.concepts[c].freq))
    c1, c2, c3 = st.columns([2, 2, 1.4])
    src = c1.selectbox("From (what I know)", ordered, index=0, format_func=lambda c: names[c], key="lp_src")
    dst = c2.selectbox("To (what I want to learn)", ordered, index=min(len(ordered) - 1, 5),
                       format_func=lambda c: names[c], key="lp_dst")
    prefer = c3.checkbox("Prefer prerequisite direction", True, key="lp_pref")
    if src == dst:
        st.info("Choose two different concepts.")
    else:
        out = find_learning_path(G, P, src, dst, prefer)
        if not out["found"]:
            st.error(out["reason"])
        else:
            st.success(f"Path with {len(out['path']) - 1} step(s) - total cost {out['cost']}")
            for i, step in enumerate(out["steps"], 1):
                st.markdown(
                    f"<div class='cc-step'><b>{i}.</b> {names[step['from']]} &nbsp;"
                    f"<code>{step['arrow']}</code>&nbsp; {names[step['to']]}"
                    + (f"<div class='cc-small'>“{step['evidence']}”</div>" if step["evidence"] else "")
                    + "</div>", unsafe_allow_html=True)
            st.markdown("#### Suggested study order (prerequisites first)")
            for i, cid in enumerate(out["study_order"], 1):
                c = store.concepts[cid]
                with st.expander(f"{i}. {c.name}" + (f"  · page {c.pages[0]}" if c.pages else "")):
                    st.write(c.definition or "No definition found.")
            h = render_cached(res, 520, topic_names=topic_names(), highlight_path=out["path"],
                              hide_isolated=True)
            show_html(h, 530)
            st.caption(f"Path highlighted in red ({PATH_COLOR}).")

# ---------------------------------------------------------------------------
# Tab 3: Concept frames
# ---------------------------------------------------------------------------
with tabs[2]:
    howto("What is this?", """Every concept is stored as a **JSON frame** (the "meaning representation"): its definition, page number, topic, prerequisites (what to learn first), what it leads to, related concepts, and its **predicate-argument forms** such as `uses(n_gram_model, tokenization)`.

Search in the table, then pick a concept below it to see its full frame on the right.""", True)
    frames = {f["id"]: f for f in res.frames}
    q = st.text_input("Search concepts", "", key="c_q")
    rows = [{"Concept": f["name"], "Topic": topic_names().get(f["topic"]["id"], "-"), "Mentions": f["frequency"],
             "Page": f["page"], "Prerequisites": ", ".join(p["name"] for p in f["prerequisites"]),
             "Definition": f["definition"]} for f in res.frames
            if q.lower() in f["name"].lower() or q.lower() in f["definition"].lower()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    pick = st.selectbox("Inspect a concept frame (meaning representation)", list(frames),
                        format_func=lambda c: names[c], key="c_pick")
    f = frames[pick]
    f = {**f, "topic": {**f["topic"], "label": topic_names().get(f["topic"]["id"], f["topic"]["label"])}}
    a, b = st.columns([1, 1])
    with a:
        st.markdown(f"### {f['name']}")
        st.write(f["definition"] or "_No definition found._")
        st.markdown("**Prerequisites:** " + (", ".join(p["name"] for p in f["prerequisites"]) or "-"))
        st.markdown("**Leads to:** " + (", ".join(d["name"] for d in f["dependents"]) or "-"))
        st.markdown("**Related:** " + (", ".join(r["name"] for r in f["related"][:10]) or "-"))
        st.markdown("**Predicate-argument forms:**")
        for r in f["relations"][:12]:
            st.code(r["logical_form"], language="text")
    with b:
        st.json(f)

# ---------------------------------------------------------------------------
# Tab 4: Triples
# ---------------------------------------------------------------------------
with tabs[3]:
    howto("What is this?", """A **triple** is one fact taken from a sentence: **subject – predicate – object**, for example *Perplexity – evaluates – Language Model*. This is the *predicate-argument structure* step. The **Extractor** column says whether the spaCy dependency parse or the pattern matcher found it, and **Sentence** shows the original text, so you can check each fact.""", True)
    st.markdown("Each row is a **predicate-argument structure**: `predicate(subject, object)`.")
    types = sorted({t.rel_type for t in res.triples})
    sel_t = st.multiselect("Relation types", types, default=types, key="t_types",
                           format_func=lambda t: REL_LABELS.get(t, t))
    sel_t = sel_t or types
    q2 = st.text_input("Filter by concept or relation text", "", key="t_q").lower()
    rows = []
    for t in res.triples:
        if t.rel_type not in sel_t:
            continue
        if q2 and q2 not in (names[t.subj] + names[t.obj] + t.rel).lower():
            continue
        rows.append({"Subject": names[t.subj], "Predicate": t.rel, "Object": names[t.obj],
                     "Type": REL_LABELS.get(t.rel_type, t.rel_type), "Count": t.count,
                     "Page": ", ".join(str(p) for p in sorted(t.pages)), "Extractor": t.source,
                     "Logical form": f"{t.rel.replace(' ', '_')}({t.subj}, {t.obj})", "Sentence": t.sentence})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    cnt = Counter(REL_LABELS.get(t.rel_type, t.rel_type) for t in res.triples)
    if cnt:
        st.bar_chart(pd.Series(cnt, name="triples"))

# ---------------------------------------------------------------------------
# Tab 5: Topics
# ---------------------------------------------------------------------------
with tabs[4]:
    howto("What is this?", """**LDA** splits the text into hidden **topics** (groups of words that often appear together). Each concept is assigned to its best topic, which gives the bubble colours in the graph.

- **Coherence score** says how sensible the topics are (higher = better). The chart shows the score for each number of topics, and the app picks the best one.
- Topic names are made from the top concepts. **Rename a topic** in the text box (for example "Smoothing and evaluation") and the graph legend updates.
- Topics look mixed? In the sidebar untick *Choose number of topics by coherence* and set the number yourself (try 4 to 6 for a 30-50 page chapter), then rebuild.""", True)
    tm = res.topics
    c = st.columns(4)
    c[0].metric("Engine", tm.engine.split(" ")[0])
    c[1].metric("Topics (k)", tm.num_topics)
    c[2].metric("Documents", tm.n_docs)
    primary = tm.primary_metric
    val = tm.coherence_overall.get(primary)
    c[3].metric(f"Coherence ({primary})", f"{val:.3f}" if val is not None else "n/a")
    if primary == "c_v" and val is not None:
        verdict = "good (> 0.55)" if val > 0.55 else "fair (0.4 - 0.55)" if val > 0.4 else "weak (< 0.4)"
        st.caption(f"c_v coherence is **{verdict}**. Higher = topic words co-occur more and are easier to interpret.")
    else:
        st.caption("NPMI coherence ranges from -1 to 1; higher means the top words of a topic co-occur more often.")
    if tm.sweep:
        df = pd.DataFrame(tm.sweep).set_index("k")
        cols = [x for x in ("c_v", "npmi") if df[x].notna().any()]
        st.markdown("**Coherence vs. number of topics** (the best k is chosen automatically)")
        st.line_chart(df[cols])
    st.markdown("### Topics")
    for t in tm.topics:
        col1, col2 = st.columns([1, 3])
        newname = col1.text_input(f"Name for topic {t['id']}", value=topic_names().get(t["id"], t["label"]),
                                  key=f"tn_{t['id']}")
        st.session_state["topic_names"][t["id"]] = newname
        members = sorted([cid for cid, v in tm.concept_topic.items() if v["topic"] == t["id"]],
                         key=lambda cid: -tm.concept_topic[cid]["confidence"] * store.concepts[cid].freq)
        coh = ", ".join(f"{k}={v:.2f}" for k, v in t["coherence"].items() if v == v and v is not None)
        col2.markdown(chip(newname, topic_color(t["id"])) + f" <span class='cc-small'>coherence: {coh}</span>",
                      unsafe_allow_html=True)
        col2.markdown("**Top words:** " + ", ".join(f"{w} ({p:.2f})" for w, p in t["words"][:8]))
        col2.markdown("**Concepts:** " + (", ".join(names[c] for c in members[:15]) or "-"))
        st.divider()

# ---------------------------------------------------------------------------
# Tab 6: Language models
# ---------------------------------------------------------------------------
with tabs[5]:
    howto("What is this?", """This tab is about **n-gram language models**. The tables show the most frequent word sequences. The **perplexity table** compares unigram, bigram and trigram models on text they did not train on (**lower is better**). Try the *next-word prediction* box: type a few words and the model guesses what comes next.""", True)
    st.markdown("### Frequent n-grams")
    cols = st.columns(3)
    for col, key in zip(cols, ("unigram", "bigram", "trigram")):
        col.markdown(f"**{key.title()}s**")
        col.dataframe(pd.DataFrame(res.ngram_tables[key], columns=[key, "count"]), hide_index=True)
    st.markdown("### Collocations (pointwise mutual information)")
    st.dataframe(pd.DataFrame(res.collocations, columns=["bigram", "count", "PMI"]), hide_index=True)
    st.markdown("### Language-model evaluation: perplexity")
    rep = res.lm_report
    note = "held-out 20 % of sentences" if rep["held_out"] else "text too short for a held-out split (train = test)"
    st.caption(f"Add-k smoothed models; test set = {note}. Lower perplexity is better. "
               f"Train sentences: {rep['n_train']}, test sentences: {rep['n_test']}, vocabulary: {rep['vocab']}.")
    df = pd.DataFrame(rep["rows"])
    st.dataframe(df.drop(columns=["n"]), hide_index=True)
    pv = df.pivot(index="model", columns="add-k", values="test perplexity")
    pv.columns = [f"k={c}" for c in pv.columns]
    st.bar_chart(pv)
    if rep["best"]:
        b = rep["best"]
        st.success(f"Best on held-out text: **{b['model']}** with add-k = {b['add-k']} "
                   f"(perplexity {b['test perplexity']:.1f}).")
    st.markdown("### Next-word prediction (trigram model with back-off)")
    ctx = st.text_input("Type a few words", "an n-gram model", key="lm_ctx")
    words = [w.lower() for w in ctx.split()]
    preds = res.predictor.predict_next(words, 6) if words else []
    st.write(", ".join(f"**{w}** ({p:.2f})" for w, p in preds) or "No prediction available.")

# ---------------------------------------------------------------------------
# Tab 7: NLP pipeline inspector
# ---------------------------------------------------------------------------
with tabs[6]:
    howto("What is this?", """Pick any sentence and watch how it is processed: **tokens and lemmas**, then (with spaCy) **parts of speech, dependency tree and named entities**, then the **triples** found in that sentence. Use it to explain to your teacher how a sentence becomes a fact in the graph.""", True)
    st.markdown("Inspect how a single sentence flows through the pipeline: tokenization -> lemmas -> "
                "POS / dependencies -> concept mentions -> predicate-argument triples.")
    multi = [s for s in res.sentences if len(s.mentions) >= 2]
    choices = multi[:200] or res.sentences[:200]
    pick_s = st.selectbox("Pick a sentence from the text", range(len(choices)),
                          format_func=lambda i: choices[i].text[:110], key="n_pick")
    text = st.text_area("...or type your own sentence", value=choices[pick_s].text, height=80,
                        key=f"n_text_{pick_s}")
    toks = tokenize(text)
    s_obj = Sentence(sid=-1, text=text, page=0, para=0, tokens=toks,
                     lemmas=[simple_lemma(t[0]) if is_word(t[0]) else t[0] for t in toks])
    annotate_mentions([s_obj], store.matcher)
    mention_of = {}
    for mm in s_obj.mentions:
        for i in range(mm.start_tok, mm.end_tok):
            mention_of[i] = names[mm.cid]
    st.markdown("**1 · Tokenization, lemmatization, stop words, concept mentions**")
    st.dataframe(pd.DataFrame([{"Token": t[0], "Lemma": s_obj.lemmas[i],
                                "Stop word": t[0].lower() in STOPWORDS, "Concept": mention_of.get(i, "")}
                               for i, t in enumerate(toks)]), hide_index=True)
    doc = None
    if nlp is not None:
        doc = nlp(text)
        st.markdown("**2 · spaCy: POS tags, dependency parse, named entities**")
        st.dataframe(pd.DataFrame([{"Token": t.text, "Lemma": t.lemma_, "POS": t.pos_, "Tag": t.tag_,
                                    "Dependency": t.dep_, "Head": t.head.text,
                                    "Entity": t.ent_type_} for t in doc]), hide_index=True)
        try:
            from spacy import displacy
            svg = displacy.render(doc, style="dep", options={"compact": True, "distance": 90})
            show_html(f"<div style='background:#fff;padding:8px;overflow-x:auto'>{svg}</div>", 330)
        except Exception:  # noqa: BLE001
            pass
    else:
        st.info("Install the spaCy model to see POS tags, dependency parses and named entities "
                "(`python -m spacy download en_core_web_sm`).")
    st.markdown("**3 · Predicate-argument triples found in this sentence**")
    found = []
    if doc is not None:
        found += [(a, r, b, "dependency") for a, r, b in dependency_triples(doc, s_obj)]
    found += [(a, r, b, "pattern") for a, r, b in pattern_triples(s_obj)]
    if found:
        for a, r, b, src_ in dict.fromkeys(found):
            st.code(f"{r.replace(' ', '_')}({a}, {b})    # {names[a]} -[{r}]-> {names[b]}   [{src_}]", language="text")
    else:
        st.write("No triple found (need two known concepts connected by a relation phrase).")

# ---------------------------------------------------------------------------
# Tab 8: Export
# ---------------------------------------------------------------------------
with tabs[7]:
    howto("What is this?", """Download your results: concept frames (JSON), triples (CSV), the graph (JSON or GraphML for Gephi/Cytoscape), and the interactive graph as an HTML file you can open in any browser or send to someone.""", False)
    tn = topic_names()
    frames_now = build_meaning_representations(store, res.triples, G, P, res.topics, tn)
    triples_df = pd.DataFrame([{"subject": names[t.subj], "predicate": t.rel, "object": names[t.obj],
                                "type": t.rel_type, "count": t.count, "page": min(t.pages), "sentence": t.sentence}
                               for t in res.triples])
    graphml = "\n".join(nx.generate_graphml(graph_for_graphml(G)))
    html_full = render_cached(res, 800, topic_names=tn)
    d1, d2, d3 = st.columns(3)
    d1.download_button("Concept frames (JSON)", json.dumps(frames_now, indent=2, ensure_ascii=False),
                       "concept_frames.json", "application/json")
    d2.download_button("Triples (CSV)", triples_df.to_csv(index=False), "triples.csv", "text/csv")
    d3.download_button("Graph (node-link JSON)", json.dumps(graph_to_node_link(G), indent=2, default=str),
                       "knowledge_graph.json", "application/json")
    d4, d5 = st.columns(2)
    d4.download_button("Graph (GraphML - Gephi/Cytoscape)", graphml, "knowledge_graph.graphml", "application/xml")
    d5.download_button("Interactive graph (HTML)", html_full, "knowledge_graph.html", "text/html")
    st.caption(f"Parser used: {res.parser}  ·  Topic engine: {res.topics.engine}")

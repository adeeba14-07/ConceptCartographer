"""ConceptCartographer - upload a chapter, search any concept, see it on the map, study it.

Run:  streamlit run app.py
(app_advanced.py contains the full version with every NLP detail tab)
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os

import networkx as nx
import streamlit as st

from ccart import library, sources
from ccart.config import STOPWORDS
from ccart.explain import SOURCE_TEXT, build_tips, fill_with_ai, make_anthropic_completer
from ccart.graph import topic_color
from ccart.pathfinder import find_learning_path
from ccart.pipeline import Params, run_pipeline
from ccart.preprocess import extract_pdf_pages, is_word, key_of, pages_from_text
from ccart.sample_text import SAMPLE_CHAPTER
from ccart.visualize import render_graph_html

st.set_page_config(page_title="ConceptCartographer", page_icon="🗺️", layout="wide")
st.markdown(
    """<style>
.block-container {padding-top: 1.4rem; max-width: 1400px;}
.chip {display:inline-block; padding:3px 12px; margin:3px 5px 3px 0; border-radius:999px;
       font-size:.88rem; color:#0b1020; font-weight:600;}
.card {border:1px solid rgba(128,128,128,.35); border-radius:12px; padding:14px 18px; margin:6px 0;}
.step {border-left:4px solid #6d28d9; padding:6px 14px; margin:6px 0; background:rgba(109,40,217,.08);
       border-radius:0 10px 10px 0;}
</style>""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading the language model (first time only)...")
def load_nlp():
    try:
        import importlib
        import spacy
    except Exception:  # noqa: BLE001
        return None
    for attempt in range(2):
        try:
            nlp = spacy.load("en_core_web_sm")
            nlp.max_length = 3_000_000
            return nlp
        except OSError:
            if attempt == 0:
                try:
                    spacy.cli.download("en_core_web_sm")
                    importlib.invalidate_caches()
                except BaseException:  # noqa: BLE001
                    break
    return None


@st.cache_data(show_spinner="Reading the PDF...")
def read_pdf(digest: str, data: bytes):
    return extract_pdf_pages(data)


def show_html(html: str, height: int):
    try:
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)
    except Exception:  # noqa: BLE001
        st.warning("Could not show the interactive graph in this Streamlit version.")


def render_cached(res, height: int, **kw) -> str:
    cache = st.session_state.setdefault("_render_cache", {})
    key = (id(res), res.explain_version, height, json.dumps(kw, sort_keys=True, default=str))
    if key not in cache:
        if len(cache) > 15:
            cache.clear()
        tips = build_tips(res.G, res.explain, kw.get("topic_names") or {})
        cache[key] = render_graph_html(res.G, height=height, tips=tips, **kw)
    return cache[key]


def chip(text: str, color: str) -> str:
    return f"<span class='chip' style='background:{color}'>{text}</span>"


def contains_phrase(sent, qk) -> bool:
    words = [l for (t, _, _), l in zip(sent.tokens, sent.lemmas) if is_word(t)]
    n = len(qk)
    return any(tuple(words[i:i + n]) == tuple(qk) for i in range(len(words) - n + 1))


nlp = load_nlp()


def save_current():
    """Keep the saved copy in the library up to date (after a build, an added concept or AI definitions)."""
    cur, inp, r = st.session_state.get("current"), st.session_state.get("inputs"), st.session_state.get("result")
    if not (cur and inp and r):
        return
    try:
        library.save(cur, inp["pages"], inp["params"], r)
        st.session_state.pop("lib_error", None)
    except Exception as exc:  # noqa: BLE001 - e.g. a read-only folder: the map itself still works
        st.session_state["lib_error"] = str(exc)


def do_build(build_pages, params, meta=None) -> bool:
    bar = st.progress(0.0, text="Starting...")
    try:
        res_new = run_pipeline(build_pages, params, nlp, lambda f, m: bar.progress(min(max(f, 0), 1.0), text=m))
    except ValueError as exc:
        bar.empty()
        st.error(str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        bar.empty()
        st.error(f"Something went wrong: {exc}")
        return False
    bar.empty()
    for k in [k for k in st.session_state if k.startswith("s_") and k != "s_find"]:
        del st.session_state[k]
    st.session_state["result"] = res_new
    st.session_state["inputs"] = {"pages": build_pages, "params": params}
    if meta:
        st.session_state["current"] = meta
    save_current()
    return True


def _clear_view_state():
    for k in [k for k in st.session_state if k.startswith("s_")]:
        del st.session_state[k]
    st.session_state["tried_terms"] = set()


def open_doc(doc_id_: str):  # button callbacks run before the page is drawn
    st.session_state["open_doc"] = doc_id_


def ask_delete(doc_id_: str):
    if st.session_state.get("confirm_del") == doc_id_:  # second click confirms
        library.delete(doc_id_)
        st.session_state.pop("confirm_del", None)
        if (st.session_state.get("current") or {}).get("id") == doc_id_:
            st.session_state.pop("current", None)
    else:
        st.session_state["confirm_del"] = doc_id_


def when(ts: float) -> str:
    d = datetime.datetime.fromtimestamp(ts)
    today = datetime.date.today()
    if d.date() == today:
        return "Today " + d.strftime("%H:%M")
    if d.date() == today - datetime.timedelta(days=1):
        return "Yesterday"
    return d.strftime("%d %b %Y")


# open a saved document chosen in the sidebar
if st.session_state.get("open_doc"):
    _id = st.session_state.pop("open_doc")
    try:
        _meta, _pages, _params, _res = library.load(_id)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not open that saved document: {exc}")
    else:
        _clear_view_state()
        st.session_state["s_find"] = ""
        st.session_state["current"] = _meta
        if _res is not None:
            st.session_state["result"] = _res
            st.session_state["inputs"] = {"pages": _pages, "params": _params}
        else:
            st.info("Rebuilding this saved document (it was saved by an older version)...")
            do_build(_pages, _params, _meta)

st.title("ConceptCartographer")
st.caption("Upload a textbook chapter, search any concept, and see it on an interactive map.")

# ------------------------------------------------------------------ sidebar
loaded, sel_pages, first_page, title = None, [], 1, ""
with st.sidebar:
    st.header("📚 My library")
    try:
        docs = library.list_docs()
    except Exception:  # noqa: BLE001
        docs = []
    if not docs:
        st.caption("Every document you build is saved here, so you can come back and keep studying later.")
    cur_id = (st.session_state.get("current") or {}).get("id")
    for m in docs[:40]:
        c1, c2 = st.columns([6, 1])
        c1.button(f"{'▶ ' if m['id'] == cur_id else ''}{sources.KIND_ICON.get(m.get('kind'), '📄')} {m['title'][:32]}",
                  key=f"lib_open_{m['id']}", on_click=open_doc, args=(m["id"],))
        c1.caption(f"{when(m.get('updated', 0))} · {m.get('n_concepts', 0)} concepts")
        c2.button("❓" if st.session_state.get("confirm_del") == m["id"] else "🗑", key=f"lib_del_{m['id']}",
                  on_click=ask_delete, args=(m["id"],), help="Delete this saved document (click twice to confirm)")
    if st.session_state.get("lib_error"):
        st.warning("Could not save to the library: " + st.session_state["lib_error"][:120] +
                   ". The map itself still works.")
    st.divider()

    st.header("➕ Add a document")
    source = st.radio("Source", ["Sample chapter", "Upload file", "Web link or YouTube", "Paste text"],
                      label_visibility="collapsed")
    if source == "Upload file":
        up = st.file_uploader("PDF, PowerPoint (.pptx), Word (.docx) or text", type=["pdf", "pptx", "docx", "txt"])
        if up is not None:
            data, ext = up.getvalue(), up.name.lower().rsplit(".", 1)[-1]
            stem = up.name.rsplit(".", 1)[0]
            try:
                if ext == "pdf":
                    pg = read_pdf(hashlib.md5(data).hexdigest(), data)
                    if not any(p.strip() for p in pg):
                        raise ValueError("No text found. Scanned PDFs need OCR first.")
                    loaded = sources.Loaded(stem, pg, "pdf", up.name)
                elif ext == "pptx":
                    loaded = sources.load_pptx(data, up.name)
                elif ext == "docx":
                    loaded = sources.load_docx(data, up.name)
                else:
                    loaded = sources.Loaded(stem, pages_from_text(data.decode("utf-8", errors="ignore")), "txt", up.name)
            except ValueError as exc:
                st.error(str(exc))
    elif source == "Web link or YouTube":
        url = st.text_input("Paste a link", placeholder="https://en.wikipedia.org/wiki/...  or a YouTube link").strip()
        st.caption("Works with Wikipedia and other article pages, PDF / PowerPoint links, and YouTube videos that "
                   "have captions.")
        if url:
            cache = st.session_state.setdefault("link_cache", {})
            if url not in cache:
                with st.spinner("Reading the link..."):
                    try:
                        cache[url] = sources.load_url(url)
                    except (ValueError, RuntimeError) as exc:
                        cache[url] = str(exc)
            if isinstance(cache[url], str):
                st.error(cache[url])
                st.button("Try again", key="link_retry", on_click=lambda: cache.pop(url, None))
            else:
                loaded = cache[url]
    elif source == "Paste text":
        txt = st.text_area("Paste your text", height=200, placeholder="Paste notes or an article here...")
        if txt.strip():
            loaded = sources.Loaded("Pasted text " + datetime.date.today().strftime("%d %b"), pages_from_text(txt),
                                    "paste", "pasted text")
    else:
        loaded = sources.Loaded("Sample chapter (NLP)", pages_from_text(SAMPLE_CHAPTER), "sample", "sample")

    if loaded:
        digest = library.doc_id(loaded.kind, loaded.source, loaded.pages)
        sel_pages = loaded.pages
        n = len(loaded.pages)
        st.caption(f"{sources.KIND_ICON.get(loaded.kind, '📄')} {n} {'slide' if loaded.kind == 'pptx' else 'page'}(s) found.")
        if loaded.note:
            st.info(loaded.note)
        if loaded.kind in ("pdf", "pptx") and n > 1:
            c1, c2 = st.columns(2)
            a = c1.number_input("From", 1, n, 1, key=f"from_{digest}")
            b = c2.number_input("To", 1, n, n, key=f"to_{digest}")
            if b < a:
                st.warning("'To' must not be smaller than 'From'.")
                sel_pages = []
            else:
                sel_pages, first_page = loaded.pages[int(a) - 1:int(b)], int(a)
        title = st.text_input("Name", value=loaded.title, key=f"title_{digest}")

    with st.expander("Optional settings"):
        amount = st.selectbox("Concepts to find", ["Automatic (recommended)", "Fewer (40)", "More (150)"])
        auto_k = st.checkbox("Choose number of topics automatically", True)
        k_fixed = st.number_input("Number of topics (if not automatic)", 2, 10, 4)
        include = st.text_area("Always include these concepts (one per line)", height=70)
        ignore = st.text_area("Ignore these words (one per line)", placeholder="cat\ncricket", height=70)

    with st.expander("AI definitions (optional)"):
        st.caption("The document's own definitions are always used first. If you add a key, concepts that the "
                   "document does NOT define get a simple definition written by AI (clearly labelled).")
        default_key = os.environ.get("ANTHROPIC_API_KEY", "")
        try:
            default_key = default_key or st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:  # noqa: BLE001 - no secrets file
            pass
        ai_key = st.text_input("Anthropic API key", value=default_key, type="password")

    run = st.button("Build map", type="primary", disabled=not sel_pages)
    

if run and loaded and sel_pages:
    st.session_state["tried_terms"] = set()
    do_build(sel_pages, Params(
        max_concepts={"Automatic (recommended)": 0, "Fewer (40)": 40, "More (150)": 150}[amount],
        auto_topics=auto_k, num_topics=int(k_fixed), k_min=3, k_max=7, first_page=first_page,
        exclude_terms=[l.strip() for l in ignore.splitlines() if l.strip()],
        extra_terms=[l.strip() for l in include.splitlines() if l.strip()]
        + (["bag of words"] if loaded.kind == "sample" else [])),
        meta={"id": library.doc_id(loaded.kind, loaded.source, sel_pages), "title": title.strip() or loaded.title,
              "kind": loaded.kind, "source": loaded.source, "first_page": first_page})

res = st.session_state.get("result")
if res is None:
    st.info("👈 Pick a text in the sidebar and press **Build map**. Start with the sample chapter.")
    st.stop()

G, P, store, tm = res.G, res.P, res.store, res.topics
key_tag = hashlib.md5(ai_key.encode()).hexdigest() if ai_key else ""
if ai_key and res.ai_report.get("key") != key_tag:
    with st.spinner("Writing simple definitions for concepts the document does not define..."):
        rep = fill_with_ai(res.explain, store, res.sentences, G, make_anthropic_completer(ai_key))
    rep["key"] = key_tag
    res.ai_report = rep
    res.explain_version += 1
    save_current()
    if rep["errors"] and not rep["written"]:
        st.warning("AI definitions could not be written (" + rep["errors"][0][:150] + "). "
                   "Showing definitions built from the document instead.")
names = {cid: c.name for cid, c in store.concepts.items()}
tlabel = {t["id"]: t["label"] for t in tm.topics}
used_topics = [t for t in tm.topics if any(v["topic"] == t["id"] for v in tm.concept_topic.values())]
_cur = st.session_state.get("current") or {}
if _cur.get("title"):
    st.markdown(f"**{sources.KIND_ICON.get(_cur.get('kind'), '📄')} Studying:** {_cur['title']}")
st.success(f"Found **{res.stats['concepts']} concepts**, **{res.stats['triple_edges']} relations** "
           f"and **{res.stats['topics']} topics** in {res.stats['sentences']} sentences.")

tab_map, tab_path, tab_topics = st.tabs(["🗺️ Map & search", "🧭 Learning path", "🧠 Topics"])


def concept_card(cid: str):
    c, d, e = store.concepts[cid], G.nodes[cid], res.explain[cid]
    badge_color = {"document": "#34d399", "document_section": "#34d399", "ai": "#fbbf24", "auto": "#94a3b8"}[e["def_source"]]
    how = (f"<p style='margin:6px 0 0 0'><b>How it works:</b> {e['how']}"
           f"{' <i>(AI)</i>' if e['how_source'] == 'ai' else ''}</p>") if e["how"] else ""
    st.markdown(f"<div class='card'><h3 style='margin:0'>{c.name}</h3>"
                f"{chip(tlabel.get(d.get('topic'), '-'), topic_color(d.get('topic')))}"
                f"<span style='opacity:.7'>page {c.pages[0] if c.pages else '-'} · mentioned {c.freq}×</span>"
                f"<p style='margin-top:8px'>{e['definition']}</p>"
                f"<span style='font-size:.8rem;color:{badge_color}'>● {SOURCE_TEXT[e['def_source']]}</span>"
                f"{how}</div>", unsafe_allow_html=True)
    succ, pred = set(G.successors(cid)), set(G.predecessors(cid))
    subs = sorted(x for x in succ if G[cid][x].get("rel_type") == "subtopic")
    parent = sorted(x for x in pred if G[x][cid].get("rel_type") == "subtopic")
    covers = sorted(x for x in succ if G[cid][x].get("kind") == "section" and x not in subs)
    first = [x for x in P.predecessors(cid) if x not in parent]
    then = [x for x in P.successors(cid) if x not in subs]
    shown = set(subs) | set(parent) | set(covers) | set(first) | set(then)
    rel = sorted(x for x in (succ | pred) if x not in shown)
    if parent:
        st.markdown("**📍 Part of:** " + ", ".join(names[x] for x in parent))
    if subs:
        st.markdown("**📂 Sub-topics:** " + ", ".join(names[x] for x in subs))
    if covers:
        st.markdown("**📖 Explained in this section:** " + ", ".join(names[x] for x in covers[:12]))
    st.markdown("**📘 Learn first:** " + (", ".join(names[x] for x in first) or "nothing needed"))
    st.markdown("**➡️ Then learn:** " + (", ".join(names[x] for x in then) or "-"))
    st.markdown("**🔗 Related:** " + (", ".join(names[x] for x in sorted(rel, key=lambda n: names[n])[:10]) or "-"))
    with st.expander("Sentences from the text"):
        for sid in c.sentence_ids[:5]:
            st.markdown(f"- (page {res.sentences[sid].page}) {res.sentences[sid].text}")


# ------------------------------------------------------------------ map & search
with tab_map:
    q_raw = st.text_input("🔎 Search any concept or word from your chapter", key="s_find",
                          placeholder="e.g. morphological models, finite-state machine, stemming").strip()
    qk = key_of(q_raw)
    focus = None
    if qk:
        st.button("← Back to full map", key="s_back", on_click=lambda: st.session_state.update({"s_find": ""}))
        inputs = st.session_state["inputs"]
        failed = st.session_state.setdefault("s_failed", set())
        ranked = sorted(store.concepts, key=lambda c: -G.nodes[c].get("importance", 0))

        def hay(cid: str):
            return list(store.concepts[cid].key) + [a.lower() for a in store.concepts[cid].aliases]

        full = [c for c in ranked if all(any(t in h for h in hay(c)) for t in qk)]
        qtok = [t for t in qk if t not in STOPWORDS] or list(qk)
        need = max(1, -(-len(qtok) // 2))
        scored_c = sorted(((sum(1 for t in qtok if any(t in h for h in hay(c))), c) for c in ranked),
                          key=lambda x: -x[0])
        closest = [c for n_, c in scored_c if n_ >= need]
        exact = [s_ for s_ in res.sentences if contains_phrase(s_, qk)]

        tried = st.session_state.setdefault("tried_terms", set())  # one automatic attempt per phrase: no loops

        def add_as_concept() -> bool:
            tried.add(q_raw.lower())
            newp = copy.deepcopy(inputs["params"])  # only saved if the rebuild really finishes
            newp.extra_terms.append(q_raw)
            ok = do_build(inputs["pages"], newp)
            if not ok:
                failed.add(q_raw.lower())
            return ok

        if not full and exact and q_raw.lower() not in tried and len(q_raw) >= 3:
            st.info(f"“{q_raw}” is in your PDF but is not a concept yet. Adding it to the map - please wait...")
            if add_as_concept():
                st.rerun()
        candidates = full or closest  # (reached only if the term is already a concept or adding it failed)
        if candidates:
            focus = candidates[0] if len(candidates) == 1 else st.selectbox(
                f"{len(candidates)} matching concepts" if full else "Closest concepts", candidates[:30],
                format_func=lambda c: names[c], key="s_hit")
            if not full:
                st.info(f"No concept is named exactly “{q_raw}”, so the map shows the closest match.")
        if exact and not full:
            with st.expander(f"Where “{q_raw}” appears in your PDF ({len(exact)} sentence(s))", expanded=not focus):
                for s_ in exact[:8]:
                    st.markdown(f"- (page {s_.page}) {s_.text}")
                if focus and st.button(f"➕ Add “{q_raw}” as its own concept", key="s_addexact"):
                    if add_as_concept():
                        st.rerun()
        if not focus and not exact:
            loose = [s_ for s_ in res.sentences if all(any(t in lem for lem in s_.lemmas) for t in qk)]
            if loose:
                st.warning(f"No exact match for “{q_raw}”. These sentences contain similar words:")
                for s_ in loose[:6]:
                    st.markdown(f"- (page {s_.page}) {s_.text}")
            else:
                st.error(f"“{q_raw}” was not found in the selected pages. Check the spelling and the page range.")

    if focus:
        near = {focus} | set(G.successors(focus)) | set(G.predecessors(focus))
        if len(near) > 30:
            def _prio(n):
                d = G.get_edge_data(focus, n) or G.get_edge_data(n, focus) or {}
                return (0 if d.get("kind") == "structure" else 1, -G.nodes[n].get("importance", 0))
            near = {focus} | set(sorted(near - {focus}, key=_prio)[:29])
        left, right = st.columns([3, 1.5])
        with left:
            show_html(render_cached(res, 600, topic_names=tlabel, show_cooc=True, show_edge_labels=True,
                                    hide_isolated=False, nodes_filter=sorted(near), highlight_path=[focus]), 610)
            st.caption(f"Showing **{names[focus]}** (red outline) and the concepts connected to it. "
                       "Press **← Back to full map** to see all topics again.")
        with right:
            concept_card(focus)
    elif not qk:
        c1, c2, c3 = st.columns([2, 1.5, 1.5])
        choice = c1.selectbox("Filter by topic (optional)", ["All topics"] + [t["id"] for t in used_topics],
                              format_func=lambda x: x if x == "All topics" else tlabel[x], key="s_topic")
        total = G.number_of_nodes()
        shown = c2.slider("Concepts drawn", min(10, total), total, min(45, total), key="s_shown") \
            if total > 10 else total
        weak = c3.checkbox("Show weak links (dashed)", False, key="s_weak")
        pool = [n for n, d in G.nodes(data=True) if choice == "All topics" or d.get("topic") == choice]
        keep = sorted(pool, key=lambda n: -G.nodes[n].get("importance", 0))[:shown]
        st.markdown("".join(chip(t["label"], topic_color(t["id"])) for t in used_topics), unsafe_allow_html=True)
        show_html(render_cached(res, 640, topic_names=tlabel, show_cooc=weak, show_edge_labels=True,
                                hide_isolated=True, nodes_filter=sorted(keep)), 650)
        st.caption("Bubbles are concepts, arrows are facts from the text, colours are topics. "
                   "Hover for a definition. Use the search box above to jump to any concept.")

# ------------------------------------------------------------------ learning path
with tab_path:
    st.caption("Pick what you already know and what you want to learn. You can type inside each box to search the list.")
    allc = sorted(store.concepts, key=lambda c: names[c].lower())
    top_c = max(allc, key=lambda c: G.nodes[c].get("importance", 0))
    reach = nx.single_source_shortest_path_length(G.to_undirected(), top_c)
    far_c = max(reach, key=lambda n: reach[n]) if len(reach) > 1 else top_c
    x1, x2 = st.columns(2)
    src = x1.selectbox("I already know", allc, index=allc.index(top_c), format_func=lambda c: names[c], key="s_from")
    dst = x2.selectbox("I want to learn", allc, index=allc.index(far_c), format_func=lambda c: names[c], key="s_to")
    if src == dst:
        st.info("Choose two different concepts.")
    else:
        out = find_learning_path(G, P, src, dst, True)
        if not out["found"]:
            st.error(out["reason"])
        else:
            for i, step in enumerate(out["steps"], 1):
                st.markdown(f"<div class='step'><b>{i}.</b> {names[step['from']]} <code>{step['arrow']}</code> "
                            f"{names[step['to']]}</div>", unsafe_allow_html=True)
            st.markdown("**Study in this order:** " + " → ".join(names[x] for x in out["study_order"]))
            show_html(render_cached(res, 480, topic_names=tlabel, highlight_path=out["path"],
                                    hide_isolated=True, nodes_filter=sorted(set(out["path"]) |
                                    set(out["study_order"]))), 490)

# ------------------------------------------------------------------ topics
with tab_topics:
    st.caption("Concepts that tend to appear together are grouped into a topic (LDA). "
               "The same colours are used on the map.")
    score = tm.coherence_overall.get(tm.primary_metric)
    if score is not None:
        st.markdown(f"Topic quality score ({tm.primary_metric}): **{score:.2f}** - higher means cleaner topics.")
    for t in used_topics:
        members = sorted([cid for cid, v in tm.concept_topic.items() if v["topic"] == t["id"]],
                         key=lambda cid: -G.nodes[cid].get("importance", 0))
        st.markdown(f"<div class='card'>{chip(t['label'], topic_color(t['id']))}"
                    f"<div style='margin-top:6px'>" + " · ".join(names[m] for m in members) +
                    "</div></div>", unsafe_allow_html=True)

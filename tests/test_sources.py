"""Loaders (PPTX, DOCX, web, PDF link, YouTube), link safety, and the saved library.  python tests/test_sources.py"""
import io, os, sys, tempfile, zipfile, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccart import library, sources
from ccart.pipeline import Params, run_pipeline

NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')

def sp(text_paras, ph=None):
    phx = f'<p:ph type="{ph}"/>' if ph else '<p:ph idx="1"/>'
    paras = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in text_paras)
    return f'<p:sp><p:nvSpPr><p:cNvPr id="2" name="x"/><p:cNvSpPr/><p:nvPr>{phx}</p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/>{paras}</p:txBody></p:sp>'

def slide_xml(title, bullets, footer=True):
    shapes = sp([title], "title") + sp(bullets)
    if footer: shapes += sp(["7"], "sldNum")
    return f'<?xml version="1.0"?><p:sld {NS}><p:cSld><p:spTree>{shapes}</p:spTree></p:cSld></p:sld>'

def make_pptx(slides):
    """slides: [(title, bullets, notes)] ; file order is REVERSED vs presentation order to test ordering."""
    buf = io.BytesIO(); z = zipfile.ZipFile(buf, "w")
    n = len(slides)
    ids, rels = "", ""
    for i, (title, bullets, notes) in enumerate(slides):
        fname = n - i                       # slide file numbers run backwards
        z.writestr(f"ppt/slides/slide{fname}.xml", slide_xml(title, bullets))
        ids += f'<p:sldId id="{256+i}" r:id="rId{i+10}"/>'
        rels += f'<Relationship Id="rId{i+10}" Type="x/slide" Target="slides/slide{fname}.xml"/>'
        if notes:
            z.writestr(f"ppt/slides/_rels/slide{fname}.xml.rels",
                       f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                       f'<Relationship Id="rId1" Type="http://x/notesSlide" Target="../notesSlides/notesSlide{fname}.xml"/></Relationships>')
            z.writestr(f"ppt/notesSlides/notesSlide{fname}.xml",
                       f'<p:notes {NS}><p:cSld><p:spTree>{sp(["slide image"], "sldImg")}{sp([notes])}</p:spTree></p:cSld></p:notes>')
    z.writestr("ppt/presentation.xml", f'<p:presentation {NS}><p:sldIdLst>{ids}</p:sldIdLst></p:presentation>')
    z.writestr("ppt/_rels/presentation.xml.rels",
               f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>')
    z.close(); return buf.getvalue()

def make_docx(paras):
    W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    body = ""
    for style, text in paras:
        ps = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        body += f"<w:p>{ps}<w:r><w:t>{text}</w:t></w:r></w:p>"
    buf = io.BytesIO(); z = zipfile.ZipFile(buf, "w")
    z.writestr("word/document.xml", f'<w:document {W}><w:body>{body}</w:body></w:document>'); z.close()
    return buf.getvalue()

WIKI = """<html><head><title>Photosynthesis - Wikipedia</title></head><body>
<nav>Main page Contents Random article Donate Create account Log in</nav>
<div id="mw-content-text"><h1>Photosynthesis</h1>
<div class="infobox">Infobox noise words about nothing useful at all here</div>
<p>Photosynthesis is the process by which plants make glucose from sunlight.<sup class="reference">[1]</sup> Photosynthesis takes place in the chloroplasts. Chloroplasts contain chlorophyll and chlorophyll absorbs sunlight.</p>
<div class="mw-heading mw-heading2"><h2 id="L">Light reactions</h2><span class="mw-editsection">[edit]</span></div>
<p>The light reactions capture sunlight in the chloroplast membranes. The light reactions produce ATP and release oxygen from water.</p>
<div class="mw-heading mw-heading3"><h3 id="C">Chlorophyll</h3><span class="mw-editsection">[edit]</span></div>
<p>Chlorophyll is the green pigment that drives the light reactions. Chlorophyll absorbs red and blue light.</p>
<h2>Calvin cycle</h2>
<p>The Calvin cycle builds glucose from carbon dioxide. The Calvin cycle uses ATP from the light reactions.</p>
<h2>References</h2><ol><li>Ref text should never appear in the output of the reader at all</li></ol></div>
<footer>Privacy policy Terms of use Mobile view</footer></body></html>"""

def test_pptx():
    data = make_pptx([("Cell Biology", ["The cell is the basic unit of life", "A cell membrane controls what enters the cell"], "The cell membrane is selectively permeable and protects the cell."),
                      ("Photosynthesis", ["Plants make glucose from sunlight", "Chlorophyll absorbs sunlight in the chloroplast"], ""),
                      ("Thank you", ["Questions are welcome at any time"], "")])
    ld = sources.load_pptx(data, "biology.pptx")
    text = "\n".join(ld.pages)
    assert ld.kind == "pptx" and ld.title == "biology"
    assert ld.pages[0].startswith("1. Cell Biology") and ld.pages[1].startswith("2. Photosynthesis"), ld.pages[:2]  # presentation order, not file order
    assert "selectively permeable" in text, "speaker notes missing"
    assert "slide image" not in text and "\n7" not in text, "placeholder noise leaked"
    assert "3. Thank you" not in text
    print("  pptx ok:", len(ld.pages), "slides")

def test_docx():
    ld = sources.load_docx(make_docx([("Title", "Study notes on databases"), ("Heading1", "Relational Databases"),
        (None, "A relational database stores data in tables with rows and columns."), ("Heading2", "Primary Keys"),
        (None, "A primary key identifies each row of a table in the database."), ("Heading1", "Indexes"),
        (None, "An index speeds up queries on a table in the database.")]), "notes.docx")
    t = "\n".join(ld.pages)
    assert "2. Relational Databases" in t and "2.1 Primary Keys" in t and "3. Indexes" in t, t[:300]
    print("  docx ok")

def test_html():
    title, pages = sources.html_to_pages(WIKI)
    t = "\n".join(pages)
    assert title == "Photosynthesis", title
    for bad in ("Main page", "Infobox noise", "[edit]", "[1]", "Ref text", "Privacy policy"):
        assert bad not in t, f"noise leaked: {bad}"
    assert "1. Light reactions" in t and "1.1 Chlorophyll" in t and "2. Calvin cycle" in t, t[:500]
    print("  html ok")

def test_pipeline_on_loaded():
    title, pages = sources.html_to_pages(WIKI)
    res = run_pipeline(pages, Params(max_concepts=0, min_freq=2))
    assert res.stats["outline_edges"] >= 1, "section structure from web headings not used"
    names = {c.name for c in res.store.concepts.values()}
    assert any("Light" in n for n in names) and any("Calvin" in n for n in names), names
    print("  pipeline on web page ok:", len(names), "concepts,", res.stats["outline_edges"], "section links")

def test_links():
    for bad in ("http://127.0.0.1/x", "http://localhost:8501", "ftp://example.org/a", "file:///etc/passwd"):
        try:
            sources.load_url(bad); raise SystemExit(f"NOT BLOCKED: {bad}")
        except ValueError:
            pass
    ok = lambda u: (WIKI.encode(), "text/html; charset=utf-8", u)
    ld = sources.load_url("en.wikipedia.org/wiki/Photosynthesis", fetch=ok)
    assert ld.kind == "web" and ld.title == "Photosynthesis" and ld.source.startswith("https://")
    # text link and PDF link routing
    ld = sources.load_url("https://x.org/a.txt", fetch=lambda u: (("Cells are small. " * 30).encode(), "text/plain", u))
    assert ld.kind == "web"
    from reportlab.pdfgen import canvas
    b = io.BytesIO(); c = canvas.Canvas(b)
    for i, line in enumerate(["Photosynthesis is the process by which plants make glucose from sunlight."] * 6):
        c.drawString(60, 800 - 15 * i, line)
    c.save()
    ld = sources.load_url("https://x.org/paper.pdf", fetch=lambda u: (b.getvalue(), "application/pdf", u))
    assert ld.kind == "pdf-link" and "Photosynthesis" in "".join(ld.pages)
    print("  links ok (private/local/ftp/file blocked; html, text, pdf routed)")

def test_youtube():
    assert sources.youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=5") == "dQw4w9WgXcQ"
    assert sources.youtube_id("https://youtu.be/dQw4w9WgXcQ?si=abc") == "dQw4w9WgXcQ"
    assert sources.youtube_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert sources.youtube_id("https://www.youtube.com/") is None
    words = ("photosynthesis is how plants make food from sunlight and the chloroplast holds chlorophyll "
             "which absorbs light energy to build glucose").split()
    segs, t = [], 0.0
    for i in range(0, 80, 4):
        segs.append({"text": " ".join((words * 10)[i:i + 4]) + ("\n" if i % 8 == 0 else ""), "start": t, "duration": 1.6}); t += 1.7 if (i // 4) % 3 else 3.8   # a pause every third caption line
    segs.insert(0, {"text": "[Music]", "start": 0, "duration": 1})
    ld = sources.load_youtube("https://youtu.be/dQw4w9WgXcQ", fetcher=lambda v: segs, titler=lambda u: "Photosynthesis explained")
    text = "\n".join(ld.pages)
    assert ld.title == "Photosynthesis explained" and ld.kind == "youtube" and "[Music]" not in text
    assert text.count(".") >= 5, "no sentences were made from the captions"
    print("  youtube ok:", text.count("."), "sentences from", len(segs), "caption lines")

def test_library():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["CC_LIBRARY_DIR"] = tmp
        title, pages = sources.html_to_pages(WIKI)
        p = Params(max_concepts=0)
        res = run_pipeline(pages, p)
        did = library.doc_id("web", "https://x/photosynthesis", pages)
        assert did == library.doc_id("web", "https://x/photosynthesis", pages)
        meta = library.save({"id": did, "title": title, "kind": "web", "source": "https://x/photosynthesis"}, pages, p, res)
        assert [m["id"] for m in library.list_docs()] == [did] and meta["n_concepts"] == res.stats["concepts"]
        m2, pg2, p2, r2 = library.load(did)
        assert r2 is not None and len(r2.store.concepts) == len(res.store.concepts) and pg2 == pages and p2 == p
        # saving again updates the same entry and keeps the creation date
        library.save({"id": did, "title": "Renamed", "kind": "web", "source": "https://x/photosynthesis"}, pages, p, res)
        docs = library.list_docs(); assert len(docs) == 1 and docs[0]["created"] == meta["created"]
        library.rename(did, "My biology notes"); assert library.list_docs()[0]["title"] == "My biology notes"
        # a damaged saved map must not break opening: the pages remain, so the map is rebuilt
        (library.library_dir() / did / "result.pkl").write_bytes(b"garbage")
        m3, pg3, p3, r3 = library.load(did); assert r3 is None and pg3 == pages
        library.delete(did); assert library.list_docs() == []
        print("  library ok: save / list / load / update / rename / damaged-file fallback / delete")


def test_link_sites():
    # share links are turned into direct downloads
    assert sources.rewrite_known_link("https://docs.google.com/presentation/d/AbC_123-x/edit#slide=id.p")[0] == \
        "https://docs.google.com/presentation/d/AbC_123-x/export/pptx"
    assert sources.rewrite_known_link("https://docs.google.com/document/d/AbC_123-x/edit")[0].endswith("/export?format=docx")
    assert sources.rewrite_known_link("https://drive.google.com/file/d/FILEID123/view?usp=sharing")[0] == \
        "https://drive.google.com/uc?export=download&id=FILEID123"
    assert sources.rewrite_known_link("https://example.org/a")[0] == "https://example.org/a"
    # sites that never give their text to programs: a clear explanation, not a vague error
    for url, word in [("https://canva.link/kgualxosn1quyyu", "Canva"),
                      ("https://www.google.co.in/books/edition/Multilingual_Natural_Language_Processing/-eHztm8JRxwC?printsec=frontcover".replace("www.google.co.in/books", "books.google.co.in/books"), "Google Books"),
                      ("https://1drv.ms/p/abc", "OneDrive")]:
        try:
            sources.load_url(url); raise SystemExit("not explained: " + url)
        except ValueError as exc:
            assert word in str(exc), (url, str(exc))
    # Google Slides link -> real pptx download (fetch mocked), no file extension needed
    pptx = make_pptx([("Cell Biology", ["The cell is the basic unit of life", "A cell membrane controls the cell"], ""),
                      ("Photosynthesis", ["Plants make glucose from sunlight", "Chlorophyll absorbs sunlight"], "")])
    seen = {}
    def fetch_pptx(u): seen["u"] = u; return pptx, "application/octet-stream", u
    ld = sources.load_url("https://docs.google.com/presentation/d/AbC_123-x/edit", fetch=fetch_pptx)
    assert seen["u"].endswith("/export/pptx") and ld.kind == "pptx" and "Google Slides" in ld.note and len(ld.pages) == 2
    # a private share link returns a login page: say so
    login = b"<html><head><title>Sign in</title></head><body><p>Sign in to continue</p></body></html>"
    try:
        sources.load_url("https://docs.google.com/presentation/d/AbC_123-x/edit", fetch=lambda u: (login, "text/html", u))
        raise SystemExit("private link accepted")
    except ValueError as exc:
        assert "Anyone with the link" in str(exc), str(exc)
    # office files at links without an extension are recognised by their content
    ld = sources.load_url("https://x.org/download?id=7", fetch=lambda u: (make_docx([("Heading1", "Databases"),
        (None, "A database stores data in tables with rows and columns."), (None, "A key identifies each row of a table.")]), "application/octet-stream", u))
    assert ld.kind == "docx"
    print("  link sites ok (google links rewritten, canva / books / onedrive explained, private link explained)")

def _fake_youtube(kind, transcripts, raise_in_list=None):
    import types
    class Snip:
        def __init__(s, text, start, duration): s.text, s.start, s.duration = text, start, duration
    class T:
        def __init__(s, d): s.d = d; s.language_code = d["code"]; s.is_generated = d["gen"]; s.is_translatable = d.get("tr", False)
        def translate(s, code):
            if not s.is_translatable: raise Exception("cannot translate")
            return T({**s.d, "code": code, "gen": s.d["gen"], "tr": False, "translated": True})
        def fetch(s):
            segs = s.d["segs"]
            return [Snip(**x) for x in segs] if kind == "v1" else segs
    def lst(*a):
        if raise_in_list: raise raise_in_list
        return [T(d) for d in transcripts]
    attrs = {"list": lambda self, vid: lst()} if kind == "v1" else {"list_transcripts": staticmethod(lambda vid: lst())}
    API = type("YouTubeTranscriptApi", (), attrs)
    sys.modules["youtube_transcript_api"] = types.SimpleNamespace(YouTubeTranscriptApi=API)

def _segs(word):
    return [{"text": f"{word} is how plants make food from sunlight number {i}", "start": i * 5.0, "duration": 4.0} for i in range(30)]

def test_youtube_api_shapes():
    for kind in ("v1", "old"):
        # manual English beats auto-generated English
        _fake_youtube(kind, [{"code": "en", "gen": True, "segs": _segs("auto")}, {"code": "en-GB", "gen": False, "segs": _segs("manual")}])
        segs, note = sources._fetch_transcript("dQw4w9WgXcQ")
        assert segs[0]["text"].startswith("manual") and note == "", (kind, segs[0], note)
        # only auto-generated English
        _fake_youtube(kind, [{"code": "en", "gen": True, "segs": _segs("auto")}])
        assert sources._fetch_transcript("dQw4w9WgXcQ")[0][0]["text"].startswith("auto")
        # only Hindi, translatable -> translated to English with a note (this was the user's failure)
        _fake_youtube(kind, [{"code": "hi", "gen": True, "tr": True, "segs": _segs("hindi")}])
        segs, note = sources._fetch_transcript("dQw4w9WgXcQ")
        assert "'hi'" in note and "translated" in note and len(segs) == 30, note
        # only Hindi, not translatable -> clear message naming the language
        _fake_youtube(kind, [{"code": "hi", "gen": False, "segs": _segs("hindi")}])
        try:
            sources._fetch_transcript("dQw4w9WgXcQ"); raise SystemExit("should fail")
        except RuntimeError as exc:
            assert "hi" in str(exc) and "English" in str(exc), str(exc)
    # YouTube's own error types become plain sentences
    class TranscriptsDisabled(Exception): pass
    _fake_youtube("v1", [], raise_in_list=TranscriptsDisabled("x"))
    try:
        sources._fetch_transcript("dQw4w9WgXcQ"); raise SystemExit("should fail")
    except RuntimeError as exc:
        assert "turned off" in str(exc)
    # whole path through load_youtube with a translated transcript
    _fake_youtube("v1", [{"code": "hi", "gen": True, "tr": True, "segs": _segs("hindi")}])
    ld = sources.load_youtube("https://youtu.be/dQw4w9WgXcQ", titler=lambda u: "Python course")
    assert ld.kind == "youtube" and "translated" in ld.note and ld.title == "Python course"
    sys.modules.pop("youtube_transcript_api", None)
    print("  youtube captions ok (manual > auto > translated; both library versions; clear errors)")

if __name__ == "__main__":
    for t in (test_pptx, test_docx, test_html, test_pipeline_on_loaded, test_links, test_link_sites, test_youtube, test_youtube_api_shapes, test_library):
        t()
    print("ALL SOURCE CHECKS PASSED")

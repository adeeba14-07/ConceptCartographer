"""The real app code: save to library, list, open later, update, delete, and the new input sources.
python tests/test_library_flow.py"""
import io, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "run_app_with_stub.py")).read().split("APP = sys.argv")[0])
sys.path.insert(0, ROOT)
from ccart import library, sources
import tests.test_sources as ts  # fixtures: make_pptx, make_docx, WIKI

code = open(os.path.join(ROOT, "app.py")).read()

def run_once(press=False, find=""):
    RUN_BUTTON["value"], FIND["v"] = press, find
    try:
        exec(compile(code, "app.py", "exec"), {"__name__": "__main__"}); return "finished"
    except StopApp:
        return "rerun"

def click(key):
    fn, args = CALLBACKS[key]; fn(*args)

BIO = ("The cell is the basic unit of life. Every cell is surrounded by a cell membrane. The cell membrane controls "
       "what enters the cell. Mitochondria produce energy for the cell. Mitochondria convert glucose into ATP. "
       "ATP is the energy currency of the cell. The nucleus stores the DNA of the cell. DNA carries instructions "
       "for proteins. Ribosomes build proteins from DNA instructions. Photosynthesis makes glucose from sunlight. "
       "Chloroplasts contain chlorophyll. Chlorophyll absorbs sunlight in the chloroplasts.")

# 1) sample chapter -> saved automatically
assert run_once(True) == "finished"
docs = library.list_docs(); assert len(docs) == 1 and docs[0]["kind"] == "sample", docs
id_sample = docs[0]["id"]
print("1 built sample, saved:", docs[0]["title"], "|", docs[0]["n_concepts"], "concepts")

# 2) pasted text -> second entry
RADIO["v"] = "Paste text"; TEXTAREA["Paste your text"] = BIO
assert run_once(True) == "finished"
docs = library.list_docs(); assert len(docs) == 2 and docs[0]["kind"] == "paste", docs
id_paste = docs[0]["id"]; print("2 pasted text saved ->", len(docs), "documents in the library")

# 3) come back later: open the sample from the library
RADIO["v"] = None
run_once(False); assert f"lib_open_{id_sample}" in CALLBACKS
click(f"lib_open_{id_sample}"); run_once(False)
cur = st.session_state["current"]; r = st.session_state["result"]
assert cur["id"] == id_sample and r.stats["concepts"] == docs[1]["n_concepts"], (cur, r.stats)
print("3 reopened the sample from the library instantly (no rebuild):", r.stats["concepts"], "concepts")

# 4) a concept added by search is saved back into the library entry
import ccart.pipeline as pl
before = r.stats["concepts"]
out = run_once(False, "kneser-ney smoothing"); run_once(False, "kneser-ney smoothing")
saved = library.load(id_sample)[3]
assert any("Kneser" in c.name for c in saved.store.concepts.values()), "added concept not saved"
print("4 concept added by search was saved to the library entry")

# 5) delete needs two clicks
click(f"lib_del_{id_paste}"); run_once(False); assert len(library.list_docs()) == 2
click(f"lib_del_{id_paste}"); run_once(False); assert len(library.list_docs()) == 1
print("5 delete asks for a second click, then removes the document")

# 6) a web link (fetch is mocked) and a YouTube link
RADIO["v"] = "Web link or YouTube"; TEXTVAL["Paste a link"] = "https://en.wikipedia.org/wiki/Photosynthesis"
sources_load = sources.load_url
sources.load_url = lambda u: sources.load_url_orig(u) if False else sources.Loaded(*(lambda t, p: (t, p, "web", u))(*sources.html_to_pages(ts.WIKI)))
assert run_once(True) == "finished"
d = [m for m in library.list_docs() if m["kind"] == "web"]; assert d and d[0]["title"] == "Photosynthesis"
print("6 web link built and saved:", d[0]["title"])
sources.load_url = lambda u: sources.load_youtube(u, fetcher=lambda v: [{"text": f"photosynthesis lets plants make glucose from sunlight number {i}", "start": i * 5.0, "duration": 4.0} for i in range(40)], titler=lambda u: "Photosynthesis video")
TEXTVAL["Paste a link"] = "https://youtu.be/dQw4w9WgXcQ"
assert run_once(True) == "finished"
assert any(m["kind"] == "youtube" for m in library.list_docs()); print("  youtube link built and saved")
sources.load_url = sources_load

# 7) PowerPoint and Word uploads
class Up:
    def __init__(s, name, data): s.name, s._d = name, data
    def getvalue(s): return s._d
RADIO["v"] = "Upload file"
slides = [("Cell Biology", ["The cell is the basic unit of life", "A cell membrane controls what enters the cell"], "The cell membrane protects the cell."),
          ("Photosynthesis", ["Plants make glucose from sunlight", "Chlorophyll absorbs sunlight in the chloroplast"], ""),
          ("Mitochondria", ["Mitochondria produce ATP for the cell", "ATP is the energy currency of the cell"], "")]
UPLOAD["v"] = Up("biology.pptx", ts.make_pptx(slides))
assert run_once(True) == "finished"
assert any(m["kind"] == "pptx" for m in library.list_docs()); print("7 PowerPoint uploaded, built and saved")
UPLOAD["v"] = Up("notes.docx", ts.make_docx([("Heading1", "Relational Databases"),
    (None, "A relational database stores data in tables with rows and columns."),
    (None, "A primary key identifies each row of a table in the database."),
    (None, "A foreign key links one table of the database to another table."),
    ("Heading2", "Indexes"), (None, "An index speeds up queries on a table in the database."),
    (None, "A database builds an index on a column of the table."),
    ("Heading1", "Transactions"), (None, "A transaction is a group of database operations that succeed or fail together."),
    (None, "A transaction keeps the database consistent after every operation.")]))
assert run_once(True) == "finished"
assert any(m["kind"] == "docx" for m in library.list_docs()); print("  Word file uploaded, built and saved")
print("LIBRARY FLOW OK -", len(library.list_docs()), "documents saved")

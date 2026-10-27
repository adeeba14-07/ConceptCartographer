"""Executes app.py end-to-end against a stub 'streamlit' module (no browser needed).
This checks that every tab's code path runs without exceptions."""
import os, sys, types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import tempfile
os.environ.setdefault("CC_LIBRARY_DIR", tempfile.mkdtemp())  # tests never touch a real library
sys.path.insert(0, ROOT)
os.chdir(ROOT)

class StopApp(Exception):
    pass

st = types.ModuleType("streamlit")
calls = {}
def _count(name):
    calls[name] = calls.get(name, 0) + 1

class Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name):
        return getattr(st, name)

class Bar:
    def progress(self, *a, **k): pass
    def empty(self): pass

class SS(dict):
    pass

st.session_state = SS()
RUN_BUTTON = {"value": os.environ.get("RUN","1")=="1"}

def _simple(name):
    def f(*a, **k):
        _count(name)
        return None
    return f

for n in ["set_page_config","markdown","header","caption","info","warning","error","success","divider",
          "metric","subheader","write","code","json","dataframe","bar_chart","line_chart","download_button","text","title"]:
    setattr(st, n, _simple(n))

RADIO, TEXTVAL, TEXTAREA, UPLOAD, CALLBACKS = {"v": None}, {}, {}, {"v": None}, {}
def radio(label, options, index=0, **k): return RADIO["v"] or options[index]
def button(label, **k):
    if k.get("on_click"):
        CALLBACKS[k.get("key")] = (k["on_click"], k.get("args", ()))
    return RUN_BUTTON["value"] and not k.get("disabled", False)
def slider(label, mn=None, mx=None, value=None, step=None, **k): return value if value is not None else mn
def checkbox(label, value=False, **k): return value
def text_area(label, value="", **k): return TEXTAREA.get(label, value)
FIND = {"v": ""}
def text_input(label, value="", **k):
    return FIND["v"] if k.get("key") == "s_find" else TEXTVAL.get(label, value)
def number_input(label, mn=None, mx=None, value=None, **k): return value if value is not None else mn
def selectbox(label, options, index=0, **k):
    opts = list(options); return opts[index]
def multiselect(label, options, default=None, **k): return list(default or [])
def file_uploader(*a, **k): return UPLOAD["v"]
def columns(spec, **k):
    n = spec if isinstance(spec, int) else len(spec)
    return [Ctx() for _ in range(n)]
def tabs(labels): return [Ctx() for _ in labels]
def expander(*a, **k): return Ctx()
def spinner(*a, **k): return Ctx()
def progress(*a, **k): return Bar()
def stop(): raise StopApp()
def rerun(): raise StopApp()
def cache_resource(*a, **k):
    if a and callable(a[0]): return a[0]
    return lambda f: f
cache_data = cache_resource
st.cache_data = cache_resource
st.rerun = rerun

for fn in [radio, button, slider, checkbox, text_area, text_input, number_input, selectbox, multiselect,
           file_uploader, columns, tabs, expander, spinner, progress, stop, cache_resource, cache_data]:
    setattr(st, fn.__name__, fn)
st.sidebar = Ctx()

comp = types.ModuleType("streamlit.components"); v1 = types.ModuleType("streamlit.components.v1")
def html(h, height=None, scrolling=False): _count("components.html"); assert "<" in h
v1.html = html; comp.v1 = v1; st.components = comp
sys.modules.update({"streamlit": st, "streamlit.components": comp, "streamlit.components.v1": v1})

APP = sys.argv[1] if len(sys.argv) > 1 else "app.py"
src = open(os.path.join(ROOT, APP)).read()

def run_once(press, find):
    RUN_BUTTON["value"] = press
    FIND["v"] = find
    try:
        exec(compile(src, APP, "exec"), {"__name__": "__main__"})
        return "finished"
    except StopApp:
        return "stopped/rerun"

if os.environ.get("SEQ"):
    # search flow on the simple app: build -> focus search -> auto-add unknown term -> rerun
    print("1 build         :", run_once(True, ""))
    r = st.session_state["result"]
    print("  concepts:", len(r.store.concepts))
    print("2 search known  :", run_once(False, "smoothing"))
    unknown = "kneser-ney smoothing"
    before = [c.name for c in r.store.concepts.values() if "kneser" in c.name.lower()]
    print("3 search unknown:", run_once(False, unknown), "| before:", before)
    r2 = st.session_state["result"]
    after = [c.name for c in r2.store.concepts.values() if "kneser" in c.name.lower()]
    print("  after add:", after)
    print("4 search again  :", run_once(False, unknown))
    print("5 search missing:", run_once(False, "zzzqqq"))
    print("6 closest match :", run_once(False, "language model evaluation perplexity scores"))
    assert not before and after, "auto-add failed"
    print("SEQ OK")
else:
    print(run_once(RUN_BUTTON["value"], ""))
    print("calls:", calls)
    print("result stored:", st.session_state.get("result") is not None)

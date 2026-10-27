"""A small on-disk library: every document you build is saved and can be reopened later.

Layout (one folder per document, inside `library/` next to the app or in $CC_LIBRARY_DIR):
    meta.json    title, kind, source, dates
    pages.json   the extracted text pages (so the map can always be rebuilt)
    params.json  the settings used
    result.pkl   the finished map (opens instantly; ignored and rebuilt if it cannot be read)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .pipeline import Params, PipelineResult

SCHEMA = 3


def library_dir() -> Path:
    base = os.environ.get("CC_LIBRARY_DIR")
    path = Path(base) if base else Path(__file__).resolve().parent.parent / "library"
    path.mkdir(parents=True, exist_ok=True)
    return path


def doc_id(kind: str, source: str, pages: List[str]) -> str:
    """Stable id: the same document opened again updates its entry instead of adding a duplicate."""
    h = hashlib.sha1()
    h.update(kind.encode())
    h.update(source.encode("utf-8", "ignore"))
    h.update(str(len(pages)).encode())
    h.update("".join(pages)[:4000].encode("utf-8", "ignore"))
    return h.hexdigest()[:12]


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save(meta: Dict[str, Any], pages: List[str], params: Params, result: Optional[PipelineResult]) -> Dict[str, Any]:
    """Create or update a library entry. `meta` needs: id, title, kind, source (+ optional first_page)."""
    d = library_dir() / meta["id"]
    d.mkdir(parents=True, exist_ok=True)
    now = time.time()
    old = {}
    if (d / "meta.json").exists():
        try:
            old = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            old = {}
    meta = {**old, **meta, "created": old.get("created", now), "updated": now, "schema": SCHEMA,
            "n_pages": len(pages), "n_concepts": result.stats["concepts"] if result else old.get("n_concepts", 0)}
    _atomic_write(d / "pages.json", json.dumps(pages, ensure_ascii=False).encode("utf-8"))
    _atomic_write(d / "params.json", json.dumps(dataclasses.asdict(params), ensure_ascii=False).encode("utf-8"))
    if result is not None:
        _atomic_write(d / "result.pkl", pickle.dumps(result, protocol=4))
    _atomic_write(d / "meta.json", json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))
    return meta


def list_docs() -> List[Dict[str, Any]]:
    out = []
    for d in library_dir().iterdir():
        mp = d / "meta.json"
        if d.is_dir() and mp.exists():
            try:
                out.append(json.loads(mp.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 - skip a damaged entry
                continue
    return sorted(out, key=lambda m: m.get("updated", 0), reverse=True)


def load(doc_id_: str) -> Tuple[Dict[str, Any], List[str], Params, Optional[PipelineResult]]:
    """-> (meta, pages, params, result or None). A missing/unreadable result means: rebuild from the pages."""
    d = library_dir() / doc_id_
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    pages = json.loads((d / "pages.json").read_text(encoding="utf-8"))
    raw = json.loads((d / "params.json").read_text(encoding="utf-8"))
    fields = {f.name for f in dataclasses.fields(Params)}
    params = Params(**{k: v for k, v in raw.items() if k in fields})
    result = None
    rp = d / "result.pkl"
    if rp.exists() and meta.get("schema") == SCHEMA:
        try:
            result = pickle.loads(rp.read_bytes())
            if not isinstance(result, PipelineResult):
                result = None
        except Exception:  # noqa: BLE001 - e.g. saved by an older version of the app
            result = None
    return meta, pages, params, result


def delete(doc_id_: str) -> None:
    d = library_dir() / doc_id_
    if d.is_dir() and d.parent == library_dir():
        shutil.rmtree(d, ignore_errors=True)


def rename(doc_id_: str, title: str) -> None:
    mp = library_dir() / doc_id_ / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
        meta["title"] = title.strip() or meta.get("title", "Untitled")
        _atomic_write(mp, json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from warpdb.api.models import SearchRequest, UpsertRequest
from warpdb.db import WarpDB

# ---------- App + DB lifecycle ----------

app = FastAPI(title="warpdb", version="0.1.0")

_db: Optional[WarpDB] = None

def get_db() -> WarpDB:
    global _db
    dim = 768 # TODO: Configurable dimension
    if _db is None:
        _db = WarpDB(dim)
    return _db

# ---------- Routes ----------

@app.get("/health")
def health() -> Dict[str, Any]:
    db = get_db()
    return {
        "ok": True,
        "count": db.count(),
    }

@app.delete("/vectors/{name}")
def delete_vector(name: str) -> Dict[str, Any]:
    db = get_db()
    try:
        db.delete(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "ok": True,
        "count": db.count(),
    }

@app.put("/vectors/{name}")
def upsert(name: str, req: UpsertRequest) -> Dict[str, Any]:
    db = get_db()
    try:
        db.upsert(name, req.vector, req.metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "ok": True,
        "count": db.count(),
    }

@app.post("/search")
def search(req: SearchRequest) -> List[Dict[str, Any]]:
    db = get_db()
    results = db.search(req.vector, req.k)
    return [{"distance": dist, "name": name} for dist, name in results]

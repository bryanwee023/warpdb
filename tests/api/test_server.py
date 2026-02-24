import random

import pytest
from fastapi.testclient import TestClient

import warpdb.api.server as server_module
from warpdb.api.server import app
from warpdb.db import WarpDB

DIM = 4

@pytest.fixture
def client(tmp_path, monkeypatch):
    # Inject a fresh, isolated WarpDB so tests don't share state
    # and files land in a temp directory instead of cwd.
    monkeypatch.setattr(server_module, "_db", WarpDB(dim=DIM, data_dir=str(tmp_path)))
    return TestClient(app)

def vec(values):
    return list(float(x) for x in values)

# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    res = client.get("/health")
    assert res.status_code == 200

def test_health_body(client):
    res = client.get("/health")
    assert res.json() == {"ok": True, "count": 0}

def test_health_count_reflects_upserts(client):
    random.seed(0)
    client.post("/upsert", json={"id": "a", "vector": vec([1, 0, 0, 0])})
    client.post("/upsert", json={"id": "b", "vector": vec([0, 1, 0, 0])})
    res = client.get("/health")
    assert res.json()["count"] == 2

# ---------------------------------------------------------------------------
# POST /upsert
# ---------------------------------------------------------------------------

def test_upsert_returns_200(client):
    random.seed(0)
    res = client.post("/upsert", json={"id": "a", "vector": vec([1, 0, 0, 0])})
    assert res.status_code == 200

def test_upsert_response_body(client):
    random.seed(0)
    res = client.post("/upsert", json={"id": "a", "vector": vec([1, 0, 0, 0])})
    assert res.json() == {"ok": True, "count": 1}

def test_upsert_count_increments_in_response(client):
    random.seed(0)
    res1 = client.post("/upsert", json={"id": "a", "vector": vec([1, 0, 0, 0])})
    res2 = client.post("/upsert", json={"id": "b", "vector": vec([0, 1, 0, 0])})
    assert res1.json()["count"] == 1
    assert res2.json()["count"] == 2

def test_upsert_with_metadata(client):
    random.seed(0)
    res = client.post("/upsert", json={
        "id": "a",
        "vector": vec([1, 0, 0, 0]),
        "metadata": {"color": "red"},
    })
    assert res.status_code == 200

def test_upsert_without_metadata(client):
    random.seed(0)
    res = client.post("/upsert", json={"id": "a", "vector": vec([1, 0, 0, 0])})
    assert res.status_code == 200

def test_upsert_duplicate_id_returns_400(client):
    random.seed(0)
    client.post("/upsert", json={"id": "a", "vector": vec([1, 0, 0, 0])})
    res = client.post("/upsert", json={"id": "a", "vector": vec([0, 1, 0, 0])})
    assert res.status_code == 400
    assert "'a' already exists" in res.json()["detail"]

def test_upsert_missing_id_returns_422(client):
    res = client.post("/upsert", json={"vector": vec([1, 0, 0, 0])})
    assert res.status_code == 422

def test_upsert_missing_vector_returns_422(client):
    res = client.post("/upsert", json={"id": "a"})
    assert res.status_code == 422

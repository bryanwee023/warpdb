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
    client.put("/vectors/a", json={"vector": vec([1, 0, 0, 0])})
    client.put("/vectors/b", json={"vector": vec([0, 1, 0, 0])})
    res = client.get("/health")
    assert res.json()["count"] == 2

# ---------------------------------------------------------------------------
# PUT /vectors/{id}
# ---------------------------------------------------------------------------

def test_upsert_returns_200(client):
    random.seed(0)
    res = client.put("/vectors/a", json={"vector": vec([1, 0, 0, 0])})
    assert res.status_code == 200

def test_upsert_response_body(client):
    random.seed(0)
    res = client.put("/vectors/a", json={"vector": vec([1, 0, 0, 0])})
    assert res.json() == {"ok": True, "count": 1}

def test_upsert_count_increments_in_response(client):
    random.seed(0)
    res1 = client.put("/vectors/a", json={"vector": vec([1, 0, 0, 0])})
    res2 = client.put("/vectors/b", json={"vector": vec([0, 1, 0, 0])})
    assert res1.json()["count"] == 1
    assert res2.json()["count"] == 2

def test_upsert_with_metadata(client):
    random.seed(0)
    res = client.put("/vectors/a", json={
        "vector": vec([1, 0, 0, 0]),
        "metadata": {"color": "red"},
    })
    assert res.status_code == 200

def test_upsert_without_metadata(client):
    random.seed(0)
    res = client.put("/vectors/a", json={"vector": vec([1, 0, 0, 0])})
    assert res.status_code == 200

def test_upsert_duplicate_id_returns_400(client):
    random.seed(0)
    client.put("/vectors/a", json={"vector": vec([1, 0, 0, 0])})
    res = client.put("/vectors/a", json={"vector": vec([0, 1, 0, 0])})
    assert res.status_code == 400
    assert "'a' already exists" in res.json()["detail"]

def test_upsert_missing_vector_returns_422(client):
    res = client.put("/vectors/a", json={})
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /vectors/{id}
# ---------------------------------------------------------------------------

def test_delete_returns_200(client):
    random.seed(0)
    client.put("/vectors/a", json={"vector": vec([1, 0, 0, 0])})
    res = client.delete("/vectors/a")
    assert res.status_code == 200

def test_delete_response_body(client):
    random.seed(0)
    client.put("/vectors/a", json={"vector": vec([1, 0, 0, 0])})
    res = client.delete("/vectors/a")
    assert res.json() == {"ok": True, "count": 0}

def test_delete_decrements_health_count(client):
    random.seed(0)
    client.put("/vectors/a", json={"vector": vec([1, 0, 0, 0])})
    client.put("/vectors/b", json={"vector": vec([0, 1, 0, 0])})
    client.delete("/vectors/a")
    res = client.get("/health")
    assert res.json()["count"] == 1

def test_delete_nonexistent_returns_404(client):
    res = client.delete("/vectors/ghost")
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------

def test_search_returns_200(client):
    res = client.post("/search", json={"vector": vec([1, 0, 0, 0])})
    assert res.status_code == 200

def test_search_empty_db_returns_empty(client):
    res = client.post("/search", json={"vector": vec([1, 0, 0, 0])})
    assert res.json() == []

def test_search_returns_nearest(client):
    random.seed(42)
    client.put("/vectors/a", json={"vector": vec([1, 0, 0, 0])})
    client.put("/vectors/b", json={"vector": vec([0, 1, 0, 0])})
    res = client.post("/search", json={"vector": vec([1, 0, 0, 0]), "k": 1})
    results = res.json()
    assert len(results) == 1
    assert results[0]["name"] == "a"

def test_search_result_has_distance_and_name(client):
    random.seed(0)
    client.put("/vectors/a", json={"vector": vec([1, 0, 0, 0])})
    res = client.post("/search", json={"vector": vec([1, 0, 0, 0]), "k": 1})
    result = res.json()[0]
    assert "distance" in result
    assert "name" in result

def test_search_respects_k(client):
    random.seed(0)
    for i in range(5):
        client.put(f"/vectors/{i}", json={"vector": vec([float(i), 0, 0, 0])})
    res = client.post("/search", json={"vector": vec([0, 0, 0, 0]), "k": 2})
    assert len(res.json()) <= 2

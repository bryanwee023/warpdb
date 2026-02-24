import random

import pytest

from warpdb.db import WarpDB

DIM = 4

@pytest.fixture
def db(tmp_path):
    return WarpDB(dim=DIM, data_dir=str(tmp_path))

# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count_is_zero_initially(db):
    assert db.count() == 0

def test_count_increments_after_upsert(db):
    random.seed(0)
    db.upsert("a", [1, 0, 0, 0])
    assert db.count() == 1
    db.upsert("b", [0, 1, 0, 0])
    assert db.count() == 2

# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------

def test_upsert_accepts_metadata(db):
    random.seed(0)
    db.upsert("a", [1, 0, 0, 0], metadata={"color": "red"})
    assert db.count() == 1

def test_upsert_accepts_no_metadata(db):
    random.seed(0)
    db.upsert("a", [1, 0, 0, 0])
    assert db.count() == 1

def test_upsert_duplicate_id_raises(db):
    random.seed(0)
    db.upsert("a", [1, 0, 0, 0])
    with pytest.raises(ValueError, match="'a' already exists"):
        db.upsert("a", [0, 1, 0, 0])

def test_upsert_different_ids_are_independent(db):
    random.seed(0)
    db.upsert("a", [1, 0, 0, 0])
    db.upsert("b", [0, 1, 0, 0])
    assert db.count() == 2

# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_empty_db_returns_empty(db):
    assert db.search([1, 0, 0, 0], k=5) == []

def test_search_returns_string_ids(db):
    random.seed(0)
    db.upsert("my-id", [1, 0, 0, 0])
    results = db.search([1, 0, 0, 0], k=1)
    assert len(results) == 1
    dist, id = results[0]
    assert isinstance(id, str)
    assert id == "my-id"

def test_search_exact_match_has_zero_distance(db):
    random.seed(0)
    db.upsert("a", [1, 2, 3, 4])
    results = db.search([1, 2, 3, 4], k=1)
    assert results[0][0] == pytest.approx(0.0)

def test_search_returns_at_most_k_results(db):
    random.seed(0)
    for i in range(10):
        db.upsert(str(i), [float(i), 0, 0, 0])
    results = db.search([0, 0, 0, 0], k=3)
    assert len(results) <= 3

def test_search_returns_fewer_than_k_when_db_is_small(db):
    random.seed(0)
    db.upsert("a", [1, 0, 0, 0])
    db.upsert("b", [0, 1, 0, 0])
    results = db.search([0, 0, 0, 0], k=10)
    assert len(results) <= 2

def test_search_results_sorted_ascending_by_distance(db):
    random.seed(0)
    for i in range(10):
        db.upsert(str(i), [float(i) * 0.1, 0, 0, 0])
    results = db.search([0, 0, 0, 0], k=10)
    dists = [d for d, _ in results]
    assert dists == sorted(dists)

# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_delete_decrements_count(db):
    random.seed(0)
    db.upsert("a", [1, 0, 0, 0])
    assert db.count() == 1
    db.delete("a")
    assert db.count() == 0

def test_delete_removes_from_search(db):
    random.seed(0)
    db.upsert("a", [1, 0, 0, 0])
    db.delete("a")
    results = db.search([1, 0, 0, 0], k=5)
    ids = [id for _, id in results]
    assert "a" not in ids

def test_delete_nonexistent_raises(db):
    with pytest.raises(ValueError, match="not found"):
        db.delete("ghost")

def test_delete_already_deleted_raises(db):
    random.seed(0)
    db.upsert("a", [1, 0, 0, 0])
    db.delete("a")
    with pytest.raises(ValueError, match="not found"):
        db.delete("a")

def test_upsert_same_id_after_delete_succeeds(db):
    random.seed(0)
    db.upsert("a", [1, 0, 0, 0])
    db.delete("a")
    db.upsert("a", [0, 1, 0, 0])  # should not raise
    assert db.count() == 1

def test_recovery_crash_after_delete_intent(tmp_path):
    """Simulate: DELETE WAL record written, metadata not yet updated. Recovery should delete."""
    import random as _random
    _random.seed(0)

    db1 = WarpDB(dim=DIM, data_dir=str(tmp_path))
    db1.upsert("a", [1, 0, 0, 0])
    # Simulate crash after log_delete but before metadata.delete by manipulating WAL directly
    from warpdb.storage.wal import WAL
    wal = WAL(str(tmp_path / "wal.bin"), DIM)
    wal.log_delete("a")
    # Do NOT commit or update metadata — simulates crash mid-delete

    # Restart
    _random.seed(0)
    db2 = WarpDB(dim=DIM, data_dir=str(tmp_path))
    assert db2.count() == 0
    results = db2.search([1, 0, 0, 0], k=5)
    assert all(id != "a" for _, id in results)


def test_search_nearest_neighbor_clearly_separated(db):
    """Vectors on scaled basis axes are trivially separable."""
    random.seed(42)
    ids = ["x", "y", "z", "w"]
    for i, name in enumerate(ids):
        v = [0.0] * DIM
        v[i] = 100.0
        db.upsert(name, v)

    query = [0.0] * DIM
    query[2] = 100.0  # nearest to "z"

    results = db.search(query, k=1)
    assert results[0][1] == "z"

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
    wal = WAL(str(tmp_path / "data" / "wal.bin"), DIM)
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


# ---------------------------------------------------------------------------
# compact
# ---------------------------------------------------------------------------

def test_compact_reduces_vector_count(db):
    random.seed(0)
    for i in range(5):
        db.upsert(str(i), [float(i), 0, 0, 0])
    # Delete 1 of 5 (20% dead — below 25% threshold, no auto-compact)
    db.delete("0")
    assert db._vector_store.count() == 5
    db.compact()
    assert db._vector_store.count() == 4

def test_search_works_after_compact(db):
    random.seed(0)
    db.upsert("a", [1, 0, 0, 0])
    db.upsert("b", [0, 1, 0, 0])
    db.upsert("c", [0, 0, 1, 0])
    db.delete("b")
    db.compact()

    results = db.search([1, 0, 0, 0], k=1)
    assert results[0][1] == "a"

def test_compact_is_idempotent(db):
    random.seed(0)
    db.upsert("a", [1, 0, 0, 0])
    db.compact()
    db.compact()  # should not raise
    assert db.count() == 1

def test_delete_does_not_auto_compact(tmp_path):
    """Deletes should not trigger compaction automatically."""
    random.seed(0)
    db = WarpDB(dim=DIM, data_dir=str(tmp_path))
    for i in range(4):
        db.upsert(str(i), [float(i), 0, 0, 0])
    assert db._vector_store.count() == 4

    db.delete("0")
    db.delete("1")  # 50% dead — no auto-compact
    assert db._vector_store.count() == 4

    db.compact()  # explicit compact still works
    assert db._vector_store.count() == 2

def test_upsert_after_compact_works(db):
    random.seed(0)
    db.upsert("a", [1, 0, 0, 0])
    db.upsert("b", [0, 1, 0, 0])
    db.delete("a")
    db.compact()

    db.upsert("c", [0, 0, 1, 0])
    assert db.count() == 2
    results = db.search([0, 0, 1, 0], k=1)
    assert results[0][1] == "c"


# ---------------------------------------------------------------------------
# compact — crash recovery
# ---------------------------------------------------------------------------

def test_recovery_crash_before_file_swap(tmp_path):
    """COMPACT WAL record written, temp file exists, but os.replace never happened.
    Recovery should discard temp file; data stays at pre-compaction state."""
    import os, shutil

    random.seed(0)
    db1 = WarpDB(dim=DIM, data_dir=str(tmp_path))
    db1.upsert("a", [1, 0, 0, 0])
    db1.upsert("b", [0, 1, 0, 0])
    db1.delete("a")

    # Simulate: COMPACT WAL record (no commit) + temp file exists
    from warpdb.storage.wal import WAL
    data_dir = tmp_path / "data"
    wal = WAL(str(data_dir / "wal.bin"), DIM)
    wal.log_compact()

    tmp_vec_path = str(data_dir / "vectors.compact.f32")
    shutil.copy(str(data_dir / "vectors.f32"), tmp_vec_path)

    # Restart — should discard temp file, keep original vector file
    random.seed(0)
    db2 = WarpDB(dim=DIM, data_dir=str(tmp_path))
    assert not os.path.exists(tmp_vec_path)
    assert db2.count() == 1
    results = db2.search([0, 1, 0, 0], k=1)
    assert results[0][1] == "b"


def test_recovery_crash_after_file_swap(tmp_path):
    """os.replace succeeded but metadata not yet updated.
    Recovery should recompute offsets from MetadataStore."""
    random.seed(0)
    db1 = WarpDB(dim=DIM, data_dir=str(tmp_path))
    db1.upsert("a", [1, 0, 0, 0])
    db1.upsert("b", [0, 1, 0, 0])
    db1.upsert("c", [0, 0, 1, 0])
    db1.delete("a")

    # Manually perform compaction steps up to the crash point:
    live_offsets = db1._metadata_store.iter_offsets()

    # Write COMPACT WAL record (no commit)
    from warpdb.storage.wal import WAL
    wal = WAL(str(tmp_path / "data" / "wal.bin"), DIM)
    wal.log_compact()

    # Rewrite vector file (os.replace happens inside)
    db1._vector_store.compact(live_offsets)

    # "Crash" here — metadata still has OLD offsets, no COMMIT in WAL

    # Restart — recovery should recompute offsets, fix metadata
    random.seed(0)
    db2 = WarpDB(dim=DIM, data_dir=str(tmp_path))
    assert db2.count() == 2
    results = db2.search([0, 1, 0, 0], k=1)
    assert results[0][1] == "b"
    results = db2.search([0, 0, 1, 0], k=1)
    assert results[0][1] == "c"

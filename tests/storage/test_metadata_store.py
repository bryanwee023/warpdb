import sqlite3

import pytest

from warpdb.storage import MetadataStore

@pytest.fixture
def store():
    return MetadataStore(":memory:")

def test_exists_returns_false_for_unknown_name(store):
    assert store.exists("nonexistent") is False

def test_exists_returns_true_after_insert(store):
    store.insert(0, 100, "abc")
    assert store.exists("abc") is True

def test_get_returns_none_for_unknown_id(store):
    assert store.get(99) is None

def test_get_returns_correct_record(store):
    store.insert(7, 200, "abc")
    result = store.get(7)
    assert result == {"id": 7, "file_offset": 200, "name": "abc", "metadata": None}

def test_insert_with_metadata(store):
    meta = {"color": "red", "score": 0.9}
    store.insert(0, 100, "abc", metadata=meta)
    result = store.get(0)
    assert result["metadata"] == meta

def test_insert_without_metadata_returns_none(store):
    store.insert(0, 100, "abc")
    result = store.get(0)
    assert result["metadata"] is None

def test_insert_duplicate_name_raises(store):
    store.insert(0, 100, "abc")
    with pytest.raises(sqlite3.IntegrityError):
        store.insert(1, 200, "abc")

def test_insert_duplicate_id_raises(store):
    store.insert(0, 100, "abc")
    with pytest.raises(sqlite3.IntegrityError):
        store.insert(0, 200, "xyz")

def test_multiple_records_stored_independently(store):
    store.insert(0, 100, "a", metadata={"x": 1})
    store.insert(1, 200, "b", metadata={"x": 2})
    store.insert(2, 300, "c")

    assert store.get(0) == {"id": 0, "file_offset": 100, "name": "a", "metadata": {"x": 1}}
    assert store.get(1) == {"id": 1, "file_offset": 200, "name": "b", "metadata": {"x": 2}}
    assert store.get(2) == {"id": 2, "file_offset": 300, "name": "c", "metadata": None}

def test_exists_does_not_cross_contaminate(store):
    store.insert(0, 100, "a")
    assert store.exists("a") is True
    assert store.exists("b") is False


# ---------------------------------------------------------------------------
# delete / count
# ---------------------------------------------------------------------------

def test_delete_makes_exists_return_false(store):
    store.insert(0, 100, "a")
    store.delete("a")
    assert store.exists("a") is False

def test_delete_makes_get_return_none(store):
    store.insert(0, 100, "a")
    store.delete("a")
    assert store.get(0) is None

def test_delete_is_idempotent(store):
    store.insert(0, 100, "a")
    store.delete("a")
    store.delete("a")  # should not raise
    assert store.exists("a") is False

def test_delete_nonexistent_is_noop(store):
    store.delete("ghost")  # should not raise

def test_count_starts_at_zero(store):
    assert store.count() == 0

def test_count_increments_on_insert(store):
    store.insert(0, 100, "a")
    store.insert(1, 200, "b")
    assert store.count() == 2

def test_count_decrements_on_delete(store):
    store.insert(0, 100, "a")
    store.insert(1, 200, "b")
    store.delete("a")
    assert store.count() == 1

def test_reinsert_after_delete_succeeds(store):
    store.insert(0, 100, "a")
    store.delete("a")
    store.insert(1, 200, "a")  # same name, new id
    assert store.exists("a") is True
    assert store.get(1)["name"] == "a"


# ---------------------------------------------------------------------------
# get_max_id
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# iter_offsets
# ---------------------------------------------------------------------------

def test_iter_offsets_empty(store):
    assert list(store.iter_offsets()) == []

def test_iter_offsets_returns_file_offsets(store):
    store.insert(1, 100, "a")
    store.insert(2, 200, "b")
    assert list(store.iter_offsets()) == [100, 200]

def test_iter_offsets_ordered_by_offset(store):
    store.insert(10, 300, "c")
    store.insert(3, 100, "a")
    store.insert(7, 200, "b")
    assert list(store.iter_offsets()) == [100, 200, 300]

def test_iter_offsets_excludes_deleted(store):
    store.insert(1, 100, "a")
    store.insert(2, 200, "b")
    store.insert(3, 300, "c")
    store.delete("b")
    assert list(store.iter_offsets()) == [100, 300]


# ---------------------------------------------------------------------------
# get_max_id
# ---------------------------------------------------------------------------

def test_get_max_id_returns_zero_on_empty(store):
    assert store.get_max_id() == 0

def test_get_max_id_returns_highest_id(store):
    store.insert(5, 100, "a")
    store.insert(10, 200, "b")
    store.insert(3, 300, "c")
    assert store.get_max_id() == 10

def test_get_max_id_after_delete(store):
    store.insert(1, 100, "a")
    store.insert(5, 200, "b")
    store.delete("b")
    assert store.get_max_id() == 1

def test_get_max_id_returns_zero_after_all_deleted(store):
    store.insert(1, 100, "a")
    store.delete("a")
    assert store.get_max_id() == 0


# ---------------------------------------------------------------------------
# update_offsets
# ---------------------------------------------------------------------------

def test_update_offsets_changes_offsets(store):
    store.insert(0, 100, "a")
    store.insert(1, 200, "b")
    store.update_offsets({100: 0, 200: 16})
    assert store.get(0)["file_offset"] == 0
    assert store.get(1)["file_offset"] == 16

def test_update_offsets_preserves_other_fields(store):
    store.insert(0, 100, "a", metadata={"x": 1})
    store.update_offsets({100: 0})
    row = store.get(0)
    assert row["name"] == "a"
    assert row["metadata"] == {"x": 1}

def test_update_offsets_reflected_in_iter_offsets(store):
    store.insert(0, 100, "a")
    store.insert(1, 200, "b")
    store.update_offsets({100: 0, 200: 16})
    assert list(store.iter_offsets()) == [0, 16]

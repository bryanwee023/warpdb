import sqlite3

import pytest

from warpdb.storage import MetadataStore

@pytest.fixture
def store():
    return MetadataStore(":memory:")

def test_exists_returns_false_for_unknown_id(store):
    assert store.exists("nonexistent") is False

def test_exists_returns_true_after_insert(store):
    store.insert("abc", vec_id=0)
    assert store.exists("abc") is True

def test_get_returns_none_for_unknown_vec_id(store):
    assert store.get(99) is None

def test_get_returns_correct_record(store):
    store.insert("abc", vec_id=7)
    result = store.get(7)
    assert result == {"id": "abc", "vec_id": 7, "metadata": None}

def test_insert_with_metadata(store):
    meta = {"color": "red", "score": 0.9}
    store.insert("abc", vec_id=0, metadata=meta)
    result = store.get(0)
    assert result["metadata"] == meta

def test_insert_without_metadata_returns_none(store):
    store.insert("abc", vec_id=0)
    result = store.get(0)
    assert result["metadata"] is None

def test_insert_duplicate_id_raises(store):
    store.insert("abc", vec_id=0)
    with pytest.raises(sqlite3.IntegrityError):
        store.insert("abc", vec_id=1)

def test_insert_duplicate_vec_id_raises(store):
    store.insert("abc", vec_id=0)
    with pytest.raises(sqlite3.IntegrityError):
        store.insert("xyz", vec_id=0)

def test_multiple_records_stored_independently(store):
    store.insert("a", vec_id=0, metadata={"x": 1})
    store.insert("b", vec_id=1, metadata={"x": 2})
    store.insert("c", vec_id=2)

    assert store.get(0) == {"id": "a", "vec_id": 0, "metadata": {"x": 1}}
    assert store.get(1) == {"id": "b", "vec_id": 1, "metadata": {"x": 2}}
    assert store.get(2) == {"id": "c", "vec_id": 2, "metadata": None}

def test_exists_does_not_cross_contaminate(store):
    store.insert("a", vec_id=0)
    assert store.exists("a") is True
    assert store.exists("b") is False

import numpy as np
import pytest

from warpdb.storage import VectorStore

DIM = 4

@pytest.fixture
def store(tmp_path):
    return VectorStore(str(tmp_path / "vectors.f32"), DIM)

def test_initial_count_is_zero(store):
    assert store.count() == 0

def test_append_returns_sequential_ids(store):
    v = np.ones(DIM, dtype=np.float32)
    assert store.append(v) == 0
    assert store.append(v) == 1
    assert store.append(v) == 2

def test_count_increments_after_append(store):
    v = np.ones(DIM, dtype=np.float32)
    store.append(v)
    assert store.count() == 1
    store.append(v)
    assert store.count() == 2

def test_get_returns_appended_vector(store):
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    store.append(v)
    result = store.get(0)
    np.testing.assert_array_equal(result, v)

def test_get_multiple_vectors_independent(store):
    v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    store.append(v0)
    store.append(v1)
    store.append(v2)
    np.testing.assert_array_equal(store.get(0), v0)
    np.testing.assert_array_equal(store.get(1), v1)
    np.testing.assert_array_equal(store.get(2), v2)

def test_append_casts_float64_to_float32(store):
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    store.append(v)
    result = store.get(0)
    assert result.dtype == np.float32
    np.testing.assert_array_almost_equal(result, v.astype(np.float32))

def test_append_rejects_wrong_dimension(store):
    v = np.ones(DIM + 1, dtype=np.float32)
    with pytest.raises(ValueError, match=str(DIM)):
        store.append(v)

def test_append_rejects_2d_array(store):
    v = np.ones((2, DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="1D"):
        store.append(v)

def test_get_raises_index_error_on_empty_store(store):
    with pytest.raises(IndexError):
        store.get(0)

def test_get_raises_index_error_for_negative_id(store):
    store.append(np.ones(DIM, dtype=np.float32))
    with pytest.raises(IndexError):
        store.get(-1)

def test_get_raises_index_error_for_id_equal_to_count(store):
    store.append(np.ones(DIM, dtype=np.float32))
    with pytest.raises(IndexError):
        store.get(1)  # count is 1, valid ids are [0]

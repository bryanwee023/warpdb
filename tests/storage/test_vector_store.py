import numpy as np
import pytest

from warpdb.storage import VectorStore

DIM = 4
BYTES_PER_VEC = DIM * 4  # 16

@pytest.fixture
def store(tmp_path):
    return VectorStore(str(tmp_path / "vectors.f32"), DIM)

def test_initial_count_is_zero(store):
    assert store.count() == 0

def test_append_returns_sequential_offsets(store):
    v = np.ones(DIM, dtype=np.float32)
    assert store.append(v) == 0
    assert store.append(v) == BYTES_PER_VEC
    assert store.append(v) == BYTES_PER_VEC * 2

def test_count_increments_after_append(store):
    v = np.ones(DIM, dtype=np.float32)
    store.append(v)
    assert store.count() == 1
    store.append(v)
    assert store.count() == 2

def test_get_returns_appended_vector(store):
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    offset = store.append(v)
    result = store.get(offset)
    np.testing.assert_array_equal(result, v)

def test_get_multiple_vectors_independent(store):
    v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    off0 = store.append(v0)
    off1 = store.append(v1)
    off2 = store.append(v2)
    np.testing.assert_array_equal(store.get(off0), v0)
    np.testing.assert_array_equal(store.get(off1), v1)
    np.testing.assert_array_equal(store.get(off2), v2)

def test_append_casts_float64_to_float32(store):
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    offset = store.append(v)
    result = store.get(offset)
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

def test_get_raises_index_error_for_negative_offset(store):
    store.append(np.ones(DIM, dtype=np.float32))
    with pytest.raises(IndexError):
        store.get(-BYTES_PER_VEC)

def test_get_raises_index_error_for_offset_past_end(store):
    store.append(np.ones(DIM, dtype=np.float32))
    with pytest.raises(IndexError):
        store.get(BYTES_PER_VEC)  # count is 1, only offset 0 is valid

def test_get_raises_value_error_for_misaligned_offset(store):
    store.append(np.ones(DIM, dtype=np.float32))
    with pytest.raises(ValueError, match="not aligned"):
        store.get(1)

def test_next_offset_starts_at_zero(store):
    assert store.next_offset() == 0

def test_next_offset_advances_after_append(store):
    v = np.ones(DIM, dtype=np.float32)
    store.append(v)
    assert store.next_offset() == BYTES_PER_VEC
    store.append(v)
    assert store.next_offset() == BYTES_PER_VEC * 2


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------

def test_size_starts_at_zero(store):
    assert store.size() == 0

def test_size_grows_after_append(store):
    v = np.ones(DIM, dtype=np.float32)
    store.append(v)
    assert store.size() == BYTES_PER_VEC
    store.append(v)
    assert store.size() == BYTES_PER_VEC * 2

def test_size_equals_next_offset(store):
    v = np.ones(DIM, dtype=np.float32)
    assert store.size() == store.next_offset()
    store.append(v)
    assert store.size() == store.next_offset()
    store.append(v)
    assert store.size() == store.next_offset()

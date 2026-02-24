import math
import random

import numpy as np
import pytest

from warpdb.index.hnsw import HNSW
from warpdb.storage import VectorStore

DIM = 8

@pytest.fixture
def store(tmp_path):
    return VectorStore(str(tmp_path / "vectors.f32"), DIM)

@pytest.fixture
def index(store):
    return HNSW(store, M=4, ef_construction=20, ef_search=20)


def _add(store, index, node_id, vector):
    """Append vector to store and insert into HNSW index."""
    file_offset = store.append(vector)
    index.insert(node_id, file_offset, vector)


# ---------------------------------------------------------------------------
# _dist
# ---------------------------------------------------------------------------

class TestDist:
    def test_identical_vectors_is_zero(self, index):
        v = np.ones(DIM, dtype=np.float32)
        assert index._dist(v, v) == pytest.approx(0.0)

    def test_known_squared_l2(self, index):
        # [0,...] vs [1,...]: each element contributes 1^2, total = DIM
        a = np.zeros(DIM, dtype=np.float32)
        b = np.ones(DIM, dtype=np.float32)
        assert index._dist(a, b) == pytest.approx(float(DIM))

    def test_is_symmetric(self, index):
        a = np.arange(DIM, dtype=np.float32)
        b = np.arange(DIM, dtype=np.float32)[::-1].copy()
        assert index._dist(a, b) == pytest.approx(index._dist(b, a))

    def test_is_squared_not_euclidean(self, index):
        # Euclidean distance = 2; squared L2 = 4
        a = np.zeros(DIM, dtype=np.float32)
        b = np.zeros(DIM, dtype=np.float32)
        b[0] = 2.0
        assert index._dist(a, b) == pytest.approx(4.0)

# ---------------------------------------------------------------------------
# _random_level
# ---------------------------------------------------------------------------

class TestRandomLevel:
    def test_always_non_negative(self, index):
        random.seed(0)
        assert all(index._random_level() >= 0 for _ in range(200))

    def test_majority_are_level_zero(self, index):
        # P(level == 0) = 1 - 1/M. With M=4, expected ~75%.
        random.seed(0)
        levels = [index._random_level() for _ in range(1000)]
        assert levels.count(0) / len(levels) > 0.6

    def test_higher_levels_are_less_frequent(self, index):
        random.seed(0)
        levels = [index._random_level() for _ in range(2000)]
        assert levels.count(0) > levels.count(1) > 0


# ---------------------------------------------------------------------------
# _select_neighbors
# ---------------------------------------------------------------------------

class TestSelectNeighbors:
    def test_selects_m_nearest_by_distance(self, index):
        candidates = [(10.0, 5), (1.0, 2), (5.0, 3), (2.0, 0)]
        result = index._select_neighbors(candidates, M=2)
        assert result == [2, 0]  # ids with distances 1.0 and 2.0

    def test_returns_all_when_fewer_than_m(self, index):
        candidates = [(3.0, 1), (1.0, 0)]
        result = index._select_neighbors(candidates, M=10)
        assert set(result) == {0, 1}

    def test_empty_candidates_returns_empty(self, index):
        assert index._select_neighbors([], M=5) == []

    def test_m_zero_returns_empty(self, index):
        candidates = [(1.0, 0), (2.0, 1)]
        assert index._select_neighbors(candidates, M=0) == []


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------

class TestInsert:
    def test_entry_point_none_before_firstinsert(self, index):
        assert index._entry_point is None

    def test_entry_point_set_after_firstinsert(self, store, index):
        random.seed(0)
        _add(store, index, 0, np.ones(DIM, dtype=np.float32))
        assert index._entry_point == 0

    def test_node_always_in_layer_zero(self, store, index):
        random.seed(0)
        _add(store, index, 0, np.ones(DIM, dtype=np.float32))
        assert 0 in index._graph[0]

    def test_max_layer_non_negative_afterinserts(self, store, index):
        random.seed(0)
        for i in range(10):
            _add(store, index, i, np.ones(DIM, dtype=np.float32))
        assert index._max_layer >= 0

    def test_offset_stored_correctly(self, store, index):
        random.seed(0)
        v = np.ones(DIM, dtype=np.float32)
        file_offset = store.append(v)
        index.insert(42, file_offset, v)
        assert index._offsets[42] == file_offset


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_empty_index_returns_empty(self, index):
        assert index.search(np.zeros(DIM, dtype=np.float32), k=5) == []

    def test_single_vector_is_found(self, store, index):
        random.seed(0)
        v = np.ones(DIM, dtype=np.float32)
        _add(store, index, 0, v)
        results = index.search(v, k=1)
        assert len(results) == 1
        assert results[0][1] == 0

    def test_returns_at_most_k_results(self, store, index):
        random.seed(0)
        np.random.seed(0)
        for i in range(20):
            _add(store, index, i, np.random.rand(DIM).astype(np.float32))
        results = index.search(np.zeros(DIM, dtype=np.float32), k=5)
        assert len(results) <= 5

    def test_returns_fewer_results_than_k_when_index_is_small(self, store, index):
        random.seed(0)
        for i in range(3):
            _add(store, index, i, np.random.rand(DIM).astype(np.float32))
        results = index.search(np.zeros(DIM, dtype=np.float32), k=10)
        assert len(results) <= 3

    def test_results_sorted_ascending_by_distance(self, store, index):
        random.seed(0)
        np.random.seed(0)
        for i in range(20):
            _add(store, index, i, np.random.rand(DIM).astype(np.float32))
        results = index.search(np.zeros(DIM, dtype=np.float32), k=10)
        dists = [d for d, _ in results]
        assert dists == sorted(dists)

    def test_distances_are_non_negative(self, store, index):
        random.seed(0)
        np.random.seed(0)
        for i in range(20):
            _add(store, index, i, np.random.rand(DIM).astype(np.float32))
        results = index.search(np.ones(DIM, dtype=np.float32), k=10)
        assert all(d >= 0 for d, _ in results)

    def test_nearest_neighbor_clearly_separated(self, store, index):
        """Vectors placed on scaled basis axes are trivially separable."""
        random.seed(42)
        # Each basis vector is 100 units from the origin along one axis.
        # Distance between any two is 100^2 * 2 = 20000, so the NN is unambiguous.
        for i in range(DIM):
            v = np.zeros(DIM, dtype=np.float32)
            v[i] = 100.0
            _add(store, index, i, v)

        target = np.zeros(DIM, dtype=np.float32)
        target[3] = 100.0
        query = target + 0.1  # slightly perturbed version of basis[3]

        results = index.search(query, k=1)
        assert results[0][1] == 3

    def test_exact_query_match_has_zero_distance(self, store, index):
        random.seed(0)
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        _add(store, index, 0, v)
        results = index.search(v, k=1)
        assert results[0][0] == pytest.approx(0.0)

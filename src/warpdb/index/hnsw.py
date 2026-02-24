import heapq
import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from warpdb.storage.vector_store import VectorStore

class HNSW:
    def __init__(
        self,
        vector_store: VectorStore,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):

        self._M = M
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._mL = 1.0 / math.log(M) # level generation normalization factor

        self._vector_store = vector_store # _graph[layer][node_id] = {neighbor_ids}
        self._offsets: Dict[int, int] = {}  # node_id -> file_offset
        self._graph: Dict[int, Dict[int, Set[int]]] = {}
        self._entry_point: Optional[int] = None
        self._max_layer: int = -1

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        """Squared L2 distance between two vectors."""
        diff = a - b
        return float(np.dot(diff, diff))

    def _random_level(self) -> int:
        """
        Decides how many layers a new node participates in,
        using exponential distribution.
        """
        return int(math.floor(-math.log(random.random()) * self._mL))

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int,
    ) -> List[Tuple[float, int]]:
        """
        Greedy search within a single layer.
        Returns up to ef nearest neighbors as (dist, node_id) sorted ascending.
        """
        visited: Set[int] = set(entry_points)
        candidates: List[Tuple[float, int]] = [] # min-heap: (dist, node_id)
        found: List[Tuple[float, int]] = [] # max-heap: (-dist, node_id)

        for eid in entry_points:
            offset = self._offsets[eid]
            dist = self._dist(query, self._vector_store.get(offset))
            heapq.heappush(candidates, (dist, eid))
            heapq.heappush(found, (-dist, eid))

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            dist_f = -found[0][0] # distance of farthest node in found

            if dist_c > dist_f:
                break # all remaining candidates are farther than worst in found

            neighbors = self._graph.get(layer, {}).get(c, set())
            for nid in list(neighbors):
                if nid not in self._offsets:
                    continue # node was lazily deleted, so skip
                if nid in visited:
                    continue

                visited.add(nid)
                offset = self._offsets[nid]
                dist_n = self._dist(query, self._vector_store.get(offset))
                dist_f = -found[0][0]

                if dist_n < dist_f or len(found) < ef:
                    heapq.heappush(candidates, (dist_n, nid))
                    heapq.heappush(found, (-dist_n, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-dist, nid) for dist, nid in found]
        result.sort()
        return result

    def _select_neighbors(
        self, candidates: List[Tuple[float, int]], M: int
    ) -> List[int]:
        """
        Given a list of (dist, node_id) candidates, select up to M nearest neighbors.
        """
        return [nid for _, nid in sorted(candidates)[:M]]

    def insert(self, node_id: int, file_offset: int, vector: np.ndarray) -> None:
        """
        Add a node that is already in the vector store to the graph.
        """
        self._offsets[node_id] = file_offset
        level = self._random_level()

        for layer_id in range(level + 1):
            self._graph.setdefault(layer_id, {})[node_id] = set()

        if self._entry_point is None:
            self._entry_point = node_id
            self._max_layer = level
            return

        eid = [self._entry_point]

        # Greedy descent from top layer to level+1 (no connections, just find entry)
        for layer_id in range(self._max_layer, level, -1):
            results = self._search_layer(vector, eid, ef=1, layer=layer_id)
            eid = [results[0][1]]

        # Insert and connect from min(level, max_layer) down to 0
        for layer_id in range(min(level, self._max_layer), -1, -1):
            M_max = 2 * self._M if layer_id == 0 else self._M # Layer 0 allows more connections
            results = self._search_layer(vector, eid, self._ef_construction, layer_id)
            neighbors = self._select_neighbors(results, M_max)

            for nid in neighbors:
                self._graph[layer_id][node_id].add(nid)
                self._graph[layer_id].setdefault(nid, set()).add(node_id)

                # Prune nid's connections if over limit
                if len(self._graph[layer_id][nid]) > M_max:
                    n_offset = self._offsets[nid]
                    n_vec = self._vector_store.get(n_offset)
                    n_candidates = [
                        (self._dist(n_vec, self._vector_store.get(self._offsets[x])), x)
                        for x in self._graph[layer_id][nid]
                    ]
                    self._graph[layer_id][nid] = set(self._select_neighbors(n_candidates, M_max))

            eid = [r[1] for r in results]

        if level > self._max_layer:
            self._max_layer = level
            self._entry_point = node_id

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[float, int]]:
        """
        Returns the k nearest neighbors as (dist, node_id) sorted by distance.
        """
        if self._entry_point is None:
            return []

        eid = [self._entry_point]

        # Greedy descent to layer 1
        for lc in range(self._max_layer, 0, -1):
            results = self._search_layer(query, eid, ef=1, layer=lc)
            eid = [results[0][1]]

        # Beam search at layer 0
        results = self._search_layer(query, eid, max(self._ef_search, k), layer=0)
        return results[:k]

    def delete(self, node_id: int) -> None:
        """Mark a node as deleted."""
        self._offsets.pop(node_id, None)

        if self._entry_point == node_id:
            if self._offsets:
                self._entry_point = next(iter(self._offsets))
            else:
                self._entry_point = None
                self._max_layer = -1

    def clear(self) -> None:
        """Remove all nodes from the graph."""
        self._offsets.clear()
        self._graph.clear()
        self._entry_point = None
        self._max_layer = -1

    def compact(self, updates: Dict[int, int]) -> None:
        """
        Apply offset updates and prune edges to deleted nodes.
        updates: {old_offset: new_offset}
        """
        new_offsets = {}
        for node_id, old_offset in self._offsets.items():
            if old_offset in updates:
                new_offsets[node_id] = updates[old_offset]

        self._offsets.clear()
        self._graph.clear()
        self._entry_point = None
        self._max_layer = -1

        for node_id, file_offset in new_offsets.items():
            vec = self._vector_store.get(file_offset)
            self.insert(node_id, file_offset, vec)

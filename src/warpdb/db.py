import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from warpdb.storage import MetadataStore, VectorStore
from warpdb.storage.wal import WAL, DeleteRecord, UpsertRecord
from warpdb.index import HNSW

class WarpDB:
    def __init__(self, dim: int, data_dir: str = "."):
        dir_path = Path(data_dir)
        self._vector_store = VectorStore(str(dir_path / "vectors.f32"), dim)
        self._metadata_store = MetadataStore(str(dir_path / "metadata.db"))
        self._wal = WAL(str(dir_path / "wal.bin"), dim)
        self._index = HNSW(self._vector_store)
        self._lock = threading.Lock()

        # Replay any writes that didn't complete before the last crash
        self._recover()

        # Rebuild the in-memory HNSW graph from the (now-consistent) vector store,
        # skipping soft-deleted entries. TODO: revisit after compaction is implemented.
        for vec_id in range(self._vector_store.count()):
            if self._metadata_store.get(vec_id) is not None:
                vec = self._vector_store.get(vec_id)
                self._index._insert(vec_id, vec)

        # All pending WAL entries are resolved; compact the log
        self._wal.checkpoint()

    def count(self) -> int:
        return self._metadata_store.count_active()

    def upsert(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[dict] = None,
    ):
        with self._lock:
            if self._metadata_store.exists(id):
                raise ValueError(f"ID '{id}' already exists")

            vec = np.array(vector, dtype=np.float32)
            vec_id = self._vector_store.count()  # safe prediction under the lock

            # Log intent before touching any persistent storage
            lsn = self._wal.log_upsert(id, vec_id, vec, metadata)

            vec_id = self._vector_store.append(vec)
            self._index._insert(vec_id, vec)
            self._metadata_store.insert(id, vec_id, metadata)

            self._wal.log_commit(lsn)

    def search(
        self,
        query: List[float],
        k: int = 10,
    ) -> List[Tuple[float, int]]:
        with self._lock:
            vec = np.array(query, dtype=np.float32)
            candidates = self._index.search(vec, k)

            results = []
            for dist, vec_id in candidates:
                metadata = self._metadata_store.get(vec_id)
                if metadata:
                    results.append((dist, metadata["id"]))
            return results

    def delete(self, id: str) -> None:
        with self._lock:
            if not self._metadata_store.exists(id):
                raise ValueError(f"ID '{id}' not found")

            lsn = self._wal.log_delete(id)
            self._metadata_store.delete(id)
            self._wal.log_commit(lsn)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _recover(self) -> None:
        """Replay any WAL entries that did not reach COMMIT before a crash."""
        for record in self._wal.get_pending():
            if isinstance(record, UpsertRecord):
                if record.vec_id >= self._vector_store.count():
                    # Vector was never written — append it now
                    self._vector_store.append(record.vector)

                if self._metadata_store.get(record.vec_id) is None:
                    self._metadata_store.insert(record.id, record.vec_id, record.metadata)

                self._wal.log_commit(record.lsn)
            elif isinstance(record, DeleteRecord):
                self._metadata_store.delete(record.id)  # idempotent
                self._wal.log_commit(record.lsn)


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

        self._next_id = self._metadata_store.get_max_id() + 1

        # Rebuild the in-memory HNSW graph from the (now-consistent) metadata store.
        for id, file_offset in self._metadata_store.iter_offsets():
            vec = self._vector_store.get(file_offset)
            self._index.insert(id, file_offset, vec)

        # All pending WAL entries are resolved; compact the log
        self._wal.checkpoint()

    def count(self) -> int:
        return self._metadata_store.count()

    def upsert(
        self,
        name: str,
        vector: List[float],
        metadata: Optional[dict] = None,
    ):
        with self._lock:
            if self._metadata_store.exists(name):
                raise ValueError(f"Name '{name}' already exists")

            vec = np.array(vector, dtype=np.float32)
            id = self._next_id
            file_offset = self._vector_store.size()

            # Log intent before touching any persistent storage
            lsn = self._wal.log_upsert(id, file_offset, vec, name, metadata)

            self._vector_store.append(vec)
            self._index.insert(id, file_offset, vec)
            self._metadata_store.insert(id, file_offset, name, metadata)

            self._next_id += 1

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
                    results.append((dist, metadata["name"]))
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
                if record.file_offset >= self._vector_store.size():
                    # Vector was never written — append it now
                    self._vector_store.append(record.vector)

                if self._metadata_store.get(record.id) is None:
                    self._metadata_store.insert(
                        record.id, record.file_offset, record.name, record.metadata)

                self._wal.log_commit(record.lsn)
            elif isinstance(record, DeleteRecord):
                self._metadata_store.delete(record.id)  # idempotent
                self._wal.log_commit(record.lsn)


import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from warpdb.storage import MetadataStore, VectorStore
from warpdb.storage.wal import WAL, CompactRecord, DeleteRecord, UpsertRecord
from warpdb.index import HNSW

_COMPACT_THRESHOLD = 0.25

class WarpDB:
    def __init__(self, dim: int, data_dir: str = "."):
        dir_path = Path(data_dir)
        self._dim = dim
        self._vector_store = VectorStore(str(dir_path / "vectors.f32"), dim)
        self._metadata_store = MetadataStore(str(dir_path / "metadata.db"))
        self._wal = WAL(str(dir_path / "wal.bin"), dim)
        self._compact_tmp_path = str(dir_path / "vectors.compact.f32")
        self._index = HNSW(self._vector_store)
        self._lock = threading.Lock()

        self._recover()

        self._next_id = self._metadata_store.get_max_id() + 1

        # Rebuild the in-memory HNSW graph from the (now-consistent) metadata store.
        for id, file_offset in self._metadata_store.iter_ids_with_offsets():
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

    def delete(self, name: str) -> None:
        with self._lock:
            id = self._metadata_store.get_id(name)
            if id is None:
                raise ValueError(f"Name '{name}' not found")

            lsn = self._wal.log_delete(name)
            self._metadata_store.delete(name)
            self._index.delete(id)
            self._wal.log_commit(lsn)

            if self._should_compact():
                self._compact()

    def compact(self) -> None:
        """Remove dead vectors from disk and update all offsets."""
        with self._lock:
            self._compact()
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
            elif isinstance(record, CompactRecord):
                if os.path.exists(self._compact_tmp_path):
                    # os.replace never happened — discard temp file
                    os.remove(self._compact_tmp_path)
                else:
                    # os.replace succeeded but metadata not updated — recompute offsets
                    bytes_per_vec = self._dim * 4
                    updates = {
                        old_offset: i * bytes_per_vec
                        for i, old_offset in enumerate(self._metadata_store.iter_offsets())
                    }
                    self._metadata_store.update_offsets(updates)
                self._wal.log_commit(record.lsn)

    def _should_compact(self) -> bool:
        total = self._vector_store.count()
        if total == 0:
            return False
        dead = total - self._metadata_store.count()
        return dead / total > _COMPACT_THRESHOLD

    def _compact(self) -> None:
        live_offsets = self._metadata_store.iter_offsets()

        lsn = self._wal.log_compact()
        updates = self._vector_store.compact(live_offsets)
        self._metadata_store.update_offsets(updates)
        self._wal.log_commit(lsn)

        # Rebuild HNSW with updated offsets
        self._index.compact(updates)


import os
from typing import Dict, Iterator, List, Optional

import numpy as np

class VectorStore:
    def __init__(self, path: str, dim: int):
        self._path = path
        self._dim = dim
        self._mmap: Optional[np.memmap] = None

        bytes_per_vec = dim * 4
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            self._count = file_size // bytes_per_vec
            # Truncate any partial vector written before a crash
            if file_size % bytes_per_vec:
                os.truncate(path, self._count * bytes_per_vec)
        else:
            self._count = 0

    def count(self) -> int:
        return self._count

    def size(self) -> int:
        """Return the size of the vector store in bytes."""
        return self._count * self._dim * 4

    def next_offset(self) -> int:
        """Return the byte offset where the next append will write."""
        return self._count * self._dim * 4

    def _get_mmap(self) -> np.memmap:
        if self._mmap is None:
            self._mmap = np.memmap(self._path, dtype=np.float32, mode="r", shape=(self._count, self._dim))
        return self._mmap

    def append(self, vector: np.ndarray) -> int:
        """Append a single vector and return its byte offset in the file."""
        if vector.ndim != 1:
            raise ValueError("Input vectors must be a 1D array.")

        dim = vector.shape[0]

        if dim != self._dim:
            raise ValueError(f"Expected vectors with dimension {self._dim}, got {dim}.")

        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)

        offset = self._count * self._dim * 4

        with open(self._path, "ab") as f:
            vector.tofile(f)
            f.flush()
            os.fsync(f.fileno())

        self._count += 1
        self._mmap = None  # invalidate cached mmap

        return offset

    def get(self, offset: int) -> np.ndarray:
        """Retrieve a single vector by its byte offset in the file."""
        bytes_per_vec = self._dim * 4
        if offset % bytes_per_vec != 0:
            raise ValueError(f"Offset {offset} is not aligned to vector size ({bytes_per_vec} bytes).")
        index = offset // bytes_per_vec
        if index < 0 or index >= self._count:
            raise IndexError("Offset out of range.")

        return self._get_mmap()[index]

    def compact(self, live_offsets: Iterator[int]) -> Dict[int, int]:
        """Rewrite file with only the vectors at the given offsets.

        Returns mapping of old_offset -> new_offset.
        """
        tmp_path = self._path.replace(".f32", ".compact.f32")
        bytes_per_vec = self._dim * 4
        mapping: Dict[int, int] = {}

        count = 0
        with open(tmp_path, "wb") as f:
            for i, live_offset in enumerate(live_offsets):
                vec = self.get(live_offset)
                vec.tofile(f)
                mapping[live_offset] = i * bytes_per_vec
                count += 1
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, self._path)
        self._count = count
        self._mmap = None
        return mapping


import os
from typing import Optional

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

    def _get_mmap(self) -> np.memmap:
        if self._mmap is None:
            self._mmap = np.memmap(self._path, dtype=np.float32, mode="r", shape=(self._count, self._dim))
        return self._mmap

    def append(self, vectors: np.ndarray) -> int:
        """
        vectors: shape (N, dim), dtype float32
        Appends to the flat binary .f32 file.
        """
        if vectors.ndim != 1:
            raise ValueError("Input vectors must be a 1D array.")

        dim = vectors.shape[0]

        if dim != self._dim:
            raise ValueError(f"Expected vectors with dimension {self._dim}, got {dim}.")
        
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        with open(self._path, "ab") as f:
            vectors.tofile(f)

        self._count += 1
        self._mmap = None  # invalidate cached mmap

        return self._count - 1

    def get(self, id: int) -> np.ndarray:
        """
        Retrieves a single vector by its index.
        """
        if id < 0 or id >= self._count:
            raise IndexError("Index out of range.")

        return self._get_mmap()[id]


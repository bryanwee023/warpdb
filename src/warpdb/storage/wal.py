import json
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

_MAGIC = b"WWAL"
_VERSION = 1

# File layout:
#   Header:  magic(4) version(2) dim(4)  = 10 bytes at offset 0, written once
#   Records: appended after the header
_HEADER_SIZE = 10

# Record type bytes
_TYPE_UPSERT = 0x01
_TYPE_COMMIT = 0x02

# UPSERT layout: magic(4) type(1) lsn(4) id_len(4) id(N) vec_id(4) vector(dim*4) meta_len(4) meta(M)
# COMMIT layout: magic(4) type(1) lsn(4)


@dataclass
class UpsertRecord:
    lsn: int
    id: str
    vec_id: int
    vector: np.ndarray
    metadata: Optional[Dict[str, Any]]


class WAL:
    def __init__(self, path: str, dim: int):
        self._path = path
        self._dim = dim

        if os.path.exists(path) and os.path.getsize(path) >= _HEADER_SIZE:
            self._validate_header()
        else:
            self._write_header()

        self._lsn = self._next_lsn()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_upsert(
        self,
        id: str,
        vec_id: int,
        vector: np.ndarray,
        metadata: Optional[dict],
    ) -> int:
        """Write an UPSERT record and fsync. Returns the LSN."""
        lsn = self._lsn
        self._lsn += 1

        id_bytes = id.encode("utf-8")
        meta_bytes = json.dumps(metadata).encode("utf-8") if metadata is not None else b""
        vector_f32 = vector.astype(np.float32)

        header = _MAGIC + struct.pack("<BI", _TYPE_UPSERT, lsn)
        body = (
            struct.pack("<I", len(id_bytes))
            + id_bytes
            + struct.pack("<I", vec_id)
            + vector_f32.tobytes()
            + struct.pack("<I", len(meta_bytes))
            + meta_bytes
        )

        with open(self._path, "ab") as f:
            f.write(header + body)
            f.flush()
            os.fsync(f.fileno())

        return lsn

    def log_commit(self, lsn: int) -> None:
        """Write a COMMIT record and fsync."""
        record = _MAGIC + struct.pack("<BI", _TYPE_COMMIT, lsn)

        with open(self._path, "ab") as f:
            f.write(record)
            f.flush()
            os.fsync(f.fileno())

    def get_pending(self) -> List[UpsertRecord]:
        """Return the trailing UPSERT record if it has no matching COMMIT (at most one).

        Because upsert() holds a lock, only one operation is in flight at a time,
        so a pending entry is always at the tail of the WAL.
        """
        last_upsert: Optional[UpsertRecord] = None

        for rec_type, _, payload in self._iter_records():
            if rec_type == _TYPE_UPSERT:
                assert payload is not None
                last_upsert = payload
            elif rec_type == _TYPE_COMMIT:
                last_upsert = None

        return [last_upsert] if last_upsert is not None else []

    def checkpoint(self) -> None:
        """Rewrite the file to header-only (call only when all entries are committed)."""
        self._write_header()
        self._lsn = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_header(self) -> None:
        """Create or overwrite the WAL file with just the 10-byte header."""
        with open(self._path, "wb") as f:
            f.write(_MAGIC + struct.pack("<HI", _VERSION, self._dim))
            f.flush()
            os.fsync(f.fileno())

    def _validate_header(self) -> None:
        """Read the header and raise if magic, version, or dim don't match."""
        with open(self._path, "rb") as f:
            data = f.read(_HEADER_SIZE)

        if data[:4] != _MAGIC:
            raise ValueError(f"WAL file corrupt: bad magic {data[:4]!r}")

        version, dim = struct.unpack("<HI", data[4:10])

        if version != _VERSION:
            raise ValueError(f"WAL version mismatch: file={version}, expected={_VERSION}")

        if dim != self._dim:
            raise ValueError(f"WAL dim mismatch: file={dim}, expected={self._dim}")

    def _next_lsn(self) -> int:
        """Scan WAL records and return the next available LSN."""
        max_lsn = -1
        for _, lsn, _ in self._iter_records():
            if lsn > max_lsn:
                max_lsn = lsn
        return max_lsn + 1

    def _iter_records(self):
        """
        Yield (type, lsn, payload) for each valid record after the header.
        Stops silently on truncated/corrupt data (treat as end-of-file).
        payload is an UpsertRecord for UPSERT records, None for COMMIT records.
        """
        if not os.path.exists(self._path):
            return

        with open(self._path, "rb") as f:
            f.seek(_HEADER_SIZE)  # skip the file header

            while True:
                # Every record starts with: magic(4) + type(1) + lsn(4)
                raw = f.read(4 + 1 + 4)
                if len(raw) < 9:
                    break  # truncated

                magic = raw[:4]
                rec_type = raw[4]
                lsn = struct.unpack("<I", raw[5:9])[0]

                if magic != _MAGIC:
                    break # corrupt

                if rec_type == _TYPE_COMMIT:
                    yield _TYPE_COMMIT, lsn, None

                elif rec_type == _TYPE_UPSERT:
                    record = self._read_upsert_payload(f, lsn, self._dim)
                    if record is None:
                        break  # truncated mid-record
                    yield _TYPE_UPSERT, lsn, record

                else:
                    break  # unknown type — stop

    @staticmethod
    def _read_upsert_payload(f, lsn: int, dim: int) -> Optional[UpsertRecord]:
        """Read the variable-length body of an UPSERT record from file f."""
        # id_len (4 bytes)
        raw = f.read(4)
        if len(raw) < 4:
            return None
        (id_len,) = struct.unpack("<I", raw)

        # id
        id_bytes = f.read(id_len)
        if len(id_bytes) < id_len:
            return None
        id_ = id_bytes.decode("utf-8")

        # vec_id (4 bytes) — dim comes from the file header, not per-record
        raw = f.read(4)
        if len(raw) < 4:
            return None
        (vec_id,) = struct.unpack("<I", raw)

        # vector
        vec_bytes = f.read(dim * 4)
        if len(vec_bytes) < dim * 4:
            return None
        vector = np.frombuffer(vec_bytes, dtype=np.float32).copy()

        # meta_len (4)
        raw = f.read(4)
        if len(raw) < 4:
            return None
        (meta_len,) = struct.unpack("<I", raw)

        # metadata
        meta_bytes = f.read(meta_len)
        if len(meta_bytes) < meta_len:
            return None
        metadata = json.loads(meta_bytes.decode("utf-8")) if meta_len > 0 else None

        return UpsertRecord(lsn=lsn, id=id_, vec_id=vec_id, vector=vector, metadata=metadata)

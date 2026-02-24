import numpy as np
import pytest

from warpdb.storage.wal import WAL, DeleteRecord, UpsertRecord

DIM = 4


@pytest.fixture
def wal(tmp_path):
    return WAL(str(tmp_path / "wal.bin"), DIM)


def _vec(val: float) -> np.ndarray:
    return np.full(DIM, val, dtype=np.float32)


# ---------------------------------------------------------------------------
# Basic log_upsert behaviour
# ---------------------------------------------------------------------------

def test_log_upsert_creates_pending_record(wal):
    lsn = wal.log_upsert(0, 100, _vec(1.0), "a", {"k": "v"})
    pending = wal.get_pending()
    assert len(pending) == 1
    assert pending[0].lsn == lsn
    assert pending[0].id == 0
    assert pending[0].file_offset == 100
    assert pending[0].name == "a"
    np.testing.assert_array_equal(pending[0].vector, _vec(1.0))
    assert pending[0].metadata == {"k": "v"}


def test_no_pending_on_empty_wal(wal):
    assert wal.get_pending() == []


def test_metadata_none_roundtrips(wal):
    wal.log_upsert(0, 5, _vec(0.5), "x", None)
    pending = wal.get_pending()
    assert pending[0].metadata is None


def test_metadata_dict_roundtrips(wal):
    meta = {"label": "cat", "score": 0.99}
    wal.log_upsert(0, 7, _vec(0.1), "y", meta)
    pending = wal.get_pending()
    assert pending[0].metadata == meta


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def test_checkpoint_truncates_wal(wal):
    wal.log_upsert(0, 0, _vec(1.0), "a", None)
    wal.checkpoint()
    assert wal.get_pending() == []


def test_checkpoint_resets_lsn(wal):
    wal.log_upsert(0, 0, _vec(1.0), "a", None)
    wal.checkpoint()
    lsn2 = wal.log_upsert(1, 100, _vec(2.0), "b", None)
    assert lsn2 == 0  # numbering restarts after checkpoint


# ---------------------------------------------------------------------------
# Header validation
# ---------------------------------------------------------------------------

def test_header_written_on_creation(tmp_path):
    import os
    path = str(tmp_path / "wal.bin")
    WAL(path, DIM)
    assert os.path.getsize(path) == 10  # header only, no records yet


def test_header_dim_mismatch_raises(tmp_path):
    path = str(tmp_path / "wal.bin")
    WAL(path, DIM)  # creates file with dim=4
    with pytest.raises(ValueError, match="dim mismatch"):
        WAL(path, DIM + 1)  # tries to open with dim=5


def test_header_survives_checkpoint(tmp_path):
    path = str(tmp_path / "wal.bin")
    w = WAL(path, DIM)
    w.log_upsert(0, 0, _vec(1.0), "a", None)
    w.checkpoint()
    # Should be readable again with the same dim
    w2 = WAL(path, DIM)
    assert w2.get_pending() == []


# ---------------------------------------------------------------------------
# Persistence across WAL instances (simulates restart)
# ---------------------------------------------------------------------------

def test_pending_survives_wal_restart(tmp_path):
    path = str(tmp_path / "wal.bin")

    w1 = WAL(path, DIM)
    w1.log_upsert(0, 0, _vec(1.0), "a", None)
    # no commit — simulates crash

    w2 = WAL(path, DIM)  # new instance reads same file
    pending = w2.get_pending()
    assert len(pending) == 1
    assert pending[0].id == 0
    assert pending[0].name == "a"


def test_checkpoint_clears_pending_after_restart(tmp_path):
    path = str(tmp_path / "wal.bin")

    w1 = WAL(path, DIM)
    w1.log_upsert(0, 0, _vec(1.0), "a", None)
    w1.checkpoint()

    w2 = WAL(path, DIM)
    assert w2.get_pending() == []


def test_lsn_continues_after_restart(tmp_path):
    path = str(tmp_path / "wal.bin")

    w1 = WAL(path, DIM)
    lsn0 = w1.log_upsert(0, 0, _vec(1.0), "a", None)

    w2 = WAL(path, DIM)
    lsn1 = w2.log_upsert(1, 100, _vec(2.0), "b", None)
    assert lsn1 > lsn0


# ---------------------------------------------------------------------------
# Recovery simulation (integration-style)
# ---------------------------------------------------------------------------

def test_recovery_crash_after_vector_write(tmp_path):
    """Simulate: vector written, metadata missing. Recovery should insert metadata."""
    from warpdb.storage import VectorStore, MetadataStore

    vs = VectorStore(str(tmp_path / "vectors.f32"), DIM)
    ms = MetadataStore(str(tmp_path / "metadata.db"))
    wal = WAL(str(tmp_path / "wal.bin"), DIM)

    vec = _vec(1.0)
    file_offset = vs.size()  # == 0

    # Log intent
    wal.log_upsert(0, file_offset, vec, "crash-victim", {"info": "test"})

    # Write vector — then "crash" (no metadata insert, no commit)
    vs.append(vec)

    # --- restart ---
    vs2 = VectorStore(str(tmp_path / "vectors.f32"), DIM)
    ms2 = MetadataStore(str(tmp_path / "metadata.db"))
    wal2 = WAL(str(tmp_path / "wal.bin"), DIM)

    # Recovery logic (mirrors WarpDB._recover)
    for record in wal2.get_pending():
        if record.file_offset >= vs2.size():
            vs2.append(record.vector)
        if ms2.get(record.id) is None:
            ms2.insert(record.id, record.file_offset, record.name, record.metadata)

    row = ms2.get(0)
    assert row is not None
    assert row["name"] == "crash-victim"
    assert row["metadata"] == {"info": "test"}
    assert vs2.count() == 1


def test_recovery_crash_before_vector_write(tmp_path):
    """Simulate: WAL written but vector never written. Recovery should write vector + metadata."""
    from warpdb.storage import VectorStore, MetadataStore

    vs = VectorStore(str(tmp_path / "vectors.f32"), DIM)
    ms = MetadataStore(str(tmp_path / "metadata.db"))
    wal = WAL(str(tmp_path / "wal.bin"), DIM)

    vec = _vec(2.0)
    file_offset = vs.size()  # == 0

    # Log intent — then "crash" immediately (nothing else written)
    wal.log_upsert(0, file_offset, vec, "no-vector", None)

    # --- restart ---
    vs2 = VectorStore(str(tmp_path / "vectors.f32"), DIM)
    ms2 = MetadataStore(str(tmp_path / "metadata.db"))
    wal2 = WAL(str(tmp_path / "wal.bin"), DIM)

    for record in wal2.get_pending():
        if record.file_offset >= vs2.size():
            vs2.append(record.vector)
        if ms2.get(record.id) is None:
            ms2.insert(record.id, record.file_offset, record.name, record.metadata)

    assert vs2.count() == 1
    np.testing.assert_array_equal(vs2.get(0), vec)
    row = ms2.get(0)
    assert row is not None
    assert row["name"] == "no-vector"


# ---------------------------------------------------------------------------
# log_delete
# ---------------------------------------------------------------------------

def test_log_delete_creates_pending_record(wal):
    lsn = wal.log_delete("a")
    pending = wal.get_pending()
    assert len(pending) == 1
    assert isinstance(pending[0], DeleteRecord)
    assert pending[0].lsn == lsn
    assert pending[0].name == "a"


def test_no_pending_on_empty_wal_after_delete_section(wal):
    assert wal.get_pending() == []


def test_pending_delete_survives_wal_restart(tmp_path):
    path = str(tmp_path / "wal.bin")

    w1 = WAL(path, DIM)
    w1.log_delete("to-delete")
    # no commit — simulates crash

    w2 = WAL(path, DIM)
    pending = w2.get_pending()
    assert len(pending) == 1
    assert isinstance(pending[0], DeleteRecord)
    assert pending[0].name == "to-delete"


def test_pending_upsert_is_upsert_record_not_delete_record(wal):
    wal.log_upsert(0, 0, _vec(1.0), "a", None)
    pending = wal.get_pending()
    assert len(pending) == 1
    assert isinstance(pending[0], UpsertRecord)

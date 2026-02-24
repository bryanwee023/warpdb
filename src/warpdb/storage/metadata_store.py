import json
from typing import Any, Dict, Iterator, List, Optional, Tuple
import sqlite3

class MetadataStore:
    def __init__(self, path: str):
        # Disable thread-origin check so the connection can be used across threads.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                id             INTEGER PRIMARY KEY,
                file_offset    INTEGER NOT NULL UNIQUE,
                name           TEXT NOT NULL UNIQUE,
                metadata       TEXT
            )
            """
        )
        self._conn.commit()

    def get_max_id(self) -> int:
        cursor = self._conn.cursor()
        cursor.execute("SELECT MAX(id) FROM metadata")
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else 0

    def insert(self, id: int, file_offset: int, name: str, metadata: Optional[dict] = None):
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO metadata (id, file_offset, name, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (id, file_offset, name, json.dumps(metadata) if metadata else None),
        )
        self._conn.commit()

    def exists(self, name: str) -> bool:
        cursor = self._conn.cursor()
        cursor.execute("SELECT 1 FROM metadata WHERE name = ?", (name,))
        return cursor.fetchone() is not None

    def get(self, id: int) -> Optional[Dict[str, Any]]:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT file_offset, name, metadata
            FROM metadata
            WHERE id = ?
            """,
            (id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        file_offset, name, metadata_json = row
        metadata = json.loads(metadata_json) if metadata_json else None
        return {
            "id": id,
            "file_offset": file_offset,
            "name": name,
            "metadata": metadata,
        }

    def get_id(self, name: str) -> Optional[int]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT id FROM metadata WHERE name = ?", (name,))
        row = cursor.fetchone()
        return row[0] if row else None

    def delete(self, name: str) -> None:
        self._conn.execute(
            "DELETE FROM metadata WHERE name = ?",
            (name,),
        )
        self._conn.commit()

    def iter_offsets(self) -> Iterator[int]:
        """Yield file_offset values ordered by file_offset."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT file_offset FROM metadata ORDER BY file_offset")
        for row in cursor:
            yield row[0]

    def iter_ids_with_offsets(self) -> Iterator[Tuple[int, int]]:
        """Yield (id, file_offset) pairs ordered by id."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT id, file_offset FROM metadata ORDER BY id")
        for row in cursor:
            yield row[0], row[1]

    def update_offsets(self, updates: Dict[int, int]) -> None:
        """Batch update file_offset for records. updates: {old_offset: new_offset}"""
        cursor = self._conn.cursor()
        cursor.execute("SELECT id, file_offset FROM metadata")
        offset_to_id = {file_offset: row_id for row_id, file_offset in cursor}

        self._conn.executemany(
            "UPDATE metadata SET file_offset = ? WHERE id = ?",
            [
                (new_offset, offset_to_id[old_offset])
                for old_offset, new_offset in updates.items()
                if old_offset in offset_to_id
            ],
        )
        self._conn.commit()

    def count(self) -> int:
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM metadata")
        return cursor.fetchone()[0]

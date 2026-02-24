import json
from typing import Any, Dict, List, Optional
import sqlite3

class MetadataStore:
    def __init__(self, path: str):
        # Disable thread-origin check so the connection can be used across threads.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                vec_id   INTEGER PRIMARY KEY,
                id       TEXT NOT NULL UNIQUE,
                metadata TEXT
            )
            """
        )
        self._conn.commit()

    def insert(self, id: str, vec_id: int, metadata: Optional[dict] = None):
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO metadata (id, vec_id, metadata)
            VALUES (?, ?, ?)
            """,
            (id, vec_id, json.dumps(metadata) if metadata else None),
        )
        self._conn.commit()

    def exists(self, id: str) -> bool:
        cursor = self._conn.cursor()
        cursor.execute("SELECT 1 FROM metadata WHERE id = ?", (id,))
        return cursor.fetchone() is not None

    def get(self, vec_id: int) -> Optional[Dict[str, Any]]:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, metadata
            FROM metadata
            WHERE vec_id = ?
            """,
            (vec_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        id, metadata_json = row
        metadata = json.loads(metadata_json) if metadata_json else None
        return {
            "id": id,
            "vec_id": vec_id,
            "metadata": metadata,
        }

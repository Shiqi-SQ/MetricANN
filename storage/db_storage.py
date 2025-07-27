import sqlite3
from typing import List
from storage.backend import StorageBackend

class DBStorage(StorageBackend):
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS storage (key TEXT PRIMARY KEY, data BLOB)")
        self.conn.commit()

    def add_item(self, key: str, data: bytes) -> None:
        self.conn.execute("REPLACE INTO storage (key, data) VALUES (?, ?)", (key, data))
        self.conn.commit()

    def get_item(self, key: str) -> bytes:
        cursor = self.conn.execute("SELECT data FROM storage WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def list_items(self) -> List[str]:
        cursor = self.conn.execute("SELECT key FROM storage")
        return [row[0] for row in cursor.fetchall()]

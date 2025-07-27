import os
from typing import List
from storage.backend import StorageBackend

class LocalStorage(StorageBackend):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def add_item(self, key: str, data: bytes) -> None:
        path = os.path.join(self.base_dir, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def get_item(self, key: str) -> bytes:
        path = os.path.join(self.base_dir, key)
        with open(path, 'rb') as f:
            return f.read()

    def list_items(self) -> List[str]:
        items = []
        for root, _, files in os.walk(self.base_dir):
            for fname in files:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, self.base_dir)
                items.append(rel)
        return items

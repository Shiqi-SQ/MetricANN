from abc import ABC, abstractmethod
from typing import List

class StorageBackend(ABC):
    """
    抽象存储后端接口，任何具体存储实现（文件系统、数据库、云存储）都应继承此类并实现以下方法。
    """

    @abstractmethod
    def add_item(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_item(self, key: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def list_items(self) -> List[str]:
        raise NotImplementedError

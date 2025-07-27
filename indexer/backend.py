from abc import ABC, abstractmethod
import numpy as np

class IndexBackend(ABC):
    @abstractmethod
    def add(self, label: str, vector: np.ndarray):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def search(self, vector: np.ndarray, k: int):
        pass

import hnswlib
import numpy as np
from indexer.backend import IndexBackend

class HNSWIndex(IndexBackend):
    def __init__(self, dim, space='l2', M=16, ef_construction=200, ef=50):
        self.dim = dim
        self.space = space
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.labels = []
        self.vectors = []
        self.index = None

    def add(self, label: str, vector: np.ndarray):
        self.labels.append(label)
        self.vectors.append(vector.astype('float32'))

    def build(self):
        num_elements = len(self.vectors)
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=num_elements, ef_construction=self.ef_construction, M=self.M)
        data = np.stack(self.vectors)
        self.index.add_items(data, np.arange(num_elements))
        self.index.set_ef(self.ef)

    def save(self, path: str):
        self.index.save_index(path + '.bin')
        with open(path + '.labels', 'w') as f:
            for label in self.labels:
                f.write(label + '\n')

    def load(self, path: str):
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.load_index(path + '.bin')
        self.index.set_ef(self.ef)
        with open(path + '.labels', 'r') as f:
            self.labels = [line.strip() for line in f]

    def search(self, vector: np.ndarray, k: int):
        vect = vector.astype('float32').reshape(1, -1)
        labels, distances = self.index.knn_query(vect, k=k)
        results = []
        for lbl_idx, dist in zip(labels[0], distances[0]):
            results.append((self.labels[lbl_idx], float(dist)))
        return results

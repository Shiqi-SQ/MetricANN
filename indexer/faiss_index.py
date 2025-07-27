import faiss
import numpy as np
from indexer.backend import IndexBackend

class FaissIndex(IndexBackend):
    def __init__(self, dim, nlist, nprobe):
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        self.vectors = []
        self.labels = []

    def add(self, label: str, vector: np.ndarray):
        self.vectors.append(vector.astype('float32'))
        self.labels.append(label)

    def build(self):
        data = np.stack(self.vectors)
        self.index.train(data)
        self.index.add(data)
        self.index.nprobe = self.nprobe

    def save(self, path: str):
        faiss.write_index(self.index, path + '.idx')
        with open(path + '.labels', 'w') as f:
            for label in self.labels:
                f.write(label + '\n')

    def load(self, path: str):
        self.index = faiss.read_index(path + '.idx')
        with open(path + '.labels', 'r') as f:
            self.labels = [line.strip() for line in f]
        self.index.nprobe = self.nprobe

    def search(self, vector: np.ndarray, k: int):
        vect = vector.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(vect, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((self.labels[idx], float(dist)))
        return results

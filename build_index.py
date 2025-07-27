import os
import numpy as np
from indexer.faiss_index import FaissIndex
from config import EMBED_DIM, INDEX_PARAMS

EMBED_ROOT = 'embeddings'
INDEX_DIR = 'index'
INDEX_PATH = os.path.join(INDEX_DIR, 'main_index')

os.makedirs(INDEX_DIR, exist_ok=True)

all_vec_paths = []
for label in os.listdir(EMBED_ROOT):
    for fn in os.listdir(os.path.join(EMBED_ROOT, label)):
        if fn.endswith('.npy'):
            all_vec_paths.append(os.path.join(EMBED_ROOT, label, fn))
total = len(all_vec_paths)

nlist_orig = INDEX_PARAMS.get('nlist', 1000)
nlist = min(nlist_orig, max(1, total // 10))

print(f"Total embeddings: {total}, using nlist={nlist}, nprobe={INDEX_PARAMS.get('nprobe')}")

idx = FaissIndex(dim=EMBED_DIM, nlist=nlist, nprobe=INDEX_PARAMS.get('nprobe'))
for path in all_vec_paths:
    label = os.path.basename(os.path.dirname(path))
    vec = np.load(path)
    idx.add(label, vec)

idx.build()
idx.save(INDEX_PATH)

print("Index built and saved to", INDEX_PATH + '.idx')

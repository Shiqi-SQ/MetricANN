import os

DATA_DIR = os.getenv('DATA_DIR', './dataset')
EMBED_DIR = os.getenv('EMBED_DIR', './embeddings')
INDEX_DIR = os.getenv('INDEX_DIR', './index')

MODEL_BACKBONE = os.getenv('MODEL_BACKBONE', 'resnet50')
EMBED_DIM = int(os.getenv('EMBED_DIM', 512))

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-4))
MARGIN = float(os.getenv('MARGIN', 0.2))

INDEX_TYPE = os.getenv('INDEX_TYPE', 'faiss')
INDEX_PARAMS = {
    'nlist': int(os.getenv('INDEX_NLIST', 1000)),
    'nprobe': int(os.getenv('INDEX_NPROBE', 10))
}
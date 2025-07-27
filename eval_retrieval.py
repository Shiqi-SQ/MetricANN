import os, numpy as np
from indexer.faiss_index import FaissIndex
from model.embedding_model import EmbeddingNet
import torch
from utils.metrics import top_k_accuracy, mean_reciprocal_rank
from torchvision import transforms
from PIL import Image

device = 'cuda'
model = EmbeddingNet().to(device)
model.load_state_dict(torch.load('model.pt', map_location=device))
model.eval()

idx = FaissIndex(dim=512, **INDEX_PARAMS)
idx.load('index/main_index')

y_true, y_preds = [], []
transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
for label in os.listdir('test_set'):
    for fn in os.listdir(f'test_set/{label}'):
        img = Image.open(f'test_set/{label}/{fn}').convert('RGB')
        emb = model(transform(img).unsqueeze(0).to(device)).cpu().numpy()[0]
        res = idx.search(emb, k=5)
        preds = [r[0] for r in res]
        y_true.append(label); y_preds.append(preds)

print("Top‑1:", top_k_accuracy(y_true, y_preds, k=1))
print("Top‑5:", top_k_accuracy(y_true, y_preds, k=5))
print("MRR   :", mean_reciprocal_rank(y_true, y_preds))

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from model.embedding_model import EmbeddingNet
from config import DATA_DIR, EMBED_DIR, INDEX_TYPE, INDEX_PARAMS, EMBED_DIM
from indexer.faiss_index import FaissIndex
from indexer.hnsw_index import HNSWIndex

def update_index(model_path, index_path, new_data_dir, device='cpu'):
    model = EmbeddingNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    if INDEX_TYPE == 'faiss':
        index = FaissIndex(dim=EMBED_DIM, nlist=INDEX_PARAMS['nlist'], nprobe=INDEX_PARAMS['nprobe'])
    else:
        index = HNSWIndex(dim=EMBED_DIM)
    index.load(index_path)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    for label in os.listdir(new_data_dir):
        class_dir = os.path.join(new_data_dir, label)
        if not os.path.isdir(class_dir): continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            path = os.path.join(class_dir, fname)
            img = Image.open(path).convert('RGB')
            emb = model(transform(img).unsqueeze(0).to(device)).cpu().numpy()[0]
            if INDEX_TYPE == 'faiss':
                index.index.add(emb.astype('float32').reshape(1, -1))
                index.labels.append(label)
            else:
                idx_id = len(index.labels)
                index.index.add_items(emb.astype('float32').reshape(1, -1), np.array([idx_id]))
                index.labels.append(label)
    index.save(index_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--new_data', default=DATA_DIR)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    update_index(args.model, args.index, args.new_data, args.device)
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from config import INDEX_TYPE, INDEX_PARAMS, EMBED_DIM
from model.embedding_model import EmbeddingNet
from indexer.faiss_index import FaissIndex
from indexer.hnsw_index import HNSWIndex

def load_index(path):
    if INDEX_TYPE == 'faiss':
        idx = FaissIndex(dim=EMBED_DIM, nlist=INDEX_PARAMS['nlist'], nprobe=INDEX_PARAMS['nprobe'])
    else:
        idx = HNSWIndex(dim=EMBED_DIM)
    idx.load(path)
    return idx

def extract_embedding(model_path, image_path, device):
    model = EmbeddingNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).cpu().numpy()[0]
    return emb

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--index', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    emb = extract_embedding(args.model, args.image, args.device)
    idx = load_index(args.index)
    results = idx.search(emb, args.k)
    for label, dist in results:
        print(f"{label}\t{dist:.4f}")

if __name__ == '__main__':
    main()

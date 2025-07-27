import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from config import DATA_DIR, EMBED_DIR, BATCH_SIZE
from model.embedding_model import EmbeddingNet

def extract_embeddings(model_path, device='cpu'):
    model = EmbeddingNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    os.makedirs(EMBED_DIR, exist_ok=True)
    samples = []
    for label in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append((label, os.path.join(class_dir, fname)))
    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i+BATCH_SIZE]
        imgs = []
        keys = []
        for label, path in batch:
            img = Image.open(path).convert('RGB')
            imgs.append(transform(img).unsqueeze(0))
            keys.append((label, os.path.splitext(os.path.basename(path))[0]))
        imgs = torch.cat(imgs, dim=0).to(device)
        with torch.no_grad():
            embs = model(imgs).cpu().numpy()
        for emb, (label, name) in zip(embs, keys):
            out_dir = os.path.join(EMBED_DIR, label)
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, f"{name}.npy"), emb)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    extract_embeddings(args.model, args.device)

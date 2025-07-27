import torch
from data_loader import get_dataloader
from model.embedding_model import EmbeddingNet

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 1. 初始化 DataLoader
    loader = get_dataloader(shuffle=False, batch_size=8, num_workers=2)
    imgs, labels = next(iter(loader))
    print("Batch images shape:", imgs.shape)            # 期望 (8, 3, 224, 224)
    print("Batch labels:", labels)                      # 8 个标签字符串

    # 2. 模型前向
    model = EmbeddingNet().to(device)
    model.eval()
    with torch.no_grad():
        embs = model(imgs.to(device))
    print("Embeddings shape:", embs.shape)             # 期望 (8, EMBED_DIM)

if __name__ == '__main__':
    main()

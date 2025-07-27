import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from config import DATA_DIR, BATCH_SIZE

class PlushieDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, transform=None):
        self.data_dir = data_dir
        default_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        self.transform = transform or default_transform
        self.samples = []
        for label in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, label)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(labels)

def get_dataloader(transform=None, batch_size=BATCH_SIZE, shuffle=True, num_workers=4):
    ds = PlushieDataset(transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

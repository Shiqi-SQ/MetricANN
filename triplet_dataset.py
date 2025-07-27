import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import DATA_DIR

class TripletDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, transform=None):
        self.data_dir = data_dir
        default_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        self.transform = transform or default_transform

        self.samples = {}
        for label in os.listdir(data_dir):
            d = os.path.join(data_dir, label)
            if not os.path.isdir(d): continue
            imgs = [os.path.join(d, f)
                    for f in os.listdir(d)
                    if f.lower().endswith(('.jpg','.png','.jpeg'))]
            if len(imgs) >= 2:
                self.samples[label] = imgs

        self.labels = list(self.samples.keys())

    def __len__(self):
        return sum(len(v) for v in self.samples.values())

    def __getitem__(self, idx):
        anchor_label = random.choice(self.labels)
        imgs = self.samples[anchor_label]
        anchor_path, positive_path = random.sample(imgs, 2)
        neg_label = random.choice([l for l in self.labels if l != anchor_label])
        negative_path = random.choice(self.samples[neg_label])

        anchor = self.transform(Image.open(anchor_path).convert('RGB'))
        positive = self.transform(Image.open(positive_path).convert('RGB'))
        negative = self.transform(Image.open(negative_path).convert('RGB'))

        return anchor, positive, negative

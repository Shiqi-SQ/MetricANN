import torch
import torch.nn as nn
import torchvision.models as models
from config import MODEL_BACKBONE, EMBED_DIM

class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        if MODEL_BACKBONE == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {MODEL_BACKBONE}")
        self.backbone = backbone
        self.fc = nn.Linear(in_features, EMBED_DIM)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

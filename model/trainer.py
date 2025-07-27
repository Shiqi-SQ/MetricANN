import torch
import torch.optim as optim
from torch.nn import TripletMarginLoss
from model.embedding_model import EmbeddingNet

class TripletTrainer:
    def __init__(self, device, margin=0.2, lr=1e-4):
        self.device = device
        self.model = EmbeddingNet().to(device)
        self.loss_fn = TripletMarginLoss(margin=margin)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.start_epoch = 1

    def train_epoch(self, train_loader, logger):
        self.model.train()
        total_loss = 0.0
        for batch in train_loader:
            anchor, positive, negative = batch
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            anc_emb = self.model(anchor)
            pos_emb = self.model(positive)
            neg_emb = self.model(negative)
            loss = self.loss_fn(anc_emb, pos_emb, neg_emb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_loader)
        logger.info(f"Epoch {self.current_epoch}: avg loss {avg:.4f}")
        return avg

    def save_checkpoint(self, path):
        ckpt = {
            'epoch': self.current_epoch,
            'model_state': self.model.state_dict(),
            'opt_state': self.optimizer.state_dict(),
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['opt_state'])
        self.start_epoch = ckpt['epoch'] + 1
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from triplet_dataset import TripletDataset
from model.trainer import TripletTrainer
from config import BATCH_SIZE, DATA_DIR
from utils.logging import get_logger

def validate(trainer, val_loader, device):
    trainer.model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for anchor, positive, negative in val_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            anc_emb = trainer.model(anchor)
            pos_emb = trainer.model(positive)
            neg_emb = trainer.model(negative)
            loss = trainer.loss_fn(anc_emb, pos_emb, neg_emb)
            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)

def main():
    parser = argparse.ArgumentParser(description="Plushie Recognizer Training")
    parser.add_argument('--epochs', type=int, default=10, help="Maximum number of epochs")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader workers")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Training device ('cpu' or 'cuda')")
    parser.add_argument('--model_out', default='model.pt', help="Path to save final model")
    parser.add_argument('--ckpt_dir', default='checkpoints', help="Directory for checkpoints")
    parser.add_argument('--resume', default=None, help="Checkpoint path to resume from")
    parser.add_argument('--val_data', default=None, help="Directory for validation dataset")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    logger = get_logger("train")

    # Startup summary
    logger.info("===== Training Configuration =====")
    logger.info(f"Device: {args.device} (CUDA available: {torch.cuda.is_available()})")
    if args.device.startswith('cuda'):
        logger.info(f"CUDA device count: {torch.cuda.device_count()}, current device: {torch.cuda.current_device()}")
        torch.cuda.set_device(torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
    logger.info(f"Max epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Num workers: {args.num_workers}")
    logger.info(f"Data directory: {DATA_DIR}")
    if args.val_data:
        logger.info(f"Validation directory: {args.val_data}")
    logger.info(f"Patience: {args.patience}")

    # Dataset stats
    train_dataset = TripletDataset(data_dir=DATA_DIR)
    num_classes = len(train_dataset.labels)
    total_images = sum(len(v) for v in train_dataset.samples.values())
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Total images (for triplets): {total_images}")

    logger.info(f"Model output path: {args.model_out}")
    logger.info(f"Checkpoint directory: {args.ckpt_dir}")
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
    logger.info("==================================")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = None
    if args.val_data:
        val_dataset = TripletDataset(data_dir=args.val_data)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    # Trainer
    trainer = TripletTrainer(device=args.device)
    logger.info(f"Model parameters on device: {next(trainer.model.parameters()).device}")
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Early stopping setup
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    for epoch in range(trainer.start_epoch, args.epochs + 1):
        trainer.current_epoch = epoch
        logger.info(f"Start epoch {epoch}/{args.epochs}")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        total_loss = 0.0
        for anchor, positive, negative in pbar:
            anchor = anchor.to(args.device)
            positive = positive.to(args.device)
            negative = negative.to(args.device)
            loss = trainer.loss_fn(
                trainer.model(anchor),
                trainer.model(positive),
                trainer.model(negative)
            )
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch} done, avg train loss {avg_train_loss:.4f}")

        # Validation & early stopping
        if val_loader:
            avg_val_loss = validate(trainer, val_loader, args.device)
            logger.info(f"Epoch {epoch} validation loss {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # save best model
                best_model_path = os.path.join(args.ckpt_dir, 'best_model.pt')
                torch.save(trainer.model.state_dict(), best_model_path)
                logger.info(f"New best model saved to {best_model_path}")
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement for {epochs_no_improve}/{args.patience} epochs")
                if epochs_no_improve >= args.patience:
                    logger.info("Early stopping triggered")
                    break

        # Save checkpoint at each epoch
        ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_epoch{epoch}.pt")
        trainer.save_checkpoint(ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

    # Save final model if early stopping didn't save best
    if not args.val_data:
        torch.save(trainer.model.state_dict(), args.model_out)
        logger.info(f"Training complete, final model saved to {args.model_out}")

if __name__ == '__main__':
    main()

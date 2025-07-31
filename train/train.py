# train/train.py

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from dataset.patch_dataset import MultiModalPatchDataset
from utils.collate import custom_collate
from models.model import MultiModalClassifier
from loss.focal_loss import FocalLoss
from train.train_utils import train_phase, save_metrics

def main(args):
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load CSV
    train_df = pd.read_csv(args.train_csv)
    train_cases = train_df["case_id"].tolist()
    train_labels = train_df["label"].tolist()

    # Dataset and DataLoader
    train_set = MultiModalPatchDataset(args.file_dir, train_cases, train_labels)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=custom_collate, num_workers=args.num_workers)

    # Model
    model = MultiModalClassifier(freeze_a=True, freeze_b=False, patch_batch_size=480).to(device)

    # Class weights for focal loss
    class_counts = torch.tensor([812, 259], dtype=torch.float32)
    weights = (1.0 / class_counts)
    weights = weights / weights.sum()
    weights = weights.to(device)

    loss_fn = FocalLoss(alpha=weights)
    scaler = GradScaler()

    # Phase 1: Train encoder B
    train_phase(model, train_loader, loss_fn, scaler, epochs=8, phase=1)

    # Phase 2: Train encoder A
    for p in model.encoder_b.parameters(): p.requires_grad = False
    for p in model.encoder_a.parameters(): p.requires_grad = True
    train_phase(model, train_loader, loss_fn, scaler, epochs=8, phase=2)

    # Phase 3: Joint fine-tuning
    for p in model.parameters(): p.requires_grad = True
    train_phase(model, train_loader, loss_fn, scaler, val_loader=None, epochs=15, phase=3, final=True)

    # Save final model
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    torch.save(model.state_dict(), args.model_output)
    print(f"✅ Training complete. Model saved to: {args.model_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MPMRecNet model")

    parser.add_argument("--train_csv", type=str, default="data/train.csv",
                        help="Path to training CSV file")
    parser.add_argument("--file_dir", type=str, default="data/allpatch512tensor",
                        help="Directory containing patch tensors")
    parser.add_argument("--model_output", type=str, default="model/final_model.pt",
                        help="Path to save the trained model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of DataLoader workers")

    args = parser.parse_args()
    main(args)

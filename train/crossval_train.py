# train/crossval_train.py

import os
import argparse
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from dataset.patch_dataset import MultiModalPatchDataset
from utils.collate import custom_collate
from models.model import MultiModalClassifier
from loss.focal_loss import FocalLoss
from train.train_utils import train_phase, evaluate, save_metrics


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.train_csv)
    case_ids = df["case_id"].tolist()
    labels = df["label"].tolist()
    labels_np = np.array(labels)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    class_counts = torch.tensor([812, 259], dtype=torch.float32)
    weights = (1.0 / class_counts)
    weights = weights / weights.sum()
    weights = weights.to(device)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(case_ids, labels_np)):
        print(f"\n=== Fold {fold + 1}/{args.folds} ===")

        train_cases = [case_ids[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_cases = [case_ids[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        train_set = MultiModalPatchDataset(args.file_dir, train_cases, train_labels)
        val_set = MultiModalPatchDataset(args.file_dir, val_cases, val_labels)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=custom_collate, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                                collate_fn=custom_collate, num_workers=args.num_workers)

        model = MultiModalClassifier(freeze_a=True, freeze_b=False, patch_batch_size=480).to(device)
        loss_fn = FocalLoss(alpha=weights)
        scaler = GradScaler()

        # Phase 1: encoder B
        train_phase(model, train_loader, loss_fn, scaler, epochs=8, phase=1)

        # Phase 2: encoder A
        for p in model.encoder_b.parameters(): p.requires_grad = False
        for p in model.encoder_a.parameters(): p.requires_grad = True
        train_phase(model, train_loader, loss_fn, scaler, epochs=8, phase=2)

        # Phase 3: joint
        for p in model.parameters(): p.requires_grad = True
        train_phase(model, train_loader, loss_fn, scaler, val_loader=val_loader, epochs=15, phase=3, final=True)

        # Evaluation + save
        metrics = evaluate(model, val_loader)
        metrics["fold"] = fold + 1

        os.makedirs(args.output_dir, exist_ok=True)
        save_metrics([metrics], os.path.join(args.output_dir, f"fold{fold+1}_metrics.csv"))
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_fold{fold+1}.pt"))
        fold_metrics.append(metrics)

    # Save all fold metrics
    save_metrics(fold_metrics, os.path.join(args.output_dir, "fold_metrics.csv"))
    print("✅ All folds completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 10-fold cross-validation for MPMRecNet")
    parser.add_argument("--train_csv", type=str, default="data/train.csv",
                        help="Path to training CSV")
    parser.add_argument("--file_dir", type=str, default="data/allpatch512tensor",
                        help="Path to patch tensor directory")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpointsnew",
                        help="Directory to save models and metrics")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=10, help="Dataloader worker threads")
    parser.add_argument("--folds", type=int, default=10, help="Number of cross-validation folds")

    args = parser.parse_args()
    main(args)

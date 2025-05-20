# train/crossval_train.py

import os
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

pt_dir = "data/allpatch512tensor"
train_csv = "data/train.csv"
df = pd.read_csv(train_csv)

case_ids = df["case_id"].tolist()
labels = df["label"].tolist()
labels_np = np.array(labels)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
class_counts = torch.tensor([812, 259], dtype=torch.float32)
weights = (1.0 / class_counts)
weights = weights / weights.sum()
weights = weights.cuda()

fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(case_ids, labels_np)):
    print(f"=== Fold {fold + 1}/10 ===")

    train_cases = [case_ids[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_cases = [case_ids[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_set = MultiModalPatchDataset(pt_dir, train_cases, train_labels)
    val_set = MultiModalPatchDataset(pt_dir, val_cases, val_labels)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=custom_collate, num_workers=10)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers=10)

    model = MultiModalClassifier(freeze_a=True, freeze_b=False, patch_batch_size=480).cuda()
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

    metrics = evaluate(model, val_loader)
    metrics["fold"] = fold + 1
    save_metrics([metrics], f"outputs/checkpointsnew/fold{fold+1}_metrics.csv")
    torch.save(model.state_dict(), f"outputs/checkpointsnew/model_fold{fold+1}_best.pt")

    fold_metrics.append(metrics)

save_metrics(fold_metrics, "outputs/checkpointsnew/fold_metrics.csv")
print("✅ All folds completed.")

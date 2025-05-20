# train/train.py

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report, roc_auc_score

from dataset.patch_dataset import MultiModalPatchDataset
from utils.collate import custom_collate
from models.model import MultiModalClassifier
from loss.focal_loss import FocalLoss
from train.train_utils import train_phase, evaluate, save_metrics

train_csv = "data/train.csv"
val_csv = "data/val.csv"
pt_dir = "data/allpatch512tensor"

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

train_cases = train_df["case_id"].tolist()
train_labels = train_df["label"].tolist()
val_cases = val_df["case_id"].tolist()
val_labels = val_df["label"].tolist()

train_set = MultiModalPatchDataset(pt_dir, train_cases, train_labels)
val_set = MultiModalPatchDataset(pt_dir, val_cases, val_labels)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=custom_collate, num_workers=10)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers=10)

model = MultiModalClassifier(freeze_a=True, freeze_b=False, patch_batch_size=480).cuda()

class_counts = torch.tensor([812, 259], dtype=torch.float32)
weights = (1.0 / class_counts)
weights = weights / weights.sum()
weights = weights.cuda()

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
train_phase(model, train_loader, loss_fn, scaler, val_loader=val_loader, epochs=15, phase=3, final=True)

# Final evaluation
metrics = evaluate(model, val_loader)
save_metrics([metrics], "outputs/finalmodel_metrics.csv")
torch.save(model.state_dict(), "outputs/final_model.pt")

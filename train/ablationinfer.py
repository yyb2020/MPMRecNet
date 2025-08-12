#train/ablationinfer.py


import os, csv, torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.model import MultiModalClassifier
from dataset.patch_dataset import MultiModalPatchDataset
from utils.collate import custom_collate
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CSV
val_df = pd.read_csv("val.csv")
val_cases = val_df["case_id"].tolist()
val_labels = val_df["label"].tolist()

# Dataset and DataLoader
val_set = MultiModalPatchDataset("file_dir", val_cases, val_labels)
val_loader = DataLoader(val_set, batch_size=1, shuffle=True,
                          collate_fn=custom_collate, num_workers=4)


torch.cuda.empty_cache()
model = MultiModalClassifier(freeze_a=False, freeze_b=False, patch_batch_size=480).cuda()
model.load_state_dict(torch.load("model_path"))
model.eval()

for p in model.encoder_a.parameters():
    p.requires_grad = False
for p in model.encoder_b.parameters():
    p.requires_grad = False


all_probs, all_preds, all_labels = [], [], []
case_ids = []

with torch.no_grad():
    for a, b, y, cid in tqdm(val_loader, desc="[A-only Inference]"):
        if not a or a[0].numel() == 0:
            continue
        a = [x.cuda() for x in a]
        y = y.cuda()

        logits = model.inference_a_only(a)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)



        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        case_ids.extend(cid)

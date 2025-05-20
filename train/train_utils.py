# train/train_utils.py

import torch
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import autocast
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report

def train_phase(model, train_loader, loss_fn, scaler, val_loader=None, epochs=8, phase=1, final=False):
    lr = 1e-4 if not final else 7e-5
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    if final:
        from transformers import get_cosine_schedule_with_warmup
        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"[Phase {phase}] Epoch {epoch+1}/{epochs}", leave=False)
        for a, b, y, _ in loop:
            a = [x.cuda() for x in a]
            b = [x.cuda() for x in b]
            y = y.cuda()

            optimizer.zero_grad()
            with autocast():
                logits = model(a, b)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if final:
                scheduler.step()

            loop.set_postfix(loss=loss.item())

        torch.cuda.empty_cache()

def evaluate(model, val_loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for a, b, y, _ in val_loader:
            a = [x.cuda() for x in a]
            b = [x.cuda() for x in b]
            y = y.cuda()
            logits = model(a, b)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    auc = roc_auc_score(all_labels, all_probs)
    report = classification_report(all_labels, all_preds, output_dict=True)
    return {
        "AUC": auc,
        "F1": report["weighted avg"]["f1-score"],
        "ACC": report["accuracy"],
        "ACC_nonrecur": report["0"]["recall"],
        "ACC_recur": report["1"]["recall"]
    }

def save_metrics(metrics_list, path):
    df = pd.DataFrame(metrics_list)
    df.to_csv(path, index=False)


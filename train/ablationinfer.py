


torch.cuda.empty_cache()
model = MultiModalClassifier(freeze_a=False, freeze_b=False, patch_batch_size=480).cuda()
model.load_state_dict(torch.load(final_model_path))
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

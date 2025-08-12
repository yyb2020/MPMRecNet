def extract_a_only_features(self, a_list):
    agg_a_list = []

    for a_patches in a_list:
        a_feat = self.extract_features_batchwise(self.encoder_a, a_patches)
        agg_a = self.attn_a(a_feat)  # shape: [512]
        agg_a_list.append(agg_a)

    return torch.stack(agg_a_list, dim=0)  # shape: [B, 512]


torch.cuda.empty_cache()
model = MultiModalClassifier(freeze_a=False, freeze_b=False, patch_batch_size=480).cuda()
model.load_state_dict(torch.load(final_model_path))
model.eval()

for p in model.encoder_a.parameters():
    p.requires_grad = False
for p in model.encoder_b.parameters():
    p.requires_grad = False

# ✅ 推理与收集预测结果
all_probs, all_preds, all_labels = [], [], []
case_ids = []
features_a = []

# === 推理（只用模态A） ===
with torch.no_grad():
    for a, b, y, cid in tqdm(val_loader, desc="[A-only Inference]"):
        if not a or a[0].numel() == 0:
            print(f"⚠️ 跳过空样本: {cid[0]}")
            continue
        a = [x.cuda() for x in a]
        y = y.cuda()

        # ✅ logits 推理
        logits = model.inference_a_only(a)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        # ✅ 提取特征
        feats = model.extract_a_only_features(a).cpu()  # [B, 512]

        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        case_ids.extend(cid)
        features_a.append(feats)

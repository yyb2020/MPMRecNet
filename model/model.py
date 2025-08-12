# model/model.py

import torch
import torch.nn as nn
from torchvision.models import maxvit_t, MaxVit_T_Weights
from torch.utils.checkpoint import checkpoint

class PatchAttention(nn.Module):
    """Patch-level attention mechanism."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        x = x.float()
        weights = torch.softmax(self.attn(x), dim=0)
        return torch.sum(weights * x, dim=0)


class ModalAttentionFusion(nn.Module):
    """Attention-based fusion of two modalities."""
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, a, b):
        q = self.q(a).unsqueeze(1)
        kv = torch.stack([a, b], dim=1)
        k = self.k(kv)
        v = self.v(kv)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        fused = (attn @ v).squeeze(1)
        return fused


class MaxViTFeatureExtractor(nn.Module):
    """MaxViT encoder with optional parameter freezing."""
    def __init__(self, freeze: bool = True):
        super().__init__()
        backbone = maxvit_t(weights=MaxVit_T_Weights.IMAGENET1K_V1)
        backbone.classifier = nn.Sequential(*list(backbone.classifier.children())[:4])
        self.encoder = backbone

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.training:
            return checkpoint(self.encoder, x)
        else:
            return self.encoder(x)


class MultiModalClassifier(nn.Module):
    """Main multi-modal classifier with dual encoders, attention and fusion."""
    def __init__(self, freeze_a: bool = False, freeze_b: bool = False, patch_batch_size: int = 8):
        super().__init__()
        self.encoder_a = MaxViTFeatureExtractor(freeze=freeze_a)
        self.encoder_b = MaxViTFeatureExtractor(freeze=freeze_b)
        self.attn_a = PatchAttention(512)
        self.attn_b = PatchAttention(512)
        self.fusion = ModalAttentionFusion(512)

        self.cls = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        self.patch_batch_size = patch_batch_size

    def extract_features_batchwise(self, encoder, patches):
        features = []
        grad_context = torch.no_grad() if not self.training else torch.enable_grad()

        with grad_context:
            for i in range(0, patches.size(0), self.patch_batch_size):
                batch = patches[i:i + self.patch_batch_size].cuda(non_blocking=True)
                with torch.cuda.amp.autocast(enabled=self.training):
                    out = encoder(batch)
                features.append(out)
                del batch, out
                torch.cuda.empty_cache()

        return torch.cat(features, dim=0)

    def forward(self, a_list, b_list):
        agg_a_list, agg_b_list = [], []

        for a_patches, b_patches in zip(a_list, b_list):
            a_feat = self.extract_features_batchwise(self.encoder_a, a_patches)
            b_feat = self.extract_features_batchwise(self.encoder_b, b_patches)

            agg_a = self.attn_a(a_feat)
            agg_b = self.attn_b(b_feat)

            agg_a_list.append(agg_a)
            agg_b_list.append(agg_b)

        a_tensor = torch.stack(agg_a_list, dim=0)
        b_tensor = torch.stack(agg_b_list, dim=0)

        fused = self.fusion(a_tensor, b_tensor)
        return self.cls(fused)



    
    def inference_a_only(self, a_list):
        agg_a_list = []
        for a_patches in a_list:
            if a_patches is None or a_patches.numel() == 0:
                continue
            a_feat = self.extract_features_batchwise(self.encoder_a, a_patches)
            agg_a = self.attn_a(a_feat)
            agg_a_list.append(agg_a)
    
        if not agg_a_list:
            raise RuntimeError("❌ No A patch")
    
        agg_a_tensor = torch.stack(agg_a_list, dim=0)
        logits = self.cls(agg_a_tensor)  
        return logits


    

    def inference_b_only(self, b_list):
        agg_b_list = []
        for b_patches in b_list:
            if b_patches is None or b_patches.numel() == 0:
                continue
            b_feat = self.extract_features_batchwise(self.encoder_b, b_patches)
            agg_b = self.attn_b(b_feat)
            agg_b_list.append(agg_b)

        if not agg_b_list:
            raise RuntimeError("❌ No B patch")

        agg_b_tensor = torch.stack(agg_b_list, dim=0)
        logits = self.cls(agg_b_tensor)  
        return logits



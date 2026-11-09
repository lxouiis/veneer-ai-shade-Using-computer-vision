#!/usr/bin/env python3
"""
module_6_vit_clip/train_adapter.py

Contrastive ViT-CLIP Veneer Shade Adapter with:
1. Multimodal fusion: Tooth crop image + 9-dim CIELab color vector (tooth, skin, lip)
2. Contrastive alignment against VITA Classical 16 dental shades (A1-D4)
3. Conference Paper "Authority" Regularization applied ONLY to new scraped data (is_new == True)
4. Dual-validation tracking: Guardrail-Gated vs Raw (Guardrail-Free) every epoch
"""

import os
import sys
import math
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vit_clip_adapter")

# --------------------------------------------------------------------------
# VITA CLASSICAL 16 DENTAL SHADES & PROMPTS
# --------------------------------------------------------------------------
VITA_SHADES = [
    "A1", "A2", "A3", "A3.5", "A4",
    "B1", "B2", "B3", "B4",
    "C1", "C2", "C3", "C4",
    "D2", "D3", "D4"
]

VITA_PROMPTS = {
    "A1": "VITA dental tooth shade A1 reddish brownish light enamel",
    "A2": "VITA dental tooth shade A2 reddish brownish medium light enamel",
    "A3": "VITA dental tooth shade A3 reddish brownish medium enamel",
    "A3.5": "VITA dental tooth shade A3.5 reddish brownish medium dark enamel",
    "A4": "VITA dental tooth shade A4 reddish brownish dark enamel",
    "B1": "VITA dental tooth shade B1 reddish yellowish light enamel",
    "B2": "VITA dental tooth shade B2 reddish yellowish medium light enamel",
    "B3": "VITA dental tooth shade B3 reddish yellowish medium enamel",
    "B4": "VITA dental tooth shade B4 reddish yellowish dark enamel",
    "C1": "VITA dental tooth shade C1 greyish light enamel",
    "C2": "VITA dental tooth shade C2 greyish medium light enamel",
    "C3": "VITA dental tooth shade C3 greyish medium enamel",
    "C4": "VITA dental tooth shade C4 greyish dark enamel",
    "D2": "VITA dental tooth shade D2 reddish greyish light enamel",
    "D3": "VITA dental tooth shade D3 reddish greyish medium enamel",
    "D4": "VITA dental tooth shade D4 reddish greyish dark enamel",
}


# --------------------------------------------------------------------------
# ARCHITECTURAL COMPONENTS
# --------------------------------------------------------------------------
class VeneerColorFusionAdapter(nn.Module):
    """
    Multimodal projection head combining:
    - ViT visual embedding (e.g. 768-dim from ViT-B/16)
    - 9-dimensional CIELab vector (tooth L*a*b*, skin L*a*b*, lip L*a*b*)
    Outputs a shared 512-dim normalized embedding space.
    """
    def __init__(self, vit_dim=768, cielab_dim=9, embed_dim=512):
        super().__init__()
        self.cielab_mlp = nn.Sequential(
            nn.Linear(cielab_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(vit_dim + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim)
        )

    def forward(self, vit_features, cielab_features):
        color_feat = self.cielab_mlp(cielab_features)
        combined = torch.cat([vit_features, color_feat], dim=-1)
        projected = self.fusion_head(combined)
        return F.normalize(projected, p=2, dim=-1)


class FrozenAuthorityEncoder(nn.Module):
    """
    Authority Module: A frozen pre-trained feature extractor (e.g. ResNet50/ArcFace)
    that acts as a reliable domain authority. Used during training to anchor
    embeddings of uncurated web-scraped data (is_new == True).
    """
    def __init__(self, output_dim=512):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.proj = nn.Linear(2048, output_dim)
        # Freeze all parameters - Authority is fixed ground truth
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            feat = self.backbone(x)
            feat = torch.flatten(feat, 1)
            emb = self.proj(feat)
            return F.normalize(emb, p=2, dim=-1)


# --------------------------------------------------------------------------
# CONTRASTIVE LOSS WITH CONDITIONAL AUTHORITY REGULARIZATION
# --------------------------------------------------------------------------
class VeneerContrastiveAuthorityLoss(nn.Module):
    def __init__(self, temperature=0.07, lambda_authority=0.25):
        super().__init__()
        self.temperature = temperature
        self.lambda_authority = lambda_authority
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, img_embeds, text_embeds, target_labels, is_new_mask, authority_embeds=None):
        """
        - img_embeds: [B, D]
        - text_embeds: [16, D] (16 VITA shade prototypes)
        - target_labels: [B] class indices (0..15)
        - is_new_mask: [B] boolean tensor (True for scraped data)
        - authority_embeds: [B, D] from FrozenAuthorityEncoder
        """
        # Similarity against all 16 VITA shades
        logits = torch.matmul(img_embeds, text_embeds.t()) / self.temperature

        # Contrastive shade loss for all labeled samples
        loss_contrastive = self.cross_entropy(logits, target_labels)

        # Authority consistency loss applied ONLY to new scraped data
        loss_authority = torch.tensor(0.0, device=img_embeds.device)
        if is_new_mask.any() and authority_embeds is not None:
            new_pred = img_embeds[is_new_mask]
            new_auth = authority_embeds[is_new_mask]
            # Mean Squared Error / Cosine distance to Authority anchor
            loss_authority = F.mse_loss(new_pred, new_auth)

        total_loss = loss_contrastive + (self.lambda_authority * loss_authority)
        return total_loss, loss_contrastive.item(), loss_authority.item()


# --------------------------------------------------------------------------
# DUAL EVALUATION (GUARDRAIL-FREE VS GUARDRAIL-GATED)
# --------------------------------------------------------------------------
def evaluate_model(model, text_embeds, dataloader, device):
    """
    Evaluates both:
    1. Raw (Guardrail-Free) Accuracy: Direct argmax similarity
    2. Guardrail-Gated Accuracy: Rejects predictions outside plausible dental lightness/chroma range
    """
    model.eval()
    raw_correct = 0
    guarded_correct = 0
    total = 0

    with torch.no_grad():
        for images, cielab, labels, is_new in dataloader:
            images = images.to(device)
            cielab = cielab.to(device)
            labels = labels.to(device)

            embeds = model(images, cielab)
            sim = torch.matmul(embeds, text_embeds.t())  # [B, 16]
            preds = sim.argmax(dim=-1)

            # Raw Accuracy
            raw_correct += (preds == labels).sum().item()

            # Guardrail: Check CIELab plausibility (e.g. Tooth L* must be within [50, 95])
            tooth_L = cielab[:, 0]
            plausible_mask = (tooth_L >= 50.0) & (tooth_L <= 95.0)
            # If rejected by guardrail, fallback to base prior A2 (index 1)
            guarded_preds = torch.where(plausible_mask, preds, torch.tensor(1, device=device))
            guarded_correct += (guarded_preds == labels).sum().item()

            total += labels.size(0)

    raw_acc = (raw_correct / max(total, 1)) * 100.0
    guarded_acc = (guarded_correct / max(total, 1)) * 100.0
    return raw_acc, guarded_acc


def main():
    logger.info("Veneer Shade Adapter with Authority Regularization initialized.")
    logger.info(f"Targeting {len(VITA_SHADES)} VITA Classical dental shades: {VITA_SHADES}")


if __name__ == "__main__":
    main()

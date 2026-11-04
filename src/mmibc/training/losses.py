from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) with optional per-class alpha. ``alpha`` is class-indexed. (The legacy implementation multiplied every example by one scalar alpha, which is a constant rescaling of the loss and does not weight the minority class.)"""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=-1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        loss = -((1.0 - logpt.exp()) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts: np.ndarray) -> torch.Tensor:
    counts = np.asarray(counts, dtype=float)
    w = counts.sum() / (len(counts) * np.clip(counts, 1, None))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def build_loss(tcfg, class_counts: np.ndarray, device: torch.device) -> nn.Module:
    name = tcfg.loss
    ls = float(tcfg.get("label_smoothing", 0.0))
    if not 0.0 <= ls < 0.5:
        raise ValueError(f"training.label_smoothing must be in [0, 0.5), got {ls}")
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=ls)
    w = class_weights(class_counts).to(device)
    if tcfg.sampler == "balanced":
        log.warning("loss=%s together with a balanced sampler corrects class imbalance twice; prefer loss=ce with sampler=balanced, or sampler=none.", name)
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=w, label_smoothing=ls)
    if name == "focal":
        if ls > 0:
            log.warning("training.label_smoothing is ignored by the focal loss")
        return FocalLoss(float(tcfg.focal_gamma), w)
    raise ValueError(f"unknown loss '{name}'")

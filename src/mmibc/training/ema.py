"""Exponential moving average of the trainable weights.

Fine-tuning on a few thousand images makes the validation curve noisy (val AUROC moved 0.84 -> 0.80 -> 0.82 over three consecutive
epochs of a real run), so the "best epoch" is partly luck. Averaging the weights over the last ~1/(1-decay) optimizer steps removes
most of that noise at no cost. The EMA copy is what gets validated, saved as ``best.pt`` and evaluated; the live weights keep training.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema decay must be in (0, 1), got {decay}")
        self.decay, self.updates = float(decay), 0
        self.shadow = {n: p.detach().clone().float() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))  # warm-up: the average follows the weights closely at first
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @contextmanager
    def applied(self, model: nn.Module):
        """Temporarily load the averaged weights into ``model`` (restored on exit, also on error)."""
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    backup[n] = p.detach().clone()
                    p.copy_(self.shadow[n].to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in backup:
                        p.copy_(backup[n])

    def state_dict(self) -> dict:
        return dict(decay=self.decay, updates=self.updates, shadow={n: v.detach().cpu() for n, v in self.shadow.items()})

    def load_state_dict(self, d: dict) -> None:
        missing = set(self.shadow) ^ set(d["shadow"])
        if missing:
            raise RuntimeError(f"EMA state does not match the model's trainable parameters ({len(missing)} differ). Same config?")
        self.updates = int(d["updates"])
        for n, v in d["shadow"].items():
            self.shadow[n].copy_(v.to(self.shadow[n].device))

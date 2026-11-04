"""Atomic checkpoints. Frozen pretrained weights are not stored (they are re-created from the hub); a fingerprint of
them is, so loading into a model built from different pretrained weights fails loudly instead of silently mis-evaluating."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


def frozen_fingerprint(model: nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        if not p.requires_grad:
            s += float(p.detach().double().abs().sum())
    return s


def atomic_save(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # a crash mid-write (Colab disconnect) can never leave a half-written checkpoint


def model_state(model: nn.Module, trainable_only: bool) -> dict:
    if not trainable_only:
        return dict(mode="full", state=model.state_dict(), frozen_fp=None)
    return dict(mode="trainable", state={n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, frozen_fp=frozen_fingerprint(model))


def load_model_state(model: nn.Module, blob: dict, rtol: float = 1e-6) -> None:
    if blob["mode"] == "full":
        model.load_state_dict(blob["state"], strict=True)
        return
    named = dict(model.named_parameters())
    missing = [n for n, p in named.items() if p.requires_grad and n not in blob["state"]]
    extra = [n for n in blob["state"] if n not in named]
    if missing or extra:
        raise RuntimeError(f"checkpoint does not match the model: missing {missing[:3]}, unexpected {extra[:3]}. Same config?")
    with torch.no_grad():
        for n, v in blob["state"].items():
            named[n].copy_(v)
    fp = frozen_fingerprint(model)
    if abs(fp - blob["frozen_fp"]) > rtol * max(1.0, abs(blob["frozen_fp"])):
        raise RuntimeError("frozen backbone weights differ from those the checkpoint was trained with (different pretrained weights or backbone?)")


class EarlyStopping:
    def __init__(self, mode: str = "max", patience: int = 8, min_delta: float = 0.0):
        self.mode, self.patience, self.min_delta = mode, patience, min_delta
        self.best: float | None = None
        self.bad = 0

    def improved(self, v: float) -> bool:
        if v != v:  # NaN never counts as an improvement
            return False
        if self.best is None:
            return True
        return v > self.best + self.min_delta if self.mode == "max" else v < self.best - self.min_delta

    def step(self, v: float) -> bool:
        """Returns True when ``v`` is a new best. Patience counts epochs since the last improvement."""
        if self.improved(v):
            self.best, self.bad = v, 0
            return True
        self.bad += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad >= self.patience

    def state_dict(self) -> dict:
        return dict(best=self.best, bad=self.bad)

    def load_state_dict(self, d: dict) -> None:
        self.best, self.bad = d["best"], d["bad"]

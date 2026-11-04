from __future__ import annotations

import logging
import math
import re

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

NO_DECAY_KEYS = ("bias", "norm", "cls_token", "register_tokens", "position_embeddings", "pos_embed", "mask_token", "layer_scale", "lambda1", "gamma")


def _backbone_param_ids(model: nn.Module) -> set[int]:
    ids: set[int] = set()
    for name in ("backbone_a", "backbone_b"):
        bb = getattr(model, name, None)
        if bb is not None:
            ids |= {id(p) for p in bb.parameters()}
    return ids


_BLOCK_RE = re.compile(r"\.(?:layer|layers|blocks)\.(\d+)\.")


def layer_depth(name: str, n_layers: int) -> int:
    """Depth index of a backbone parameter for layer-wise lr decay: embeddings 0, transformer block i -> i + 1, final norm -> n_layers + 1."""
    m = _BLOCK_RE.search(name)
    if m:
        return int(m.group(1)) + 1
    low = name.lower()
    if "embed" in low or "cls_token" in low or "register" in low or "mask_token" in low or "rope" in low:
        return 0
    return n_layers + 1  # final norm (the only other backbone parameters that can be trainable)


def lr_scale(model: nn.Module, name: str, layer_decay: float) -> float:
    """Layer-wise lr decay factor for parameter ``name`` of ``model``: ``decay ** (n_layers + 1 - depth)``; 1.0 for the head and for CNNs."""
    if layer_decay >= 1.0:
        return 1.0
    for stream in ("backbone_a", "backbone_b"):
        if name.startswith(stream + "."):
            n = getattr(getattr(model, stream), "n_layers", None)
            if not n:
                return 1.0
            return float(layer_decay ** (n + 1 - layer_depth(name[len(stream) + 1:], n)))
    return 1.0


def build_optimizer(model: nn.Module, tcfg) -> torch.optim.Optimizer:
    """AdamW with separate learning rates for backbone and head, and no weight decay on norms / biases / tokens.

    Asserts that EVERY trainable parameter is in the optimizer exactly once. (The legacy multimodal trainer built its
    optimizer over ``fusion_classifier.parameters()`` only, so any backbone layer unfrozen there could never update.)
    """
    bb_ids = _backbone_param_ids(model)
    ld = float(tcfg.get("layer_decay", 1.0))
    if not 0.0 < ld <= 1.0:
        raise ValueError(f"training.layer_decay must be in (0, 1], got {ld}")
    groups: dict[tuple[bool, bool, float], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        decay = p.ndim >= 2 and not any(k in name.lower() for k in NO_DECAY_KEYS)
        is_bb = id(p) in bb_ids
        groups.setdefault((is_bb, decay, round(lr_scale(model, name, ld) if is_bb else 1.0, 8)), []).append(p)
    if ld < 1.0 and not any(k[2] != 1.0 for k in groups):
        log.warning("training.layer_decay=%s has no effect: no trainable transformer-block parameters to scale", ld)
    param_groups = [
        dict(params=ps, lr=float(tcfg.lr_backbone if is_bb else tcfg.lr_head) * scale, weight_decay=float(tcfg.weight_decay) if decay else 0.0,
             base_lr=float(tcfg.lr_backbone if is_bb else tcfg.lr_head) * scale)
        for (is_bb, decay, scale), ps in sorted(groups.items())
    ]
    if not param_groups:
        raise AssertionError("model has no trainable parameters (freeze policy 'all' with no trainable head?)")
    opt = torch.optim.AdamW(param_groups)
    assert_optimizer_covers(model, opt)
    return opt


def assert_optimizer_covers(model: nn.Module, opt: torch.optim.Optimizer) -> None:
    """Every trainable parameter must be in the optimizer exactly once. Call again after ANY later (un)freezing."""
    in_opt = [id(p) for g in opt.param_groups for p in g["params"]]
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if len(in_opt) != len(set(in_opt)):
        raise AssertionError("a parameter appears in the optimizer more than once")
    if set(in_opt) != trainable:
        missing, extra = len(trainable - set(in_opt)), len(set(in_opt) - trainable)
        raise AssertionError(f"optimizer/model mismatch: {missing} trainable parameters are NOT in the optimizer (they would silently never update), {extra} frozen parameters are")


def lr_factor(step: int, total: int, warmup: int, kind: str) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    if kind == "cosine":
        t = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return 1.0


def build_scheduler(opt: torch.optim.Optimizer, tcfg, steps_per_epoch: int):
    """Per-optimizer-step LambdaLR (warm-up + cosine / constant); ReduceLROnPlateau (stepped per epoch) for ``plateau``."""
    if tcfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=tcfg.monitor_mode, factor=0.5, patience=max(1, int(tcfg.patience) // 2))
    total = int(tcfg.epochs) * steps_per_epoch
    warm = int(tcfg.warmup_epochs) * steps_per_epoch
    kind = tcfg.scheduler
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, warm, kind))

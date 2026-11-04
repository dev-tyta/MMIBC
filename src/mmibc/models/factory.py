from __future__ import annotations

import logging

from mmibc.models.backbones import make_backbone
from mmibc.models.fusion import FusionModel

log = logging.getLogger(__name__)


def build_model(mcfg, eager_attention: bool = False) -> FusionModel:
    """Build a FusionModel from ``cfg.model``; applies the backbone freeze policy exactly once per distinct backbone."""
    fusion = mcfg.fusion
    need_a = fusion != "unimodal_b"
    need_b = fusion not in ("unimodal_a", "early")

    def make():
        bb = make_backbone(mcfg.backbone, pretrained=bool(mcfg.pretrained), eager_attention=eager_attention)
        bb.set_trainable(mcfg.freeze, int(mcfg.unfreeze_last_n))
        dp = float(mcfg.get("drop_path", 0.0))
        if dp > 0 and hasattr(bb, "set_drop_path"):
            bb.set_drop_path(dp)
        return bb

    bb_a = make() if need_a else None
    if need_b:
        bb_b = bb_a if (mcfg.shared_backbone and bb_a is not None) else make()
    else:
        bb_b = None
    model = FusionModel(
        bb_a, bb_b, fusion, int(mcfg.n_classes), heads=int(mcfg.xattn_heads), layers=int(mcfg.xattn_layers),
        hidden=int(mcfg.head_hidden), dropout=float(mcfg.dropout),
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log.info("Model %s: %.2fM trainable / %.2fM total parameters", fusion, n_train / 1e6, n_all / 1e6)
    return model

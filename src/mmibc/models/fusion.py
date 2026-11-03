"""The fusion ladder. All variants share the same per-stream feature and the same classifier head, so differences
between them are differences in FUSION only.

    unimodal_a / unimodal_b   one stream only (evaluated on the same pairs as the fusion models)
    early                     both images channel-stacked into ONE backbone
    concat                    [z_a ; z_b] -> head                    (the paper's Eq. 2)
    gated                     Gated Multimodal Unit over projected z_a, z_b
    xattn                     bidirectional token-level cross-attention, then concat
    xattn_gated               token-level cross-attention, then GMU

z = [CLS ; mean(patch tokens)] (2D) for every stream.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mmibc.models.attention import CrossAttentionBlock
from mmibc.models.backbones import ViTBackbone as DinoV2Backbone  # type hint only: any backbone with the same interface works

FUSIONS = ("unimodal_a", "unimodal_b", "early", "concat", "gated", "xattn", "xattn_gated")


class GatedMultimodalUnit(nn.Module):
    """Arevalo et al. 2017: h = z * tanh(Wa a) + (1 - z) * tanh(Wb b),  z = sigmoid(W [a ; b])."""

    def __init__(self, dim: int):
        super().__init__()
        self.fa, self.fb, self.gate = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(2 * dim, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        return z * torch.tanh(self.fa(a)) + (1 - z) * torch.tanh(self.fb(b))


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _pool(cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
    return torch.cat([cls, patches.mean(dim=1)], dim=-1)


class FusionModel(nn.Module):
    def __init__(
        self, backbone_a: DinoV2Backbone | None, backbone_b: DinoV2Backbone | None, fusion: str = "xattn", n_classes: int = 2,
        heads: int = 8, layers: int = 1, hidden: int = 512, dropout: float = 0.3,
    ):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        self.fusion = fusion
        # Pass the same module for both to share weights (CC/MLO views of one modality); None for the unused stream.
        self.backbone_a, self.backbone_b = backbone_a, backbone_b
        ref = backbone_a if backbone_a is not None else backbone_b
        d = ref.embed_dim
        self.dim = d
        if fusion in ("xattn", "xattn_gated"):
            self.blocks_ab = nn.ModuleList([CrossAttentionBlock(d, heads, dropout=dropout / 3) for _ in range(layers)])  # a queries b
            self.blocks_ba = nn.ModuleList([CrossAttentionBlock(d, heads, dropout=dropout / 3) for _ in range(layers)])  # b queries a
        if fusion in ("gated", "xattn_gated"):
            self.proj_a, self.proj_b = nn.Linear(2 * d, d), nn.Linear(2 * d, d)
            self.gmu = GatedMultimodalUnit(d)
            in_dim = d
        elif fusion in ("unimodal_a", "unimodal_b", "early"):
            in_dim = 2 * d
        else:  # concat, xattn
            in_dim = 4 * d
        self.head = MLPHead(in_dim, hidden, n_classes, dropout)

    def _encode(self, which: str, x: torch.Tensor, attn: bool = False):
        bb = self.backbone_a if which == "a" else self.backbone_b
        if bb is None:
            raise RuntimeError(f"fusion='{self.fusion}' has no backbone for stream {which}")
        return bb(x, output_attentions=attn)

    def forward(self, a: torch.Tensor | None, b: torch.Tensor | None = None, return_aux: bool = False):
        aux: dict = {}
        f = self.fusion
        if f == "unimodal_a":
            oa = self._encode("a", a)
            logits, aux["out_a"] = self.head(_pool(oa.cls, oa.patches)), oa
        elif f == "unimodal_b":
            ob = self._encode("b", b)
            logits, aux["out_b"] = self.head(_pool(ob.cls, ob.patches)), ob
        elif f == "early":
            ga, gb = a.mean(dim=1), b.mean(dim=1)
            oa = self._encode("a", torch.stack([ga, gb, 0.5 * (ga + gb)], dim=1))
            logits, aux["out_a"] = self.head(_pool(oa.cls, oa.patches)), oa
        else:
            oa, ob = self._encode("a", a), self._encode("b", b)
            aux["out_a"], aux["out_b"] = oa, ob
            if f in ("concat", "gated"):
                za, zb = _pool(oa.cls, oa.patches), _pool(ob.cls, ob.patches)
            else:  # token-level cross-attention
                ta = torch.cat([oa.cls[:, None], oa.patches], dim=1)
                tb = torch.cat([ob.cls[:, None], ob.patches], dim=1)
                w_ab = w_ba = None
                for i, (bab, bba) in enumerate(zip(self.blocks_ab, self.blocks_ba)):
                    if i == len(self.blocks_ab) - 1 and return_aux:
                        new_a, w_ab = bab(ta, tb, return_weights=True)
                        new_b, w_ba = bba(tb, ta, return_weights=True)
                    else:
                        new_a, new_b = bab(ta, tb), bba(tb, ta)
                    ta, tb = new_a, new_b  # both directions use the previous layer's tokens
                za, zb = _pool(ta[:, 0], ta[:, 1:]), _pool(tb[:, 0], tb[:, 1:])
                if w_ab is not None:
                    # (B, 1+Na, 1+Nb) / (B, 1+Nb, 1+Na): which tokens of the other stream each token attends to
                    aux["attn_a_to_b"], aux["attn_b_to_a"] = w_ab, w_ba
            if f in ("gated", "xattn_gated"):
                fused = self.gmu(self.proj_a(za), self.proj_b(zb))
            else:
                fused = torch.cat([za, zb], dim=-1)
            logits = self.head(fused)
        return (logits, aux) if return_aux else logits

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from mmibc.models.attention import CrossAttentionBlock, assert_query_dependence, attention_rollout
from mmibc.models.backbones import DinoV2Backbone
from mmibc.models.factory import build_model
from mmibc.models.fusion import FUSIONS, GatedMultimodalUnit


class LegacyCrossAttention(nn.Module):
    """Verbatim logic of the legacy module (feature vectors reshaped to length-1 sequences)."""

    def __init__(self, d):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.scale = d ** -0.5

    def forward(self, q, c):
        query, key, value = self.q_proj(q).unsqueeze(1), self.k_proj(c).unsqueeze(1), self.v_proj(c).unsqueeze(1)
        w = F.softmax((query @ key.transpose(-2, -1)) * self.scale, dim=-1)
        return (w @ value).squeeze(1), w


def test_legacy_cross_attention_was_a_noop():
    torch.manual_seed(0)
    ca = LegacyCrossAttention(16)
    q, c = torch.randn(4, 16), torch.randn(4, 16)
    out, w = ca(q, c)
    assert torch.all(w == 1.0)                                          # softmax over one key
    assert torch.allclose(out, ca.v_proj(c))                            # output = v_proj(context)
    assert torch.allclose(ca(torch.randn(4, 16), c)[0], out)            # the query is ignored


def test_new_cross_attention_depends_on_query():
    torch.manual_seed(0)
    blk = CrossAttentionBlock(32, heads=4)
    assert_query_dependence(blk)
    blk.eval()
    q, kv = torch.randn(2, 5, 32), torch.randn(2, 7, 32)
    _, w = blk(q, kv, return_weights=True)
    assert w.shape == (2, 5, 7) and torch.allclose(w.sum(-1), torch.ones(2, 5), atol=1e-5)
    assert not torch.allclose(w[:, 0], w[:, 1])                          # different queries attend differently
    with pytest.raises(AssertionError):                                  # a length-1 kv sequence is the degenerate case
        assert_query_dependence(blk, n_kv=1)


def _cfg(**kw):
    base = dict(backbone="dinov2_tiny_test", pretrained=False, shared_backbone=False, fusion="xattn", n_classes=2, xattn_layers=1,
                xattn_heads=2, head_hidden=16, dropout=0.1, freeze="none", unfreeze_last_n=1)
    base.update(kw)
    return OmegaConf.create(base)


@pytest.mark.parametrize("fusion", FUSIONS)
def test_every_fusion_runs_and_trains(fusion):
    torch.manual_seed(0)
    m = build_model(_cfg(fusion=fusion))
    a, b = torch.randn(3, 3, 56, 56), torch.randn(3, 3, 56, 56)
    out = m(a, b)
    assert out.shape == (3, 2) and torch.isfinite(out).all()
    out.sum().backward()
    trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None for _, p in trainable), "some trainable parameters received no gradient"


def test_fusion_uses_both_streams():
    torch.manual_seed(0)
    for fusion in ("concat", "gated", "xattn", "xattn_gated"):
        m = build_model(_cfg(fusion=fusion)).eval()
        a, b = torch.randn(2, 3, 56, 56), torch.randn(2, 3, 56, 56)
        base = m(a, b)
        assert not torch.allclose(base, m(torch.randn_like(a), b), atol=1e-6), f"{fusion} ignores stream a"
        assert not torch.allclose(base, m(a, torch.randn_like(b)), atol=1e-6), f"{fusion} ignores stream b"


def test_unimodal_builds_only_needed_backbone_and_shared_backbone_is_shared():
    ua = build_model(_cfg(fusion="unimodal_a"))
    assert ua.backbone_b is None and ua.backbone_a is not None
    ub = build_model(_cfg(fusion="unimodal_b"))
    assert ub.backbone_a is None and ub(None, torch.randn(1, 3, 56, 56)).shape == (1, 2)
    sh = build_model(_cfg(fusion="xattn", shared_backbone=True))
    assert sh.backbone_a is sh.backbone_b
    n_shared = sum(p.numel() for p in sh.parameters())
    n_sep = sum(p.numel() for p in build_model(_cfg(fusion="xattn", shared_backbone=False)).parameters())
    assert n_shared < n_sep


def test_freeze_policies():
    full = build_model(_cfg(freeze="all"))
    assert not any(p.requires_grad for p in full.backbone_a.parameters()) and any(p.requires_grad for p in full.head.parameters())
    last = build_model(_cfg(freeze="last_n", unfreeze_last_n=1))
    layers = last.backbone_a.model.encoder.layer
    assert all(p.requires_grad for p in layers[-1].parameters()) and not any(p.requires_grad for p in layers[0].parameters())


def test_aux_outputs_and_gmu():
    m = build_model(_cfg(fusion="xattn")).eval()
    a, b = torch.randn(2, 3, 56, 56), torch.randn(2, 3, 56, 56)
    logits, aux = m(a, b, return_aux=True)
    na = 1 + aux["out_a"].patches.shape[1]
    assert aux["attn_a_to_b"].shape == (2, na, 1 + aux["out_b"].patches.shape[1])
    assert aux["out_a"].grid == (4, 4)
    g = GatedMultimodalUnit(8)
    assert g(torch.randn(3, 8), torch.randn(3, 8)).shape == (3, 8)


def test_backbone_eager_attention_and_rollout():
    bb = DinoV2Backbone("dinov2_tiny_test", pretrained=False, eager_attention=True).eval()
    out = bb(torch.randn(2, 3, 56, 56), output_attentions=True)
    assert out.attentions is not None and len(out.attentions) == 3
    roll = attention_rollout(out.attentions)
    assert roll.shape == (2, 16) and torch.isfinite(roll).all() and (roll >= 0).all()


# ----------------------------------------------------------------------------- backbone families
from mmibc.models.backbones import BACKBONES, make_backbone  # noqa: E402

FAMILIES = [("dinov2_tiny_test", 56, 1), ("dinov2reg_tiny_test", 56, 5), ("dinov3_tiny_test", 64, 5), ("tv_resnet50", 64, 1)]


@pytest.mark.parametrize("name,size,n_prefix", FAMILIES)
def test_backbone_token_layout(name, size, n_prefix):
    bb = make_backbone(name, pretrained=False, eager_attention=not name.startswith("tv")).eval()
    bb.check_input_size(size)
    assert bb.n_prefix == n_prefix
    out = bb(torch.randn(2, 3, size, size), output_attentions=not name.startswith("tv"))
    assert out.patches.shape == (2, out.grid[0] * out.grid[1], bb.embed_dim)       # registers are NOT in the patch sequence
    assert out.grid == (size // bb.patch_size, size // bb.patch_size)
    if out.attentions is not None:
        assert out.attentions[0].shape[-1] == n_prefix + out.patches.shape[1]       # but they are in the attention matrix
        roll = attention_rollout(out.attentions, n_prefix=out.n_prefix)
        assert roll.shape == (2, out.patches.shape[1]) and torch.isfinite(roll).all()
    # non-square inputs follow the grid
    h, w = size, size + bb.patch_size
    assert make_backbone(name, pretrained=False).eval()(torch.randn(1, 3, h, w)).grid == (h // bb.patch_size, w // bb.patch_size)
    with pytest.raises(ValueError):
        bb.check_input_size(size + 1)


@pytest.mark.parametrize("name,size,n_prefix", FAMILIES)
@pytest.mark.parametrize("fusion", ["concat", "xattn", "xattn_gated", "unimodal_a"])
def test_fusion_on_every_backbone_family(name, size, n_prefix, fusion):
    torch.manual_seed(0)
    m = build_model(_cfg(backbone=name, fusion=fusion, freeze="last_n", unfreeze_last_n=1))
    a, b = torch.randn(2, 3, size, size), torch.randn(2, 3, size, size)
    out = m(a, b)
    assert out.shape == (2, 2) and torch.isfinite(out).all()
    out.sum().backward()
    bad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not bad, f"trainable parameters without gradient: {bad[:3]}"
    assert any(p.requires_grad for p in m.head.parameters())
    n_back = sum(p.numel() for p in m.backbone_a.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in m.backbone_a.parameters())
    assert 0 < n_back < n_all                                                       # last_n really freezes the earlier blocks


def test_registry_flags_gated_and_unverified_models():
    assert BACKBONES["dinov3_vitb16"].gated and BACKBONES["dinov3_vitb16"].patch == 16
    assert not BACKBONES["dinov2_base"].gated and BACKBONES["dinov2_base"].patch == 14
    assert BACKBONES["dinov3_vitl16"].verified and not BACKBONES["dinov3_vitb16"].verified
    with pytest.raises(ValueError):
        make_backbone("not_a_model")


def test_frozen_tv_backbone_keeps_batchnorm_statistics():
    bb = make_backbone("tv_resnet50", pretrained=False)
    bb.set_trainable("last_n", 1)
    bb.train()
    frozen_bn = [m for m in list(bb.features.children())[:-1] for m in m.modules() if isinstance(m, nn.BatchNorm2d)]
    assert frozen_bn and all(not m.training for m in frozen_bn)
    before = frozen_bn[0].running_mean.clone()
    bb(torch.randn(2, 3, 64, 64))
    assert torch.equal(before, frozen_bn[0].running_mean)

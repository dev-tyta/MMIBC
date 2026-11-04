"""Backbone registry. Every backbone returns the global token AND the patch-token sequence plus its patch grid.

Why tokens: the cross-attention fusion needs real token sequences (the legacy implementation fed single CLS vectors,
where softmax over one key is identically 1).

Families
--------
dinov2        facebook/dinov2-*                       patch 14, 1 prefix token (CLS)              Apache-2.0
dinov2_reg    facebook/dinov2-with-registers-*        patch 14, CLS + 4 registers                 Apache-2.0
dinov3        facebook/dinov3-*-pretrain-lvd1689m     patch 16, CLS + 4 registers, RoPE           GATED, custom "dinov3-license"
tv            torchvision CNNs (ResNet / ConvNeXt / EfficientNet)  stride-32 feature map as the token grid

Register tokens are NOT patch tokens: they are skipped when the patch sequence is cut out (``n_prefix``), otherwise
they would be fed to the cross-attention and to the Grad-CAM grid as if they were image locations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class BackboneOutput(NamedTuple):
    cls: torch.Tensor  # (B, D)
    patches: torch.Tensor  # (B, N, D)   N == grid[0] * grid[1]
    grid: tuple[int, int]
    attentions: tuple | None = None  # per layer (B, heads, T, T), T = n_prefix + N   (ViT, eager attention only)
    n_prefix: int = 1  # CLS (+ registers): tokens that precede the patches in ``attentions``


@dataclass(frozen=True)
class BackboneSpec:
    family: str
    hf_id: str = ""
    patch: int = 14
    gated: bool = False
    license: str = ""
    verified: bool = False  # True only if the id was confirmed from the model card in this project
    note: str = ""


_D2 = dict(family="dinov2", patch=14, license="Apache-2.0", verified=True)
_D2R = dict(family="dinov2_reg", patch=14, license="Apache-2.0", verified=True, note="Id confirmed (the transformers documentation loads facebook/dinov2-with-registers-base; default 4 register tokens); small and large follow the same pattern.")
_D3 = dict(family="dinov3", patch=16, gated=True, license="DINOv3 License (custom, gated on Hugging Face)")

BACKBONES: dict[str, BackboneSpec] = {
    "dinov2_small": BackboneSpec(hf_id="facebook/dinov2-small", **_D2),
    "dinov2_base": BackboneSpec(hf_id="facebook/dinov2-base", **_D2),
    "dinov2_large": BackboneSpec(hf_id="facebook/dinov2-large", **_D2),
    "dinov2reg_small": BackboneSpec(hf_id="facebook/dinov2-with-registers-small", **_D2R),
    "dinov2reg_base": BackboneSpec(hf_id="facebook/dinov2-with-registers-base", **_D2R),
    "dinov2reg_large": BackboneSpec(hf_id="facebook/dinov2-with-registers-large", **_D2R),
    # DINOv3 ids follow Meta's naming scheme. Confirmed from the model card / the transformers docs: ViT-L/16 (verified=True), ViT-S/16.
    # ViT-B/16 and ViT-H+/16 follow the same pattern (third-party repos are named after the B/16 id) but were not seen on a card of their own.
    "dinov3_vits16": BackboneSpec(hf_id="facebook/dinov3-vits16-pretrain-lvd1689m", **_D3),
    "dinov3_vitb16": BackboneSpec(hf_id="facebook/dinov3-vitb16-pretrain-lvd1689m", **_D3),
    "dinov3_vitl16": BackboneSpec(hf_id="facebook/dinov3-vitl16-pretrain-lvd1689m", verified=True, **_D3),
    "dinov3_vith16plus": BackboneSpec(hf_id="facebook/dinov3-vith16plus-pretrain-lvd1689m", **_D3),
    # torchvision baselines (ImageNet-1k weights): the architectures the published mammography baselines use
    "tv_resnet50": BackboneSpec(family="tv", patch=32, license="BSD-3", verified=True),
    "tv_convnext_tiny": BackboneSpec(family="tv", patch=32, license="BSD-3", verified=True),
    "tv_convnext_base": BackboneSpec(family="tv", patch=32, license="BSD-3", verified=True),
    "tv_efficientnet_b0": BackboneSpec(family="tv", patch=32, license="BSD-3", verified=True),
    "tv_efficientnet_b5": BackboneSpec(family="tv", patch=32, license="BSD-3", verified=True),
    # tiny randomly initialised models: tests and the CPU smoke run (no download)
    "dinov2_tiny_test": BackboneSpec(family="dinov2", patch=14, note="test"),
    "dinov2reg_tiny_test": BackboneSpec(family="dinov2_reg", patch=14, note="test"),
    "dinov3_tiny_test": BackboneSpec(family="dinov3", patch=16, note="test"),
}
TEST_IMAGE_SIZE = {"dinov2": 56, "dinov2_reg": 56, "dinov3": 64}


def _tiny_config(spec: BackboneSpec):
    from transformers import DINOv3ViTConfig, Dinov2Config, Dinov2WithRegistersConfig

    common = dict(hidden_size=32, num_hidden_layers=3, num_attention_heads=2, intermediate_size=64, patch_size=spec.patch)
    if spec.family == "dinov3":
        return DINOv3ViTConfig(num_register_tokens=4, image_size=TEST_IMAGE_SIZE["dinov3"], **common)
    if spec.family == "dinov2_reg":
        return Dinov2WithRegistersConfig(num_register_tokens=4, image_size=TEST_IMAGE_SIZE["dinov2_reg"], num_channels=3, **common)
    return Dinov2Config(image_size=TEST_IMAGE_SIZE["dinov2"], num_channels=3, **common)


def _blocks(model: nn.Module) -> nn.ModuleList:
    """Transformer blocks across HF implementations (DINOv2: encoder.layer, DINOv3: model / layer)."""
    for container in (model, getattr(model, "encoder", None), getattr(model, "model", None)):
        if container is None:
            continue
        for name in ("layer", "layers"):
            m = getattr(container, name, None)
            if isinstance(m, nn.ModuleList):
                return m
    raise AttributeError(f"Cannot find the transformer blocks of {type(model).__name__}")


def _final_norm(model: nn.Module) -> nn.Module:
    for name in ("layernorm", "norm"):
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError(f"Cannot find the final norm of {type(model).__name__}")


class _DropPath(nn.Module):
    """Stochastic depth (per sample): in training the residual branch is dropped with probability ``p`` and rescaled."""

    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class ViTBackbone(nn.Module):
    """DINOv2 / DINOv2-with-registers / DINOv3 through Hugging Face ``AutoModel``."""

    def __init__(self, name: str = "dinov2_base", pretrained: bool = True, eager_attention: bool = False):
        super().__init__()
        if name not in BACKBONES or BACKBONES[name].family == "tv":
            raise ValueError(f"'{name}' is not a ViT backbone. Options: {[k for k, v in BACKBONES.items() if v.family != 'tv']}")
        spec = BACKBONES[name]
        self.name, self.spec = name, spec
        from transformers import AutoConfig, AutoModel

        if name.endswith("_tiny_test"):
            cfg = _tiny_config(spec)
            if eager_attention:
                cfg._attn_implementation = "eager"
            self.model = AutoModel.from_config(cfg)
        elif pretrained:
            kw = {"attn_implementation": "eager"} if eager_attention else {}
            if spec.gated:
                from mmibc.utils.hf import ensure_hf_token

                ensure_hf_token()  # picks up HF_TOKEN / a saved login / the Colab secret before the first request
            try:
                self.model = AutoModel.from_pretrained(spec.hf_id, **kw)
            except Exception as exc:  # gated / offline / bad id: say what to do
                hint = (
                    f" '{spec.hf_id}' is a GATED repository: accept the licence on its Hugging Face page and expose a token "
                    "(Colab: add HF_TOKEN under Secrets, or run huggingface_hub.login())."
                ) if spec.gated else ""
                raise RuntimeError(f"Could not load backbone '{name}' ({spec.hf_id}): {exc}.{hint}"
                                   + ("" if spec.verified else " (This id follows the naming scheme but was not confirmed from its model card: check it on huggingface.co.)")) from exc
        else:
            cfg = AutoConfig.from_pretrained(spec.hf_id)
            if eager_attention:
                cfg._attn_implementation = "eager"
            self.model = AutoModel.from_config(cfg)
        cfg = self.model.config
        self.embed_dim: int = cfg.hidden_size
        self.patch_size: int = int(getattr(cfg, "patch_size", spec.patch))
        self.n_prefix: int = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)

    @property
    def n_layers(self) -> int:
        return len(_blocks(self.model))

    def set_drop_path(self, rate: float) -> int:
        """Stochastic depth on the TRAINABLE blocks only (rate grows linearly to ``rate`` at the top block). Returns how many blocks got it.

        The frozen blocks are left deterministic: their pretrained function is what the trainable top is adapting to.
        """
        blocks = [b for b in _blocks(self.model) if any(p.requires_grad for p in b.parameters())]
        n = 0
        for i, blk in enumerate(blocks):
            if hasattr(blk, "drop_path"):
                blk.drop_path = _DropPath(rate * (i + 1) / len(blocks))
                n += 1
        if rate > 0 and n == 0:
            log.warning("drop_path=%s requested but '%s' blocks expose no drop_path hook: no stochastic depth applied", rate, self.name)
        return n

    def check_input_size(self, size) -> None:
        from mmibc.data.preprocess import as_hw

        if any(v % self.patch_size for v in as_hw(size)):
            raise ValueError(f"input size {size} must be a multiple of the patch size {self.patch_size} for '{self.name}' (both height and width)")

    def forward(self, x: torch.Tensor, output_attentions: bool = False) -> BackboneOutput:
        out = self.model(pixel_values=x, output_attentions=output_attentions)
        h = out.last_hidden_state
        grid = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)
        patches = h[:, self.n_prefix:]
        if patches.shape[1] != grid[0] * grid[1]:
            raise RuntimeError(f"Token count {patches.shape[1]} != grid {grid} (n_prefix={self.n_prefix}); is the input size a multiple of {self.patch_size}?")
        return BackboneOutput(h[:, 0], patches, grid, out.attentions if output_attentions else None, self.n_prefix)

    def set_trainable(self, policy: str, last_n: int = 2) -> None:
        """policy: 'all' (frozen) | 'last_n' (last n blocks + final norm trainable) | 'none' (fully trainable)."""
        if policy not in ("all", "last_n", "none"):
            raise ValueError(f"freeze policy must be all | last_n | none, got '{policy}'")
        for p in self.parameters():
            p.requires_grad = policy == "none"
        if policy == "last_n":
            blocks = _blocks(self.model)
            for blk in blocks[max(0, len(blocks) - last_n):]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in _final_norm(self.model).parameters():
                p.requires_grad = True
        # mask_token is only used for masked-image modelling, which we never run: it would be a trainable parameter
        # that never receives a gradient (and would pick up weight decay for nothing).
        for n, p in self.model.named_parameters():
            if n.endswith("mask_token"):
                p.requires_grad = False


class TVBackbone(nn.Module):
    """torchvision CNN used as a token backbone: pooled feature as ``cls``, stride-32 feature map as the token grid."""

    def __init__(self, name: str, pretrained: bool = True, eager_attention: bool = False):
        super().__init__()
        import torchvision.models as tvm

        fam = name[len("tv_"):]
        w = "DEFAULT" if pretrained else None
        net = getattr(tvm, fam)(weights=w)
        if fam.startswith("resnet"):
            feats = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4)
        else:
            feats = net.features
        self.name, self.spec, self.features = name, BACKBONES[name], feats
        self.patch_size, self.n_prefix = 32, 1
        with torch.no_grad():
            self.embed_dim = int(self.features.eval()(torch.zeros(1, 3, 64, 64)).shape[1])

    n_layers = None  # layer-wise lr decay is defined for transformer backbones only

    def check_input_size(self, size) -> None:
        from mmibc.data.preprocess import as_hw

        if any(v % 32 for v in as_hw(size)):
            raise ValueError(f"input size {size} must be a multiple of 32 for '{self.name}' (both height and width)")

    def forward(self, x: torch.Tensor, output_attentions: bool = False) -> BackboneOutput:
        f = self.features(x)
        return BackboneOutput(f.mean(dim=(2, 3)), f.flatten(2).transpose(1, 2), (f.shape[-2], f.shape[-1]), None, 1)

    def set_trainable(self, policy: str, last_n: int = 2) -> None:
        if policy not in ("all", "last_n", "none"):
            raise ValueError(f"freeze policy must be all | last_n | none, got '{policy}'")
        for p in self.parameters():
            p.requires_grad = policy == "none"
        if policy == "last_n":
            for stage in list(self.features.children())[-last_n:]:
                for p in stage.parameters():
                    p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:  # frozen stages keep their BatchNorm running statistics
            for m in self.features.children():
                if not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self


def make_backbone(name: str, pretrained: bool = True, eager_attention: bool = False):
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Options: {sorted(BACKBONES)}")
    cls = TVBackbone if BACKBONES[name].family == "tv" else ViTBackbone
    return cls(name, pretrained=pretrained, eager_attention=eager_attention)


DinoV2Backbone = ViTBackbone  # backwards-compatible name

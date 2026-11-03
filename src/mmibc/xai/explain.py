"""Attribution maps for the FUSION model.

token_cam           Grad-CAM on the patch tokens of one stream, taken w.r.t. the logit of the FUSED prediction (so the
                    map explains the cross-modal decision, not a unimodal head).  methods: gradcam | gradxact
rollout_cam         Attention rollout through one backbone (ViT, eager attention); register tokens are skipped.
cross_attention_cam Where the CLS token of one stream looks in the OTHER stream (cross-attention models only).
All maps are returned on the stream's token grid and are resized to the model-input size by ``to_input_size``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mmibc.models.attention import attention_rollout


def _norm01(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    lo, hi = flat.min(1).values[:, None], flat.max(1).values[:, None]
    return ((flat - lo) / (hi - lo).clamp_min(1e-12)).view_as(x)


def to_input_size(cam: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """(B, gh, gw) -> (B, H, W) float32 in [0, 1] (bilinear; per-sample min-max)."""
    up = F.interpolate(cam[:, None].float(), size=size, mode="bilinear", align_corners=False)[:, 0]
    return _norm01(up).cpu().numpy().astype(np.float32)


def token_cam(model, a: torch.Tensor, b: torch.Tensor, stream: str = "a", target: torch.Tensor | None = None, method: str = "gradcam") -> dict:
    model.eval()
    a, b = a.clone().requires_grad_(True), b.clone().requires_grad_(True)  # inputs require grad so the graph exists even with frozen backbones
    with torch.enable_grad():
        logits, aux = model(a, b, return_aux=True)
        out = aux.get(f"out_{stream}")
        if out is None:
            raise ValueError(f"model with fusion='{model.fusion}' has no stream '{stream}'")
        tgt = logits.argmax(1) if target is None else target.to(logits.device)
        out.patches.retain_grad()
        model.zero_grad(set_to_none=True)
        logits.gather(1, tgt[:, None]).sum().backward()
    g, act = out.patches.grad.detach(), out.patches.detach()
    if method == "gradcam":
        w = g.mean(dim=1, keepdim=True)
        cam = F.relu((w * act).sum(-1))
    elif method == "gradxact":
        cam = F.relu((g * act).sum(-1))
    else:
        raise ValueError(f"unknown CAM method '{method}'")
    gh, gw = out.grid
    return dict(cam=cam.view(-1, gh, gw), prob=torch.softmax(logits.detach(), -1).gather(1, tgt[:, None])[:, 0], target=tgt.detach())


@torch.no_grad()
def rollout_cam(backbone, x: torch.Tensor) -> torch.Tensor:
    out = backbone(x, output_attentions=True)
    r = attention_rollout(out.attentions, n_prefix=out.n_prefix)
    return r.view(-1, *out.grid)


@torch.no_grad()
def cross_attention_cam(model, a: torch.Tensor, b: torch.Tensor, query_stream: str = "a") -> torch.Tensor:
    """CLS of ``query_stream`` attending over the other stream's patches (grid of the OTHER stream)."""
    if not model.fusion.startswith("xattn"):
        raise ValueError("cross-attention maps exist only for xattn / xattn_gated models")
    model.eval()
    _, aux = model(a, b, return_aux=True)
    key = "attn_a_to_b" if query_stream == "a" else "attn_b_to_a"
    other = aux["out_b" if query_stream == "a" else "out_a"]
    return aux[key][:, 0, 1:].reshape(-1, *other.grid)


@torch.no_grad()
def deletion_insertion(model, a: torch.Tensor, b: torch.Tensor, cam: torch.Tensor, stream: str, target: torch.Tensor, steps: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Faithfulness. Deletion: blank the most-attributed patches first, track P(target); a faithful map makes it fall fast (low AUC).
    Insertion: start from a blank image and reveal the most-attributed patches first (high AUC is good). AUC over fraction of patches in [0, 1]."""
    model.eval()
    B, (gh, gw) = a.shape[0], cam.shape[-2:]
    n = gh * gw
    ranks = torch.argsort(torch.argsort(-cam.flatten(1), dim=1), dim=1)
    ks = np.unique(np.linspace(0, n, steps + 1).astype(int))
    x = a if stream == "a" else b
    del_p, ins_p = [], []
    for k in ks:
        keep_top = (ranks < int(k)).float().view(B, 1, gh, gw)
        m = F.interpolate(keep_top, size=x.shape[-2:], mode="nearest")
        for mask_x, store in ((x * (1 - m), del_p), (x * m, ins_p)):
            logits = model(mask_x, b) if stream == "a" else model(a, mask_x)
            store.append(torch.softmax(logits, -1).gather(1, target[:, None])[:, 0].cpu().numpy())
    frac = ks / n
    auc = lambda ys: np.trapezoid(np.stack(ys, 1), frac, axis=1) / max(frac[-1], 1e-12)  # noqa: E731
    return auc(del_p), auc(ins_p)

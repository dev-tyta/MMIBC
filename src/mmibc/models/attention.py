"""Cross-attention over token SEQUENCES, plus attention rollout for explanation.

Why this exists: the legacy module reshaped each (B, D) feature vector to (B, 1, D), so the attention matrix was
(B, 1, 1) and softmax over a single key is identically 1.0. The query was discarded and the "cross-attention" was
just v_proj(context). Here queries and keys/values are sequences of N tokens, so the weights genuinely depend on
the query. ``assert_query_dependence`` is the regression test for exactly that failure.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Pre-norm transformer cross-attention block: Q from ``q_tokens``, K/V from ``kv_tokens``, then an FFN."""

    def __init__(self, dim: int, heads: int = 8, ffn_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.norm_q, self.norm_kv, self.norm_ffn = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_mult * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor, return_weights: bool = False):
        kv = self.norm_kv(kv_tokens)
        out, w = self.attn(self.norm_q(q_tokens), kv, kv, need_weights=return_weights, average_attn_weights=True)
        x = q_tokens + self.drop(out)
        x = x + self.drop(self.ffn(self.norm_ffn(x)))
        return (x, w) if return_weights else x  # w: (B, Nq, Nk), rows sum to 1


@torch.no_grad()
def assert_query_dependence(block: CrossAttentionBlock, n_q: int = 6, n_kv: int = 9, atol: float = 1e-4) -> None:
    """Fail if the block ignores its query (the legacy bug): outputs must change with the query and weights must not be trivial."""
    was_training = block.training
    block.eval()
    dim = block.norm_q.normalized_shape[0]
    q1, q2, kv = torch.randn(2, n_q, dim), torch.randn(2, n_q, dim), torch.randn(2, n_kv, dim)
    _, w = block(q1, kv, return_weights=True)
    if w.shape[-1] < 2:
        raise AssertionError("Attention is computed over a single key: softmax is identically 1.")
    if not torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5):
        raise AssertionError("Attention weights do not sum to 1")
    if torch.allclose(w, torch.full_like(w, 1.0 / n_kv), atol=1e-3):
        raise AssertionError("Attention weights are uniform; scores carry no information")
    o1, o2 = block(q1, kv), block(q2, kv)
    # With an identical query residual removed, a different query must still change the attended content.
    if torch.allclose(o1 - q1, o2 - q2, atol=atol):
        raise AssertionError("Output does not depend on the query")
    block.train(was_training)


def attention_rollout(attentions: tuple[torch.Tensor, ...], discard_ratio: float = 0.0, n_prefix: int = 1) -> torch.Tensor:
    """Attention rollout (Abnar & Zuidema 2020). attentions: per-layer (B, heads, T, T).

    Returns (B, T - n_prefix): CLS -> patch relevance. ``n_prefix`` = CLS + register tokens (DINOv3 / DINOv2-with-registers
    put registers before the patches; they take part in the rollout but are not image locations).
    """
    if attentions is None:
        raise ValueError("No attentions: build the model with eager attention (build_model(..., eager_attention=True)); CNN backbones have none.")
    result = None
    for att in attentions:
        a = att.mean(dim=1)  # head average -> (B, T, T)
        if discard_ratio > 0:
            flat = a.flatten(1)
            k = int(flat.shape[1] * discard_ratio)
            if k:
                thr = flat.kthvalue(k, dim=1).values[:, None, None]
                a = torch.where(a < thr, torch.zeros_like(a), a)
        a = a + torch.eye(a.shape[-1], device=a.device)  # residual connection
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else a @ result
    return result[:, 0, n_prefix:]

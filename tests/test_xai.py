from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

from mmibc.data.preprocess import letterbox_square
from mmibc.models.factory import build_model
from mmibc.pipeline import load_trained, run_experiment, run_xai
from mmibc.utils.config import load_config
from mmibc.xai.explain import cross_attention_cam, deletion_insertion, rollout_cam, to_input_size, token_cam
from mmibc.xai.ground_truth import box_mask
from mmibc.xai.metrics import energy_in_mask, iou_at, pointing_game


def _cfg(**kw):
    base = dict(backbone="dinov2_tiny_test", pretrained=False, shared_backbone=False, fusion="xattn", n_classes=2, xattn_layers=1, xattn_heads=2, head_hidden=16, dropout=0.0, freeze="none", unfreeze_last_n=1)
    base.update(kw)
    return OmegaConf.create(base)


def test_box_mask_agrees_with_letterboxing_the_drawn_mask():
    w, h, size = 112, 84, 56
    box = (20.0, 10.0, 70.0, 40.0)
    drawn = np.zeros((h, w), np.uint8)
    drawn[10:40, 20:70] = 255
    ref = np.asarray(letterbox_square(Image.fromarray(drawn), size)[0]) > 127
    got = box_mask(np.array([box]), w, h, size)
    inter, union = (ref & got).sum(), (ref | got).sum()
    assert inter / union > 0.85 and got.any()
    assert not got[:6].any()                                              # the padded band (letterbox) is never lesion


def test_localisation_metrics():
    mask = np.zeros((10, 10), bool)
    mask[2:5, 2:5] = True
    cam = np.zeros((10, 10))
    cam[3, 3] = 1.0
    assert pointing_game(cam, mask) == 1.0 and energy_in_mask(cam, mask) == 1.0 and iou_at(cam, mask, 0.5) == pytest.approx(1 / 9)
    cam2 = np.zeros((10, 10))
    cam2[8, 8] = 1.0
    assert pointing_game(cam2, mask) == 0.0 and energy_in_mask(cam2, mask) == 0.0
    uniform = np.ones((10, 10))
    assert energy_in_mask(uniform, mask) == pytest.approx(mask.mean())     # chance level == lesion area fraction


class RegionModel(nn.Module):
    """Class-1 evidence is the mean intensity of the TOP-LEFT quadrant of stream a. Ground truth for the faithfulness metrics."""

    fusion = "concat"

    def forward(self, a, b=None, return_aux=False):
        s = a[:, 0, : a.shape[2] // 2, : a.shape[3] // 2].mean(dim=(1, 2)) * 10
        return torch.stack([-s, s], 1)


def test_deletion_insertion_rank_an_oracle_map_above_its_complement():
    a = torch.ones(3, 3, 56, 56)
    oracle = torch.zeros(3, 4, 4)
    oracle[:, :2, :2] = 1.0
    wrong = 1.0 - oracle
    tgt = torch.ones(3, dtype=torch.long)
    m = RegionModel()
    d_o, i_o = deletion_insertion(m, a, a, oracle + 0.01 * torch.rand(3, 4, 4), "a", tgt, steps=16)
    d_w, i_w = deletion_insertion(m, a, a, wrong + 0.01 * torch.rand(3, 4, 4), "a", tgt, steps=16)
    assert (d_o < d_w - 0.2).all() and (i_o > i_w + 0.2).all()
    assert ((0 <= d_o) & (d_o <= 1) & (0 <= i_o) & (i_o <= 1)).all()


@pytest.mark.parametrize("fusion", ["concat", "gated", "xattn", "xattn_gated"])
def test_token_cam_flows_through_the_fused_logit(fusion):
    torch.manual_seed(0)
    m = build_model(_cfg(fusion=fusion), eager_attention=True).eval()
    a, b = torch.randn(2, 3, 56, 56), torch.randn(2, 3, 56, 56)
    for stream in ("a", "b"):
        r = token_cam(m, a, b, stream)
        assert r["cam"].shape == (2, 4, 4) and torch.isfinite(r["cam"]).all() and (r["cam"] >= 0).all() and r["cam"].sum() > 0
        full = to_input_size(r["cam"], (56, 56))
        assert full.shape == (2, 56, 56) and full.min() >= 0 and full.max() <= 1 + 1e-6
    # depends on the explained class
    c0 = token_cam(m, a, b, "a", torch.zeros(2, dtype=torch.long), "gradxact")["cam"]
    c1 = token_cam(m, a, b, "a", torch.ones(2, dtype=torch.long), "gradxact")["cam"]
    assert not torch.allclose(c0, c1)
    # parameter-randomisation sanity check (Adebayo et al. 2018): a map that ignores the weights is not an explanation
    base = token_cam(m, a, b, "a")["cam"].flatten()
    for p in m.head.parameters():
        nn.init.normal_(p, std=0.5)
    after = token_cam(m, a, b, "a", torch.zeros(2, dtype=torch.long))["cam"].flatten()
    assert not torch.allclose(base / base.norm(), after / after.norm(), atol=1e-3)
    # the input tensors the caller passed in are untouched
    assert not a.requires_grad


def test_stream_selection_and_attention_maps():
    ua = build_model(_cfg(fusion="unimodal_a"), eager_attention=True).eval()
    x = torch.randn(1, 3, 56, 56)
    assert token_cam(ua, x, x, "a")["cam"].shape == (1, 4, 4)
    with pytest.raises(ValueError, match="no stream"):
        token_cam(ua, x, x, "b")
    xm = build_model(_cfg(fusion="xattn", backbone="dinov3_tiny_test"), eager_attention=True).eval()
    a = torch.randn(2, 3, 64, 64)
    assert cross_attention_cam(xm, a, a, "a").shape == (2, 4, 4)                 # DINOv3: 4x4 grid, registers excluded
    r = rollout_cam(xm.backbone_a, a)
    assert r.shape == (2, 4, 4) and torch.isfinite(r).all()
    with pytest.raises(ValueError, match="only for xattn"):
        cross_attention_cam(build_model(_cfg(fusion="concat")), x, x)
    assert xm.backbone_a.spec.family == "dinov3"


def test_explain_pipeline_end_to_end(project):
    paths, over = project
    cfg = load_config("smoke.yaml", over + ["task.kind=synthetic_pairs", "task.pair_mode=label_matched", "training.epochs=1", "run_name=xai_e2e", "eval.controls=[none]", "xai.n_samples=6", "xai.steps=4"])
    run_experiment(cfg, paths)
    summ = run_xai(cfg, paths, "test")
    assert {"gradcam", "gradxact", "rollout", "xattn_cls", "random"} <= set(summ["method"])
    assert set(summ["stream"]) == {"a", "b"}
    for k in ("deletion_auc", "insertion_auc"):
        assert summ[k].between(0, 1).all()
    ann = summ[summ["stream"] == "a"]
    assert (ann["n_annotated"] > 0).any() and ann["pointing"].dropna().between(0, 1).all()
    run = next((paths.runs).glob("xai_e2e*"))
    assert (run / "xai_samples_test.csv").exists() and list((run / "cams").glob("*.npy"))
    samples = pd.read_csv(run / "xai_samples_test.csv")
    assert samples.groupby(["pair_id", "stream", "method"]).size().max() == 1

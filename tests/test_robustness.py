"""The fixes and mitigations that followed the first real Colab runs (see README, "What the first real runs showed").

Each test pins one thing that was found: miscalibrated probabilities, a fusion model that cannot be attributed to its second stream,
mammograms read at a resolution where a lesion is one patch wide, a suite that dies with the runtime and loses everything.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from mmibc.data.datasets import stream_size
from mmibc.data.preprocess import as_hw, letterbox, letterbox_params
from mmibc.evaluation.metrics import apply_calibration, calibration, choose_threshold, fit_calibration, operating_points
from mmibc.models.factory import build_model
from mmibc.pipeline import build_pair_sets, build_tables, run_experiment, run_xai
from mmibc.training.ema import EMA
from mmibc.training.losses import build_loss
from mmibc.training.optim import build_optimizer
from mmibc.training.trainer import Trainer, config_fingerprint
from mmibc.utils.config import load_config
from mmibc.xai.ground_truth import box_mask


def smoke(over, *extra):
    return load_config("smoke.yaml", [*over, "training.epochs=1", "eval.bootstrap_samples=20", "eval.controls=[none,drop_b]", "xai.n_samples=4", "xai.steps=3", *extra])


# ----------------------------------------------------------------------------- config / fingerprints
def test_added_options_do_not_change_the_fingerprint_of_runs_made_before_them():
    # 48d389f3dde7 is the fingerprint stored in metrics.json of the first real VinDr run (multiview_xattn_s42): it must still resume/skip.
    assert config_fingerprint(load_config("vindr_multiview.yaml")) == "48d389f3dde7"
    base = load_config("vindr_multiview.yaml")
    for key, value in (("training.ema_decay", 0.99), ("training.label_smoothing", 0.1), ("training.layer_decay", 0.8), ("model.drop_path", 0.1),
                       ("input.image_size_a", [448, 266]), ("data.split_seed", 7)):
        changed = load_config("vindr_multiview.yaml", [f"{key}={json.dumps(value)}"])
        assert config_fingerprint(changed) != config_fingerprint(base), key
    assert config_fingerprint(load_config("vindr_multiview_robust.yaml")) != config_fingerprint(base)


def test_robust_configs_are_valid():
    for name in ("vindr_multiview_robust", "mammo_us_robust", "mammo_us_pool"):
        cfg = load_config(name)
        assert cfg.model.backbone and as_hw(stream_size(cfg, "a"))
    r = load_config("mammo_us_robust")
    assert stream_size(r, "a") == (448, 266) and stream_size(r, "b") == (224, 224) and r.training.modality_dropout > 0
    mv = load_config("vindr_multiview_robust")
    assert stream_size(mv, "b") == stream_size(mv, "a") == (448, 266)           # both views are mammograms: one size, one shared backbone
    assert stream_size(load_config("vindr_multiview"), "a") == (224, 224)       # default unchanged


# ----------------------------------------------------------------------------- geometry
def test_rectangular_letterbox_and_box_mask_agree():
    img = Image.new("L", (40, 80), 255)
    out, aff = letterbox(img, (64, 40))
    assert out.size == (40, 64) and aff == letterbox_params(40, 80, (64, 40))        # PIL size is (W, H)
    assert as_hw(56) == (56, 56) and as_hw([448, 266]) == (448, 266)
    w, h = 112, 84
    drawn = np.zeros((h, w), np.uint8)
    drawn[10:40, 20:70] = 255
    ref = np.asarray(letterbox(Image.fromarray(drawn), (84, 56))[0]) > 127
    got = box_mask(np.array([(20.0, 10.0, 70.0, 40.0)]), w, h, (84, 56))
    assert got.shape == ref.shape == (84, 56)
    assert (ref & got).sum() / (ref | got).sum() > 0.8


def test_mismatched_sizes_are_rejected_for_early_fusion(project):
    paths, over = project
    cfg = smoke(over, "task.kind=synthetic_pairs", "model.fusion=early", "input.image_size_a=[56,42]", "input.image_size_b=56", "run_name=bad_early")
    with pytest.raises(ValueError, match="sizes must match"):
        run_experiment(cfg, paths)


def test_backbone_rejects_sizes_that_are_not_patch_multiples():
    bb = build_model(smoke([]).model).backbone_a
    bb.check_input_size((56, 42))
    with pytest.raises(ValueError, match="multiple of the patch size"):
        bb.check_input_size((56, 40))


# ----------------------------------------------------------------------------- training mitigations
def test_ema_averages_and_restores_exactly():
    m = build_model(smoke([]).model)
    ema = EMA(m, 0.9)
    start = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.add_(1.0)
    ema.update(m)
    live = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    with ema.applied(m):
        for n, p in m.named_parameters():
            if p.requires_grad:
                assert torch.all(p > start[n] - 1e-6) and torch.all(p < live[n] + 1e-6)   # between the old and the new weights
                assert not torch.allclose(p, live[n])
    for n, p in m.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p, live[n])                                          # restored bit for bit
    with pytest.raises(ValueError):
        EMA(m, 1.0)
    e2 = EMA(m, 0.9)
    e2.load_state_dict(ema.state_dict())
    assert e2.updates == 1


def test_resume_with_ema_matches_an_uninterrupted_run(project):
    paths, over = project
    cfg = smoke(over, "training.epochs=3", "training.patience=5", "training.ema_decay=0.9", "training.label_smoothing=0.05", "model.drop_path=0.0", "run_name=ema_resume")
    sets = build_pair_sets(cfg, build_tables(cfg, paths), paths)

    def trainer(run):
        torch.manual_seed(0)
        return Trainer(build_model(cfg.model), cfg, paths.runs / run, sets.train, sets.val, torch.device("cpu"))

    ref = trainer("ema_ref")
    s_ref = ref.fit()
    a = trainer("ema_resumed")
    a.stop_after_epoch = 1
    assert a.fit()["interrupted"]
    b = trainer("ema_resumed")
    s_b = b.fit()
    np.testing.assert_allclose(pd.DataFrame(ref.history)[["train_loss", "val_loss"]].to_numpy(), pd.DataFrame(b.history)[["train_loss", "val_loss"]].to_numpy(), rtol=1e-5, atol=1e-6)
    assert s_ref["best_epoch"] == s_b["best_epoch"] and s_b["ema"] is True
    # best.pt holds the AVERAGED weights; last.pt the live ones. fit() ends by loading best.pt, so compare the two files, not the model
    run = paths.runs / "ema_resumed"
    best = torch.load(run / "best.pt", weights_only=False)["model"]["state"]
    last = torch.load(run / "last.pt", weights_only=False)["model"]["state"]
    trainable = {n for n, p in b.model.named_parameters() if p.requires_grad}
    assert trainable and trainable <= set(best) and trainable <= set(last)
    assert any(not torch.allclose(best[n].cpu(), last[n].cpu()) for n in trainable)


def test_layer_wise_lr_decay_orders_the_learning_rates():
    cfg = smoke([], "training.layer_decay=0.5", "model.freeze=none")
    m = build_model(cfg.model)
    opt = build_optimizer(m, cfg.training)
    lrs = {g["lr"] for g in opt.param_groups}
    lr_bb = float(cfg.training.lr_backbone)
    assert max(lrs - {float(cfg.training.lr_head)}) == pytest.approx(lr_bb)            # final norm keeps the full backbone lr
    assert min(lrs) == pytest.approx(lr_bb * 0.5 ** (m.backbone_a.n_layers + 1))      # embeddings get the smallest
    assert float(cfg.training.lr_head) in lrs
    off = build_optimizer(build_model(smoke([], "model.freeze=none").model), smoke([]).training)
    assert {g["lr"] for g in off.param_groups} == {float(cfg.training.lr_head), lr_bb}  # decay 1.0 reproduces the old grouping
    with pytest.raises(ValueError, match="layer_decay"):
        build_optimizer(m, smoke([], "training.layer_decay=1.5").training)


def test_label_smoothing_changes_the_loss_and_is_validated():
    logits, y = torch.tensor([[3.0, -3.0], [-3.0, 3.0]]), torch.tensor([0, 1])
    t0, t1 = smoke([]).training, smoke([], "training.label_smoothing=0.1").training
    l0, l1 = build_loss(t0, np.array([5, 5]), torch.device("cpu"))(logits, y), build_loss(t1, np.array([5, 5]), torch.device("cpu"))(logits, y)
    assert l1 > l0 > 0
    with pytest.raises(ValueError, match="label_smoothing"):
        build_loss(smoke([], "training.label_smoothing=0.7").training, np.array([5, 5]), torch.device("cpu"))


def test_drop_path_only_touches_trainable_blocks_and_only_in_training():
    cfg = smoke([], "model.freeze=last_n", "model.unfreeze_last_n=1", "model.drop_path=0.5")
    m = build_model(cfg.model)
    blocks = list(m.backbone_a.model.encoder.layer)
    flags = [type(b.drop_path).__name__ for b in blocks]
    assert flags[-1] == "_DropPath" and all(f == "Identity" for f in flags[:-1])
    x = torch.randn(6, 3, 56, 56)
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(x, x), m(x, x))
    m.train()
    with torch.no_grad():
        assert not torch.allclose(m(x, x), m(x, x))


# ----------------------------------------------------------------------------- calibration and operating points
def test_platt_fixes_overconfidence_and_prior_shift_without_touching_the_ranking():
    rng = np.random.default_rng(0)
    y = (rng.random(6000) < 0.05).astype(int)
    z = rng.normal(0.0, 2.5, y.size) + y * 2.2 + 2.0              # a model trained on a balanced sampler, applied at 5 % prevalence
    raw = calibration(y, 1 / (1 + np.exp(-z)))
    assert raw["brier_skill"] < 0                                   # worse than always predicting the prevalence: the observed VinDr failure
    cal = fit_calibration(y, z, "platt")
    q = apply_calibration(z, cal)
    after = calibration(y, q)
    assert after["brier_skill"] > 0 and after["ece"] < raw["ece"] / 5 and cal["nll_after"] < cal["nll_before"]
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, q) == pytest.approx(roc_auc_score(y, z))        # monotone map
    t = fit_calibration(y, z, "temperature")
    assert t["b"] == 0.0 and t["a"] > 0
    assert calibration(y, apply_calibration(z, t))["brier"] > after["brier"]   # a temperature alone cannot fix a prior shift


def test_calibration_degenerate_inputs_fall_back_to_identity():
    z = np.linspace(-3, 3, 50)
    assert fit_calibration(np.zeros(50), z, "platt")["method"] == "none"            # one class only
    anti = fit_calibration((z < 0).astype(int), z, "platt")                          # anti-informative: refuse to flip the ranking
    assert anti["method"] == "none" and anti["a"] == 1.0
    assert fit_calibration((z > 0).astype(int), z, "none")["method"] == "none"
    with pytest.raises(ValueError, match="unknown calibration"):
        fit_calibration((z > 0).astype(int) ^ (np.arange(50) % 7 == 0), z, "isotonic")


def test_specificity_target_threshold_and_operating_points():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.1).astype(int)
    p = 1 / (1 + np.exp(-(rng.normal(0, 1, y.size) + 1.5 * y)))
    thr = choose_threshold(y, p, "spec_target", 0.90)
    spec = ((p < thr) & (y == 0)).sum() / (y == 0).sum()
    assert spec >= 0.90 - 1e-9 and spec < 0.93                                       # as sensitive as it can be while keeping 90 % specificity
    ops = operating_points(y, p, y, p)
    assert set(ops) == {"sens90", "sens95", "spec90", "spec95", "youden"}
    assert ops["sens90"]["sensitivity"] >= 0.90 and ops["sens95"]["sensitivity"] >= 0.95
    assert ops["spec95"]["specificity"] >= 0.95 and ops["spec90"]["specificity"] >= 0.90
    assert ops["spec95"]["sensitivity"] <= ops["spec90"]["sensitivity"]


def test_evaluate_model_recalibrates_on_validation_only_and_reports_it(project):
    paths, over = project
    cfg = smoke(over, "run_name=cal_eval", "eval.calibration=platt")
    res = run_experiment(cfg, paths)
    run = paths.runs / "cal_eval_dinov2_tiny_test"
    assert res["calibration"]["method"] in ("platt", "none") and "eval_fingerprint" in res
    test = res["splits"]["test"]
    assert {"raw_brier", "raw_ece", "raw_brier_skill"} <= set(test["raw"]) and "brier_skill" in test
    assert set(test["operating_points"]) == {"sens90", "sens95", "spec90", "spec95", "youden"}
    from sklearn.metrics import roc_auc_score

    for split in ("val", "test"):
        df = pd.read_csv(run / f"predictions_{split}.csv")
        assert {"p", "p_raw", "logit"} <= set(df.columns)
        assert roc_auc_score(df["y"], df["p"]) == pytest.approx(roc_auc_score(df["y"], df["p_raw"]))      # calibration never reorders the cases
        assert res["splits"][split]["auroc"] == pytest.approx(roc_auc_score(df["y"], df["p"]))
    r0 = run_experiment(smoke(over, "run_name=cal_off", "eval.calibration=none"), paths)
    assert r0["calibration"]["method"] == "none"
    df0 = pd.read_csv(paths.runs / "cal_off_dinov2_tiny_test" / "predictions_test.csv")
    np.testing.assert_allclose(df0["p"], df0["p_raw"], atol=1e-6)                                         # no calibration: probabilities untouched


# ----------------------------------------------------------------------------- resumability
def test_finished_run_is_skipped_but_re_evaluated_when_only_the_evaluation_changed(project, monkeypatch):
    paths, over = project
    cfg = smoke(over, "run_name=skip_me")
    first = run_experiment(cfg, paths)
    import mmibc.evaluation.evaluate as ev

    calls = []
    real = ev.evaluate_model
    monkeypatch.setattr(ev, "evaluate_model", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    again = run_experiment(cfg, paths)
    assert not calls and again["splits"]["test"]["auroc"] == first["splits"]["test"]["auroc"]       # nothing recomputed
    run_experiment(cfg, paths, force=True)
    assert len(calls) == 1                                                                           # --force re-evaluates
    changed = smoke(over, "run_name=skip_me", "eval.calibration=temperature")
    r2 = run_experiment(changed, paths)
    assert len(calls) == 2 and r2["calibration"]["method"] in ("temperature", "none")               # re-evaluated, not retrained
    assert r2["train"]["config_fingerprint"] == first["train"]["config_fingerprint"]


def test_metrics_from_before_recalibration_are_kept(project):
    paths, over = project
    cfg = smoke(over, "run_name=legacy_metrics")
    run_experiment(cfg, paths)
    run = paths.runs / "legacy_metrics_dinov2_tiny_test"
    m = json.loads((run / "metrics.json").read_text())
    for k in ("calibration", "eval_fingerprint"):
        m.pop(k)                                                      # what a run made by the first version looks like
    (run / "metrics.json").write_text(json.dumps(m))
    new = run_experiment(cfg, paths)
    assert "calibration" in new
    assert json.loads((run / "metrics.uncalibrated.json").read_text())["splits"]["test"]["auroc"] == m["splits"]["test"]["auroc"]


# ----------------------------------------------------------------------------- suites: seeds, fusion gain, status
def test_suite_with_seeds_reports_spread_ensemble_and_fusion_gain(project, tmp_path):
    from mmibc.suite import run_suite

    paths, over = project
    suite = tmp_path / "mini_seeds.yaml"
    suite.write_text("config: vindr_multiview.yaml\nreference: concat\nsplit_seed: 3\nruns:\n"
                     "  - {name: unimodal_a, set: [model.fusion=unimodal_a]}\n  - {name: unimodal_b, set: [model.fusion=unimodal_b]}\n  - {name: concat, set: [model.fusion=concat]}\n")
    base = [*over, "model.shared_backbone=true", "model.backbone=dinov2_tiny_test", "model.pretrained=false", "model.freeze=none", "model.xattn_heads=2", "model.head_hidden=32",
            "training.epochs=1", "input.image_size=56", "eval.controls=[none]", "eval.bootstrap_samples=20", "data.store_long_side=112", "training.batch_size=8"]
    res = run_suite(suite, paths, base, seeds=[1, 2])
    out = paths.runs / "suite_mini_seeds"
    assert {"auroc_mean", "auroc_sd", "n_seeds", "ens_auroc"} <= set(res.columns) and (res["n_seeds"] == 2).all()
    assert {"unimodal_a", "unimodal_b", "concat", "late_fusion"} <= set(res["model"])
    assert (out / "results_s1.csv").exists() and (out / "results_s2.csv").exists()
    gain = pd.read_csv(out / "fusion_gain.csv")
    assert {"vs", "delta_auroc", "delta_lo", "delta_hi", "beats_reference"} <= set(gain.columns)
    assert any(v.startswith("best_unimodal") for v in gain["vs"]) and {"concat", "late_fusion"} <= set(gain["model"])
    st = pd.read_csv(out / "status.csv")
    assert len(st) == 6 and (st["state"] == "done").all()
    # same partition for both seeds (so differences are due to training only), different training seeds
    def cfg_of(sd):
        return OmegaConf.load(paths.runs / f"mini_seeds_unimodal_a_s{sd}_dinov2_tiny_test" / "config.yaml")

    assert cfg_of(1).seed == 1 and cfg_of(2).seed == 2 and cfg_of(1).data.split_seed == cfg_of(2).data.split_seed == 3
    t1, t2 = (pd.read_csv(paths.runs / f"mini_seeds_concat_s{sd}_dinov2_tiny_test" / "predictions_test.csv") for sd in (1, 2))
    assert sorted(t1["pair_id"]) == sorted(t2["pair_id"])


def test_pool_ladder_suite_is_well_formed():
    from mmibc.suite import load_suite

    spec = load_suite("pool_ladder")
    names = [r["name"] for r in spec["runs"]]
    assert {"unimodal_a", "unimodal_b", "concat", "xattn"} <= set(names) and spec["reference"] in names
    cfg = load_config(spec["config"])
    assert cfg.task.kind == "synthetic_pairs" and list(cfg.data.us_sources) == ["busi", "bus_bra"]
    for r in load_suite("recipe")["runs"]:
        c = load_config(load_suite("recipe")["config"], r["set"])
        assert as_hw(stream_size(c, "a"))


# ----------------------------------------------------------------------------- rectangular, per-stream sizes end to end
@pytest.mark.parametrize("kind", ["multiview", "synthetic_pairs"])
def test_rectangular_inputs_train_evaluate_and_explain_end_to_end(project, kind):
    paths, over = project
    cfg = smoke(over, f"task.kind={kind}", "input.image_size_a=[56,42]", "input.image_size_b=56", f"run_name=rect_{kind}", "model.fusion=xattn")
    res = run_experiment(cfg, paths)
    assert res["splits"]["test"]["auroc"] == res["splits"]["test"]["auroc"]
    summary = run_xai(cfg, paths, "test")                         # ground-truth masks must have the SAME shape as the maps they are scored against
    assert {"a", "b"} <= set(summary["stream"]) and summary["n"].min() > 0
    sets = build_pair_sets(cfg, build_tables(cfg, paths), paths)
    item = sets.test[0]
    assert tuple(item["a"].shape[-2:]) == (56, 42)
    assert tuple(item["b"].shape[-2:]) == ((56, 42) if kind == "multiview" else (56, 56))

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from mmibc.models.factory import build_model
from mmibc.pipeline import build_pair_sets, build_tables, prepare, run_experiment
from mmibc.training.checkpoint import EarlyStopping
from mmibc.training.losses import FocalLoss
from mmibc.training.optim import build_optimizer, lr_factor
from mmibc.training.trainer import Trainer, predict
from mmibc.utils.config import load_config
from mmibc.utils.paths import ProjectPaths


def make_cfg(*overrides):
    return load_config("smoke.yaml", ["audit.min_per_class_split=1", "training.num_workers=0", *overrides])


@pytest.fixture(scope="module")
def project(raw_root, tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    paths = ProjectPaths(root / "drive", root / "local", raw_on_drive=False).ensure()
    # lay the fake raw data out where the pipeline expects it
    import shutil

    for name in ("vindr_images", "vindr_annotations", "busi", "extra_us"):
        shutil.copytree(raw_root / name, paths.raw / name)
    cfg = make_cfg("data.us_sources=[busi,extra_us]", "data.us_external_source=extra_us")
    prepare(cfg, paths)
    return cfg, paths


def test_optimizer_covers_every_trainable_parameter_and_catches_omissions():
    from mmibc.training.optim import assert_optimizer_covers

    cfg = make_cfg()
    m = build_model(cfg.model)
    opt = build_optimizer(m, cfg.training)
    assert {id(p) for g in opt.param_groups for p in g["params"]} == {id(p) for p in m.parameters() if p.requires_grad}
    assert {g["lr"] for g in opt.param_groups} == {cfg.training.lr_head, cfg.training.lr_backbone}
    assert any(g["weight_decay"] == 0 for g in opt.param_groups) and any(g["weight_decay"] > 0 for g in opt.param_groups)
    assert_optimizer_covers(m, opt)
    # failure mode: layers unfrozen AFTER the optimizer was built are not in it and silently never update
    m2 = build_model(make_cfg("model.freeze=last_n", "model.unfreeze_last_n=1").model)
    opt2 = build_optimizer(m2, make_cfg().training)
    for p in m2.backbone_a.model.encoder.layer[0].parameters():
        p.requires_grad = True
    with pytest.raises(AssertionError, match="NOT in the optimizer"):
        assert_optimizer_covers(m2, opt2)
    # the trainer refuses to start in that state
    for p in m.parameters():
        p.requires_grad = False
    with pytest.raises(AssertionError, match="no trainable parameters"):
        build_optimizer(m, cfg.training)


def test_lr_schedule_warms_up_then_decays():
    f = [lr_factor(s, 100, 10, "cosine") for s in range(100)]
    assert f[0] < f[9] <= 1.0 and f[10] == pytest.approx(1.0) and f[-1] < 0.01 and all(a >= b for a, b in zip(f[10:], f[11:]))


def test_early_stopping_counts_epochs_since_best_and_ignores_nan():
    es = EarlyStopping("max", patience=3, min_delta=0.01)
    assert es.step(0.5) and es.step(0.6)
    assert not es.step(0.605) and not es.step(float("nan")) and not es.should_stop
    assert not es.step(0.59) and es.should_stop
    assert es.step(0.7) and es.bad == 0


def test_focal_loss_reduces_to_ce_at_gamma_zero_and_downweights_easy_examples():
    logits, y = torch.tensor([[2.0, -1.0], [0.1, 0.0]]), torch.tensor([0, 1])
    assert FocalLoss(0.0)(logits, y) == pytest.approx(torch.nn.functional.cross_entropy(logits, y).item(), rel=1e-6)
    assert FocalLoss(2.0)(logits, y) < FocalLoss(0.0)(logits, y)
    a = FocalLoss(0.0, torch.tensor([1.0, 3.0]))
    assert a(logits, y) > FocalLoss(0.0)(logits, y)                      # alpha indexes the TARGET class


def test_tables_and_pairs_are_audited(project):
    cfg, paths = project
    tables = build_tables(cfg, paths, stage=True)
    assert all(r.ok for r in tables.audits.values())
    sets = build_pair_sets(cfg.copy() if hasattr(cfg, "copy") else cfg, tables, paths)
    assert len(sets.train) > 0 and len(sets.val) > 0 and len(sets.test) > 0
    assert (paths.splits / "multiview" / "pairs_test.csv").exists()
    assert (paths.splits / "audit_vindr.md").exists()


def test_end_to_end_multiview_train_resume_evaluate(project):
    cfg, paths = project
    cfg = make_cfg("data.us_sources=[busi,extra_us]", "training.epochs=3", "training.patience=5", "eval.controls=[none,drop_a,drop_b,shuffle_b]", "run_name=e2e_mv")
    tables = build_tables(cfg, paths)
    sets = build_pair_sets(cfg, tables, paths)

    def trainer(run):
        torch.manual_seed(0)
        m = build_model(cfg.model)
        return m, Trainer(m, cfg, paths.runs / run, sets.train, sets.val, torch.device("cpu"))

    # uninterrupted reference run
    m_ref, t_ref = trainer("ref")
    sum_ref = t_ref.fit()
    # same run interrupted after epoch 1 (a Colab disconnect), then resumed in a FRESH process-like object
    m_a, t_a = trainer("resumed")
    t_a.stop_after_epoch = 1
    assert t_a.fit()["interrupted"]
    m_b, t_b = trainer("resumed")
    sum_b = t_b.fit()
    h_ref, h_b = pd.DataFrame(t_ref.history), pd.DataFrame(t_b.history)
    assert len(h_ref) == len(h_b) == 3
    np.testing.assert_allclose(h_ref[["train_loss", "val_loss"]].to_numpy(), h_b[["train_loss", "val_loss"]].to_numpy(), rtol=1e-5, atol=1e-6)
    assert sum_ref["best_epoch"] == sum_b["best_epoch"]
    # the finished run is a no-op on re-entry and still returns the best weights
    m_c, t_c = trainer("resumed")
    assert t_c.fit()["best_epoch"] == sum_b["best_epoch"]
    # a different configuration must refuse to resume into this directory
    bad = make_cfg("training.epochs=3", "model.dropout=0.1", "run_name=e2e_mv")
    with pytest.raises(RuntimeError, match="different configuration"):
        Trainer(build_model(bad.model), bad, paths.runs / "resumed", sets.train, sets.val, torch.device("cpu"))._try_resume()
    # evaluation: threshold from val, controls present, predictions saved
    from mmibc.evaluation.evaluate import evaluate_model

    res = evaluate_model(m_b, sets.eval_sets(), cfg, paths.runs / "resumed", torch.device("cpu"))
    assert res["threshold_chosen_on"] == "val" and {c["control"] for c in res["controls"]} == {"none", "drop_a", "drop_b", "shuffle_b"}
    assert res["controls"][0]["delta_auroc"] == 0
    test_pred = pd.read_csv(paths.runs / "resumed" / "predictions_test.csv")
    assert len(test_pred) == len(sets.test) and test_pred["p"].between(0, 1).all()
    # determinism: predicting twice gives identical numbers
    p1, p2 = predict(m_b, sets.test, cfg.training, torch.device("cpu")), predict(m_b, sets.test, cfg.training, torch.device("cpu"))
    pd.testing.assert_frame_equal(p1, p2)


def test_synthetic_pairs_experiment_with_external_source(project):
    cfg, paths = project
    cfg = make_cfg(
        "data.us_sources=[busi,extra_us]", "data.us_external_source=extra_us", "task.kind=synthetic_pairs", "task.pair_mode=size_conditioned",
        "training.epochs=2", "model.shared_backbone=false", "run_name=e2e_syn", "eval.controls=[none,shuffle_b]",
    )
    metrics = run_experiment(cfg, paths)
    assert {"val", "test", "ext_test"} <= set(metrics["splits"])
    assert metrics["splits"]["test"]["n"] > 0 and 0 <= metrics["splits"]["test"]["auroc"] <= 1
    tabs = {s: pd.read_csv(paths.splits / "synthetic_pairs_size_conditioned" / f"pairs_{s}.csv") for s in ("val", "test", "ext_test")}
    assert all(not t["uid_b"].duplicated().any() for t in tabs.values())          # frozen evaluation pairs are 1-to-1
    us = pd.read_csv(paths.splits / "ultrasound_split.csv").set_index("uid")
    assert (us.loc[tabs["ext_test"]["uid_b"], "source"] == "extra_us").all()
    assert not set(tabs["test"]["uid_b"]) & set(tabs["val"]["uid_b"])

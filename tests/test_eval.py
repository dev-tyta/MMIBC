from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mmibc.evaluation.metrics import all_metrics, auroc, bootstrap_ci, choose_threshold, point_metrics
from mmibc.evaluation.stats import delong_ci, delong_test, mcnemar_exact, paired_bootstrap


def _scores(n=600, sep=1.0, seed=0, prev=0.3):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prev).astype(int)
    return y, rng.normal(0, 1, n) + sep * y


def test_delong_auc_matches_sklearn_and_ci_covers_bootstrap():
    y, s = _scores()
    auc, lo, hi = delong_ci(y, s)
    assert auc == pytest.approx(roc_auc_score(y, s), abs=1e-9)
    bs = bootstrap_ci(y, 1 / (1 + np.exp(-s)), 0.5, n=400, seed=1)["auroc"]
    assert abs(lo - bs[0]) < 0.02 and abs(hi - bs[1]) < 0.02          # two CI methods agree


def test_delong_test_detects_a_real_difference_and_not_a_null_one():
    y, strong = _scores(sep=1.5, seed=1)
    rng = np.random.default_rng(2)
    weak = strong * 0.2 + rng.normal(0, 1.2, len(y))
    assert delong_test(y, strong, weak)["p"] < 0.001
    near = strong + rng.normal(0, 0.05, len(y))                          # nearly identical model
    assert delong_test(y, strong, near)["p"] > 0.05
    same = delong_test(y, strong, strong)
    assert same["diff"] == 0 and same["p"] == 1.0
    # Calibration under H0: two INDEPENDENT scorers with the same expected AUC -> rejection rate at alpha=0.05 should be ~5%.
    rej = []
    for k in range(300):
        rng_k = np.random.default_rng(1000 + k)
        yk = (rng_k.random(250) < 0.4).astype(int)
        s1, s2 = rng_k.normal(0, 1, 250) + yk, rng_k.normal(0, 1, 250) + yk
        rej.append(delong_test(yk, s1, s2)["p"] < 0.05)
    assert 0.02 <= np.mean(rej) <= 0.10, np.mean(rej)


def test_mcnemar_and_paired_bootstrap():
    y = np.array([1] * 30 + [0] * 30)
    a = y.copy()
    b = y.copy()
    b[:10] = 0                                                           # model b misses 10 positives a gets right
    r = mcnemar_exact(y, a, b)
    assert (r["only_1_correct"], r["only_2_correct"]) == (10, 0) and r["p"] == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar_exact(y, a, a)["p"] == 1.0
    ys, s = _scores(300, 1.5)
    pb = paired_bootstrap(ys, s, s - 3 * (ys == 1) * np.random.default_rng(0).random(len(ys)), auroc, n=300)
    assert pb["diff"] > 0 and pb["lo"] > 0 and pb["p"] < 0.05


def test_thresholds_are_chosen_on_validation_and_meet_the_sensitivity_target():
    yv, sv = _scores(1000, 1.2, seed=3)
    pv = 1 / (1 + np.exp(-sv))
    thr = choose_threshold(yv, pv, "sens_target", 0.90)
    assert point_metrics(yv, pv, thr)["sensitivity"] >= 0.90
    assert point_metrics(yv, pv, thr)["specificity"] > point_metrics(yv, pv, 0.0)["specificity"]
    assert choose_threshold(yv, pv, "fixed_0.5") == 0.5
    for m in ("youden", "f1"):
        assert 0 < choose_threshold(yv, pv, m) < 1
    assert choose_threshold(np.zeros(10), np.random.rand(10), "youden") == 0.5     # single-class validation falls back safely


def test_bootstrap_handles_degenerate_input_and_metrics_are_consistent():
    ci = bootstrap_ci(np.zeros(20), np.random.rand(20), 0.5, n=10)
    assert np.isnan(ci["auroc"][0])
    y, s = _scores(400)
    p = 1 / (1 + np.exp(-s))
    m = all_metrics(y, p, 0.5)
    ci = bootstrap_ci(y, p, 0.5, n=200)
    for k in ("auroc", "sensitivity", "specificity", "accuracy"):
        assert ci[k][0] <= m[k] <= ci[k][1]
    assert m["tp"] + m["fn"] == y.sum() and m["tn"] + m["fp"] == (1 - y).sum()

"""Significance tests for comparing two models on the SAME test pairs.

DeLong (fast, Sun & Xu 2014) for AUROC, exact McNemar for paired decisions, and a paired bootstrap for any metric.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as sps


def _delong_cov(y: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """preds: (k, n) scores for k models. Returns the k AUCs and their k x k DeLong covariance."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")  # positives first
    preds = np.atleast_2d(preds)[:, order]
    m = int(y.sum())
    n = preds.shape[1] - m
    if m == 0 or n == 0:
        raise ValueError("DeLong needs both classes")
    pos, neg = preds[:, :m], preds[:, m:]
    rank = lambda a: np.apply_along_axis(sps.rankdata, 1, a)  # noqa: E731 - average ranks == midranks
    tx, ty, tz = rank(pos), rank(neg), rank(preds)
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_ci(y: np.ndarray, p: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    aucs, cov = _delong_cov(y, np.asarray(p)[None])
    se = float(np.sqrt(cov[0, 0]))
    z = sps.norm.ppf(1 - alpha / 2)
    return float(aucs[0]), float(max(0.0, aucs[0] - z * se)), float(min(1.0, aucs[0] + z * se))


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict[str, float]:
    """Two-sided DeLong test of AUC(p1) == AUC(p2) on paired samples."""
    aucs, cov = _delong_cov(y, np.vstack([p1, p2]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return dict(auc1=float(aucs[0]), auc2=float(aucs[1]), diff=diff, z=float("nan"), p=1.0 if diff == 0 else 0.0)
    z = diff / np.sqrt(var)
    return dict(auc1=float(aucs[0]), auc2=float(aucs[1]), diff=diff, z=float(z), p=float(2 * sps.norm.sf(abs(z))))


def mcnemar_exact(y: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> dict[str, float]:
    c1, c2 = np.asarray(pred1) == np.asarray(y), np.asarray(pred2) == np.asarray(y)
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())  # b: only model 1 right, c: only model 2 right
    p = 1.0 if b + c == 0 else float(sps.binomtest(min(b, c), b + c, 0.5).pvalue)
    return dict(only_1_correct=b, only_2_correct=c, p=p)


def paired_bootstrap(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float], n: int = 1000, seed: int = 0) -> dict[str, float]:
    """Difference metric(p1) - metric(p2) with a percentile CI and a two-sided bootstrap p-value (stratified resampling)."""
    y, p1, p2 = np.asarray(y).astype(int), np.asarray(p1, float), np.asarray(p2, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = np.empty(n)
    for i in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs[i] = metric(y[idx], p1[idx]) - metric(y[idx], p2[idx])
    obs = metric(y, p1) - metric(y, p2)
    p = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    return dict(diff=float(obs), lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)), p=max(p, 1.0 / n))

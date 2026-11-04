"""Metrics, threshold selection and bootstrap confidence intervals.

Thresholds are ALWAYS chosen on validation data and then frozen for test: choosing them on the test set (as the
legacy evaluation did) inflates sensitivity/specificity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

POINT_KEYS = ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1")


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def confusion(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[int, int, int, int]:
    y, pred = np.asarray(y).astype(int), (np.asarray(p) >= thr).astype(int)
    return int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())


def point_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    tp, fp, tn, fn = confusion(y, p, thr)
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    sens, spec, prec = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    return dict(
        accuracy=div(tp + tn, tp + fp + tn + fn), sensitivity=sens, specificity=spec, precision=prec,
        balanced_accuracy=float(np.nanmean([sens, spec])), f1=div(2 * tp, 2 * tp + fp + fn), tp=tp, fp=fp, tn=tn, fn=fn,
    )


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    return dict(brier=float(np.mean((p - y) ** 2)), ece=float(ece))


def choose_threshold(y: np.ndarray, p: np.ndarray, method: str = "sens_target", sens_target: float = 0.90) -> float:
    y, p = np.asarray(y), np.asarray(p)
    if method.startswith("fixed_"):
        return float(method.split("_", 1)[1])
    if len(np.unique(y)) < 2:
        return 0.5
    if method == "youden":
        fpr, tpr, thr = roc_curve(y, p)
        return float(thr[np.argmax(tpr - fpr)])
    if method == "sens_target":  # most specific threshold that still reaches the target sensitivity
        fpr, tpr, thr = roc_curve(y, p)
        ok = np.flatnonzero(tpr >= sens_target)
        return float(thr[ok[0]]) if len(ok) else float(thr[-1])
    if method == "f1":
        prec, rec, thr = precision_recall_curve(y, p)
        f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        return float(thr[np.argmax(f1)])
    raise ValueError(f"unknown threshold method '{method}'")


def all_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    return dict(auroc=auroc(y, p), auprc=auprc(y, p), threshold=float(thr), n=int(len(y)), prevalence=float(np.mean(y)), **point_metrics(y, p, thr), **calibration(y, p))


def bootstrap_ci(y: np.ndarray, p: np.ndarray, thr: float, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    """Stratified bootstrap (resampling positives and negatives separately, so every replicate has both classes)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keys = ("auroc", "auprc") + POINT_KEYS
    draws = {k: [] for k in keys}
    if len(pos) == 0 or len(neg) == 0:
        return {k: (float("nan"), float("nan")) for k in keys}
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yy, pp = y[idx], p[idx]
        pm = point_metrics(yy, pp, thr)
        draws["auroc"].append(auroc(yy, pp))
        draws["auprc"].append(auprc(yy, pp))
        for k in POINT_KEYS:
            draws[k].append(pm[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def pct(v: list[float], q: float) -> float:
        a = np.asarray(v, float)
        return float(np.nanpercentile(a, q)) if np.isfinite(a).any() else float("nan")  # e.g. precision with no predicted positives

    return {k: (pct(v, lo), pct(v, hi)) for k, v in draws.items()}


def positive_probability(df: pd.DataFrame) -> np.ndarray:
    """Probability of the positive class from a prediction frame (binary: column ``p``)."""
    if "p" in df:
        return df["p"].to_numpy(float)
    raise ValueError("prediction frame has no 'p' column (multi-class evaluation is not supported by the binary metrics)")


def multiclass_summary(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    pred = probs.argmax(1)
    out = dict(accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro")), n=int(len(y)))
    try:
        out["macro_auroc_ovr"] = float(roc_auc_score(y, probs, multi_class="ovr", labels=list(range(probs.shape[1]))))
    except ValueError:
        out["macro_auroc_ovr"] = float("nan")
    return out

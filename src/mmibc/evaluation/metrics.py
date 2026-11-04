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
    """Brier score, ECE and the Brier SKILL score against always predicting the prevalence (> 0 means the probabilities beat that constant)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    brier = float(np.mean((p - y) ** 2))
    prev = float(y.mean())
    ref = prev * (1.0 - prev)
    return dict(brier=brier, ece=float(ece), brier_skill=float(1.0 - brier / ref) if ref > 0 else float("nan"))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, float)))  # overflow-free


def _nll(y: np.ndarray, z: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, z) - y * z))  # binary cross-entropy of sigmoid(z)


def fit_calibration(y: np.ndarray, logit: np.ndarray, method: str = "platt") -> dict:
    """Post-hoc recalibration of a binary classifier from VALIDATION data only: p = sigmoid(a * logit + b).

    temperature  b = 0, one parameter (fixes over-confidence)
    platt        a and b (additionally fixes a prior shift, e.g. the balanced training sampler vs. a 5 %-prevalence test set)
    A monotone map: AUROC and the ranking are unchanged. Falls back to the identity when validation lacks a class or the logit is anti-informative (a <= 0).
    """
    from scipy.optimize import minimize

    y, z = np.asarray(y, float), np.asarray(logit, float)
    ident = dict(method="none", a=1.0, b=0.0, nll_before=float("nan"), nll_after=float("nan"), n=int(len(y)))
    if method == "none" or len(np.unique(y)) < 2 or not np.isfinite(z).all():
        return ident
    before = _nll(y, z)
    if method == "temperature":
        f = lambda v: _nll(y, z / np.exp(v[0]))  # noqa: E731 - optimise log T so that T > 0
        r = minimize(f, x0=[0.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-10))
        a, b = float(np.exp(-r.x[0])), 0.0
    elif method == "platt":
        f = lambda v: _nll(y, v[0] * z + v[1])  # noqa: E731
        r = minimize(f, x0=[1.0, 0.0], method="BFGS")
        a, b = float(r.x[0]), float(r.x[1])
    else:
        raise ValueError(f"unknown calibration method '{method}' (none | temperature | platt)")
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        return {**ident, "nll_before": before, "nll_after": before, "note": "fit rejected (non-positive slope): identity used"}
    return dict(method=method, a=a, b=b, nll_before=before, nll_after=_nll(y, a * z + b), n=int(len(y)))


def apply_calibration(logit: np.ndarray, cal: dict) -> np.ndarray:
    return _sigmoid(cal["a"] * np.asarray(logit, float) + cal["b"])


def choose_threshold(y: np.ndarray, p: np.ndarray, method: str = "sens_target", sens_target: float = 0.90) -> float:
    """Pick a decision threshold on (validation) scores. For ``spec_target`` the 4th argument is the target SPECIFICITY."""
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
    if method == "spec_target":  # most sensitive threshold that still reaches the target specificity
        fpr, tpr, thr = roc_curve(y, p)
        ok = np.flatnonzero(fpr <= 1.0 - sens_target)  # ``sens_target`` carries the target value for this method
        t = float(thr[ok[-1]]) if len(ok) else float("inf")
        return t if np.isfinite(t) else float(np.nextafter(float(np.max(p)), np.inf))
    if method == "f1":
        prec, rec, thr = precision_recall_curve(y, p)
        f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        return float(thr[np.argmax(f1)])
    raise ValueError(f"unknown threshold method '{method}'")


OPERATING_POINTS = (("sens90", "sens_target", 0.90), ("sens95", "sens_target", 0.95), ("spec90", "spec_target", 0.90), ("spec95", "spec_target", 0.95), ("youden", "youden", 0.0))


def operating_points(y_val: np.ndarray, p_val: np.ndarray, y: np.ndarray, p: np.ndarray) -> dict[str, dict[str, float]]:
    """Sensitivity / specificity of ``p`` at thresholds chosen on validation for several clinically meaningful targets.

    Reporting several points is more informative than one (the literature quotes sensitivity at a fixed specificity and vice-versa) and the
    specificity-targeted thresholds are far more stable than the sensitivity-targeted ones when validation has few positives.
    """
    out = {}
    for name, method, target in OPERATING_POINTS:
        thr = choose_threshold(y_val, p_val, method, target)
        pm = point_metrics(y, p, thr)
        out[name] = dict(threshold=float(thr), **{k: pm[k] for k in ("sensitivity", "specificity", "precision", "balanced_accuracy", "tp", "fp", "tn", "fn")})
    return out


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

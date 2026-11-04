from __future__ import annotations

import numpy as np


def pointing_game(cam: np.ndarray, mask: np.ndarray) -> float:
    """1 if the arg-max of the map lies inside the lesion (Zhang et al. 2018)."""
    i = np.unravel_index(np.argmax(cam), cam.shape)
    return float(mask[i])


def energy_in_mask(cam: np.ndarray, mask: np.ndarray) -> float:
    """Share of the (non-negative) attribution mass that falls inside the lesion; chance level is the lesion area fraction."""
    tot = cam.sum()
    return float(cam[mask].sum() / tot) if tot > 0 else float("nan")


def iou_at(cam: np.ndarray, mask: np.ndarray, thr: float = 0.5) -> float:
    pred = cam >= thr
    union = (pred | mask).sum()
    return float((pred & mask).sum() / union) if union else float("nan")


def bootstrap_mean(x: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

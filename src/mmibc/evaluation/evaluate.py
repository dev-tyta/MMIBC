"""Final evaluation: frozen validation threshold, bootstrap CIs, and the modality-reliance controls."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mmibc.data.datasets import PairDataset
from mmibc.evaluation.metrics import all_metrics, bootstrap_ci, choose_threshold
from mmibc.evaluation.stats import delong_ci
from mmibc.training.trainer import predict

log = logging.getLogger(__name__)


def derange(n: int, seed: int) -> np.ndarray:
    """A permutation of range(n) with no fixed point (n > 1), deterministic."""
    if n < 2:
        return np.arange(n)
    idx = np.random.default_rng(seed).permutation(n)
    out = np.empty(n, dtype=int)
    out[idx] = np.roll(idx, 1)
    return out


def shuffled_partner_dataset(ds: PairDataset, seed: int) -> PairDataset:
    """Same anchors and targets, but every stream-b image replaced by another pair's: breaks the A<->B correspondence."""
    pairs = ds.pairs.copy()
    perm = derange(len(pairs), seed)
    for col in ("path_b", "uid_b"):
        pairs[col] = ds.pairs[col].to_numpy()[perm]
    pairs["pair_id"] = pairs["uid_a"].astype(str) + "|" + pairs["uid_b"].astype(str)
    return PairDataset(pairs, ds.root_a, ds.root_b, ds.tf_a, ds.tf_b)


def _split_metrics(pred: pd.DataFrame, thr: float, n_boot: int) -> dict:
    y, p = pred["y"].to_numpy(), pred["p"].to_numpy()
    m = all_metrics(y, p, thr)
    ci = bootstrap_ci(y, p, thr, n=n_boot)
    m["ci"] = {k: list(v) for k, v in ci.items()}
    if len(np.unique(y)) == 2:
        a, lo, hi = delong_ci(y, p)
        m["auroc_delong_ci"] = [lo, hi]
    return m


def evaluate_model(model, datasets: dict[str, PairDataset], cfg, run_dir: Path, device) -> dict:
    """Predict every split, pick the threshold on validation, report test (and ext_test) with CIs, then run controls.

    controls: none | drop_a | drop_b | shuffle_b  (reported on 'test').  If a control barely changes the score, the model
    ignores that stream or the pairing carries no usable information - that is the measurement, not a failure.
    """
    run_dir = Path(run_dir)
    ecfg, tcfg = cfg.eval, cfg.training
    if "val" not in datasets:
        raise ValueError("evaluation needs a 'val' split to choose the decision threshold")
    preds = {s: predict(model, ds, tcfg, device) for s, ds in datasets.items()}
    for s, df in preds.items():
        df.to_csv(run_dir / f"predictions_{s}.csv", index=False)
    pv = preds["val"]
    thr = choose_threshold(pv["y"].to_numpy(), pv["p"].to_numpy(), ecfg.threshold_method, float(ecfg.sens_target))
    out: dict = dict(threshold=thr, threshold_method=ecfg.threshold_method, threshold_chosen_on="val", splits={})
    for s, df in preds.items():
        out["splits"][s] = _split_metrics(df, thr, int(ecfg.bootstrap_samples))
    if "test" in datasets and len(ecfg.controls):
        base = out["splits"]["test"]["auroc"]
        rows = []
        for ctrl in ecfg.controls:
            if ctrl == "none":
                df = preds["test"]
            elif ctrl in ("drop_a", "drop_b"):
                df = predict(model, datasets["test"], tcfg, device, drop=ctrl[-1])
            elif ctrl == "shuffle_b":
                df = predict(model, shuffled_partner_dataset(datasets["test"], int(cfg.task.eval_pair_seed)), tcfg, device)
            else:
                raise ValueError(f"unknown control '{ctrl}'")
            m = _split_metrics(df, thr, max(50, int(ecfg.bootstrap_samples) // 4))
            rows.append(dict(control=ctrl, auroc=m["auroc"], auroc_lo=m["ci"]["auroc"][0], auroc_hi=m["ci"]["auroc"][1], delta_auroc=m["auroc"] - base, sensitivity=m["sensitivity"], specificity=m["specificity"]))
        out["controls"] = rows
        pd.DataFrame(rows).to_csv(run_dir / "controls.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    return out

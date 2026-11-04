"""Final evaluation: frozen validation threshold, bootstrap CIs, and the modality-reliance controls."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mmibc.data.datasets import PairDataset
from mmibc.evaluation.metrics import all_metrics, apply_calibration, bootstrap_ci, calibration, choose_threshold, fit_calibration, operating_points
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


def _split_metrics(pred: pd.DataFrame, thr: float, n_boot: int, pv: pd.DataFrame | None = None, ops: bool = False) -> dict:
    y, p = pred["y"].to_numpy(), pred["p"].to_numpy()
    m = all_metrics(y, p, thr)
    ci = bootstrap_ci(y, p, thr, n=n_boot)
    m["ci"] = {k: list(v) for k, v in ci.items()}
    if len(np.unique(y)) == 2:
        a, lo, hi = delong_ci(y, p)
        m["auroc_delong_ci"] = [lo, hi]
    if "p_raw" in pred:  # calibration quality BEFORE recalibration, so the effect of the fit stays visible
        raw = calibration(y, pred["p_raw"].to_numpy())
        m["raw"] = {f"raw_{k}": v for k, v in raw.items()}
    if ops and pv is not None:
        m["operating_points"] = operating_points(pv["y"].to_numpy(), pv["p"].to_numpy(), y, p)
    return m


def by_stratum(pred: pd.DataFrame, pairs: pd.DataFrame, thr: float) -> dict | None:
    """AUROC / sensitivity / specificity per value of the pair table's optional ``stratum`` column (e.g. the CMMD subset); None without it.

    A stratum that contains a single class has no AUROC (reported as null): that is itself the warning that the stratum predicts the label.
    """
    if "stratum" not in pairs.columns or "pair_id" not in pred.columns:
        return None
    strata = pairs.drop_duplicates("pair_id").set_index("pair_id")["stratum"].astype(str)
    label = pred["pair_id"].astype(str).map(strata)
    out: dict = {}
    for k in sorted(label.dropna().unique()):
        d = pred[label == k]
        y, p = d["y"].to_numpy().astype(int), d["p"].to_numpy()
        pos, neg = int((y == 1).sum()), int((y == 0).sum())
        row: dict = dict(n=int(len(d)), n_pos=pos, n_neg=neg, auroc=None, sensitivity=None, specificity=None)
        if pos and neg:
            row["auroc"] = float(all_metrics(y, p, thr)["auroc"])
        if pos:
            row["sensitivity"] = float((p[y == 1] >= thr).mean())
        if neg:
            row["specificity"] = float((p[y == 0] < thr).mean())
        out[k] = row
    return out or None


def _calibrated(df: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """Keep the raw probability as ``p_raw`` and replace ``p`` by the recalibrated one (every downstream consumer reads ``p``)."""
    out = df.copy()
    out["p_raw"] = out["p"]
    if cal["method"] != "none" and "logit" in out:
        out["p"] = apply_calibration(out["logit"].to_numpy(), cal)
    return out


def eval_fingerprint(cfg) -> str:
    """Hash of everything that changes what ``evaluate_model`` computes (so a finished run is re-evaluated, not retrained, when it changes)."""
    import hashlib

    blob = json.dumps({k: (dict(v) if hasattr(v, "keys") else v) for k, v in dict(cfg.eval).items()}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def evaluate_model(model, datasets: dict[str, PairDataset], cfg, run_dir: Path, device) -> dict:
    """Predict every split, recalibrate on validation, pick the threshold on validation, report test (and ext_test) with CIs, then run controls.

    Everything fitted here (calibration map, thresholds) uses the validation split only and is then frozen for the other splits.
    controls: none | drop_a | drop_b | shuffle_b  (reported on 'test').  If a control barely changes the score, the model
    ignores that stream or the pairing carries no usable information - that is the measurement, not a failure.
    """
    run_dir = Path(run_dir)
    ecfg, tcfg = cfg.eval, cfg.training
    if "val" not in datasets:
        raise ValueError("evaluation needs a 'val' split to choose the decision threshold")
    raw_preds = {s: predict(model, ds, tcfg, device) for s, ds in datasets.items()}
    pv_raw = raw_preds["val"]
    cal = fit_calibration(pv_raw["y"].to_numpy(), pv_raw["logit"].to_numpy(), str(ecfg.get("calibration", "none"))) if "logit" in pv_raw else fit_calibration([], [], "none")
    preds = {s: _calibrated(df, cal) for s, df in raw_preds.items()}
    for s, df in preds.items():
        df.to_csv(run_dir / f"predictions_{s}.csv", index=False)
    pv = preds["val"]
    thr = choose_threshold(pv["y"].to_numpy(), pv["p"].to_numpy(), ecfg.threshold_method, float(ecfg.spec_target if ecfg.threshold_method == "spec_target" else ecfg.sens_target))
    ops = bool(ecfg.get("operating_points", False))
    out: dict = dict(threshold=thr, threshold_method=ecfg.threshold_method, threshold_chosen_on="val", calibration=cal, eval_fingerprint=eval_fingerprint(cfg), splits={})
    for s, df in preds.items():
        out["splits"][s] = _split_metrics(df, thr, int(ecfg.bootstrap_samples), pv, ops)
        strata = by_stratum(df, datasets[s].pairs, thr)
        if strata:
            out["splits"][s]["by_stratum"] = strata
    if "test" in datasets and len(ecfg.controls):
        base = out["splits"]["test"]["auroc"]
        rows = []
        for ctrl in ecfg.controls:
            if ctrl == "none":
                df = preds["test"]
            elif ctrl in ("drop_a", "drop_b"):
                df = _calibrated(predict(model, datasets["test"], tcfg, device, drop=ctrl[-1]), cal)
            elif ctrl == "shuffle_b":
                df = _calibrated(predict(model, shuffled_partner_dataset(datasets["test"], int(cfg.task.eval_pair_seed)), tcfg, device), cal)
            else:
                raise ValueError(f"unknown control '{ctrl}'")
            m = _split_metrics(df, thr, max(50, int(ecfg.bootstrap_samples) // 4))
            rows.append(dict(control=ctrl, auroc=m["auroc"], auroc_lo=m["ci"]["auroc"][0], auroc_hi=m["ci"]["auroc"][1], delta_auroc=m["auroc"] - base, sensitivity=m["sensitivity"], specificity=m["specificity"]))
        out["controls"] = rows
        pd.DataFrame(rows).to_csv(run_dir / "controls.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    return out

"""Compare runs on identical test pairs: late fusion from unimodal runs, DeLong / McNemar against a reference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mmibc.evaluation.metrics import all_metrics, auroc
from mmibc.evaluation.stats import delong_ci, delong_test, mcnemar_exact


def late_fusion(pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> pd.DataFrame:
    """Average the probabilities of two unimodal runs scored on the SAME pairs (the paper's late-fusion baseline)."""
    if not (pred_a["pair_id"].to_numpy() == pred_b["pair_id"].to_numpy()).all() or not (pred_a["y"].to_numpy() == pred_b["y"].to_numpy()).all():
        raise ValueError("the two prediction tables do not cover the same pairs in the same order: re-run both on the frozen test pairs")
    out = pred_a.copy()
    out["p"] = 0.5 * (pred_a["p"].to_numpy() + pred_b["p"].to_numpy())
    return out


def compare_runs(preds: dict[str, pd.DataFrame], thresholds: dict[str, float], reference: str) -> pd.DataFrame:
    """One row per model: AUROC (DeLong CI), Delta vs reference with DeLong p, McNemar p on thresholded decisions."""
    ref = preds[reference]
    y = ref["y"].to_numpy()
    rows = []
    for name, df in preds.items():
        if not (df["pair_id"].to_numpy() == ref["pair_id"].to_numpy()).all():
            raise ValueError(f"'{name}' was not scored on the same pairs as the reference '{reference}'")
        p = df["p"].to_numpy()
        a, lo, hi = delong_ci(y, p)
        m = all_metrics(y, p, thresholds[name])
        row = dict(model=name, auroc=a, auroc_lo=lo, auroc_hi=hi, sensitivity=m["sensitivity"], specificity=m["specificity"], f1=m["f1"])
        if name != reference:
            d = delong_test(y, p, ref["p"].to_numpy())
            mc = mcnemar_exact(y, (p >= thresholds[name]).astype(int), (ref["p"].to_numpy() >= thresholds[reference]).astype(int))
            row.update(delta_auroc=d["diff"], delong_p=d["p"], mcnemar_p=mc["p"])
        rows.append(row)
    return pd.DataFrame(rows)


def load_run(run_dir: Path, split: str = "test") -> tuple[pd.DataFrame, float]:
    run_dir = Path(run_dir)
    return pd.read_csv(run_dir / f"predictions_{split}.csv"), float(json.loads((run_dir / "metrics.json").read_text())["threshold"])

"""Run a named set of configurations (an ablation) and compare them on the SAME frozen test pairs."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from mmibc.evaluation.compare import compare_runs, late_fusion, load_run
from mmibc.pipeline import run_dir_for, run_experiment
from mmibc.utils.config import config_dir, load_config
from mmibc.utils.paths import ProjectPaths

log = logging.getLogger(__name__)


def load_suite(name_or_path: str | Path) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = config_dir() / "suites" / (str(name_or_path) if str(name_or_path).endswith(".yaml") else f"{name_or_path}.yaml")
    return OmegaConf.to_container(OmegaConf.load(p), resolve=True)


def run_suite(suite: str | Path, paths: ProjectPaths, overrides: list[str] = (), only: list[str] | None = None) -> pd.DataFrame:
    spec = load_suite(suite)
    tag = Path(str(suite)).stem
    out = paths.runs / f"suite_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    runs, failures = {}, {}
    for r in spec["runs"]:
        if only and r["name"] not in only:
            continue
        cfg = load_config(spec["config"], [*overrides, *r.get("set", []), f"run_name={tag}_{r['name']}"])
        try:
            run_experiment(cfg, paths)
            runs[r["name"]] = run_dir_for(cfg, paths)
        except Exception as exc:  # noqa: BLE001
            if not spec.get("continue_on_error", False):
                raise
            failures[r["name"]] = f"{type(exc).__name__}: {exc}"
            log.error("run '%s' failed and was skipped: %s", r["name"], failures[r["name"]])
            (out / f"FAILED_{r['name']}.txt").write_text(traceback.format_exc())
    res = summarize(runs, spec.get("reference"))
    for name, msg in failures.items():
        res = pd.concat([res, pd.DataFrame([dict(model=name, error=msg)])], ignore_index=True)
    res.to_csv(out / "results.csv", index=False)
    (out / "results.md").write_text(res.to_markdown(index=False, floatfmt=".4f") if hasattr(res, "to_markdown") else res.to_string())
    return res


def summarize(runs: dict[str, Path], reference: str | None) -> pd.DataFrame:
    preds, thr = {}, {}
    for name, d in runs.items():
        preds[name], thr[name] = load_run(d, "test")
    if "unimodal_a" in preds and "unimodal_b" in preds:  # late fusion from two independent unimodal runs
        preds["late_fusion"] = late_fusion(preds["unimodal_a"], preds["unimodal_b"])
        thr["late_fusion"] = 0.5 * (thr["unimodal_a"] + thr["unimodal_b"])
    if not preds:
        return pd.DataFrame()
    ref = reference if reference in preds else next(iter(preds))
    return compare_runs(preds, thr, ref)

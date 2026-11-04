"""Run a named set of configurations (an ablation) and compare them on the SAME frozen test pairs.

A suite is a YAML file (configs/suites/*.yaml): a base config, an optional reference, optional ``seeds``, and a list of runs, each a set of
dotted overrides. Suites are built for Colab, where a runtime can disappear in the middle of an hour-long job:

* every run is resumable (``last.pt``) and a finished run is skipped, so re-running the same command simply continues;
* ``status.csv`` is rewritten after every run, so a half-finished suite already shows what is done;
* a run whose backbone cannot be loaded right now (a gated DINOv3 model without a Hugging Face token) is SKIPPED before anything is built,
  and a run that fails for another reason is recorded as FAILED when the suite says ``continue_on_error``; both show up in ``results.csv``
  and are retried by the next call, while finished runs are not repeated;
* with ``seeds`` each run is repeated per seed on IDENTICAL train/val/test partitions (``data.split_seed`` is fixed), and the summary reports
  mean +- sd across seeds and the AUROC of the seed ensemble. Single-seed differences of 0.01-0.03 AUROC are inside the confidence intervals
  of a test set of a few hundred images: do not read them as effects.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from mmibc.evaluation.compare import compare_runs, ensemble, fusion_gain, late_fusion, load_run
from mmibc.evaluation.metrics import all_metrics
from mmibc.evaluation.stats import delong_ci
from mmibc.pipeline import run_dir_for, run_experiment
from mmibc.utils.config import config_dir, load_config
from mmibc.utils.hf import unavailable_reason
from mmibc.utils.paths import ProjectPaths

log = logging.getLogger(__name__)


def load_suite(name_or_path: str | Path) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = config_dir() / "suites" / (str(name_or_path) if str(name_or_path).endswith(".yaml") else f"{name_or_path}.yaml")
    return OmegaConf.to_container(OmegaConf.load(p), resolve=True)


def _split_seed_override(spec: dict, overrides: list[str], run_set: list[str], seeds: list[int]) -> list[str]:
    """Training seeds must be compared on one partition: pin ``data.split_seed`` unless the user or the run already did."""
    given = any(o.startswith("data.split_seed=") for o in [*overrides, *run_set])
    return [] if given else [f"data.split_seed={spec.get('split_seed', seeds[0])}"]


def run_suite(
    suite: str | Path, paths: ProjectPaths, overrides: list[str] = (), only: list[str] | None = None, seeds: list[int] | None = None, force: bool = False,
) -> pd.DataFrame:
    spec = load_suite(suite)
    tag = Path(str(suite)).stem
    seeds = [int(s) for s in (seeds or spec.get("seeds") or [])]
    overrides = list(overrides)
    out = paths.runs / f"suite_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    runs: dict[int | None, dict[str, Path]] = {}
    failures: dict[str, str] = {}
    skipped: dict[str, str] = {}
    status: list[dict] = []
    for r in spec["runs"]:
        if only and r["name"] not in only:
            continue
        for sd in seeds or [None]:
            name = r["name"] if sd is None else f"{r['name']}_s{sd}"
            extra = [] if sd is None else [f"seed={sd}", *_split_seed_override(spec, overrides, list(r.get("set", [])), seeds)]
            cfg = load_config(spec["config"], [*overrides, *r.get("set", []), *extra, f"run_name={tag}_{name}"])
            reason = unavailable_reason(str(cfg.model.backbone))
            if reason:  # cheap check first: no tables, no model, no hours lost on a run that cannot start
                skipped[name] = reason
                status.append(dict(run=name, state="SKIPPED: backbone unavailable (no Hugging Face token)"))
                log.warning("run '%s' skipped: %s", name, reason)
                pd.DataFrame(status).to_csv(out / "status.csv", index=False)
                continue
            t0 = time.time()
            try:
                res = run_experiment(cfg, paths, force=force)
                runs.setdefault(sd, {})[r["name"]] = run_dir_for(cfg, paths)
                status.append(dict(run=name, state="done", test_auroc=res["splits"]["test"]["auroc"], minutes=round((time.time() - t0) / 60, 1)))
                (out / f"FAILED_{name}.txt").unlink(missing_ok=True)  # a retry that worked: the old traceback is stale
            except Exception as exc:  # noqa: BLE001
                if not spec.get("continue_on_error", False):
                    pd.DataFrame(status + [dict(run=name, state=f"FAILED: {type(exc).__name__}")]).to_csv(out / "status.csv", index=False)
                    raise
                failures[name] = f"{type(exc).__name__}: {exc}"
                status.append(dict(run=name, state=f"FAILED: {type(exc).__name__}"))
                log.error("run '%s' failed and was skipped: %s", name, failures[name])
                (out / f"FAILED_{name}.txt").write_text(traceback.format_exc())
            pd.DataFrame(status).to_csv(out / "status.csv", index=False)  # a disconnect from here on loses nothing that is already done
    res = summarize_seeds(runs, spec.get("reference"), out) if seeds else summarize(runs.get(None, {}), spec.get("reference"), out)
    for name, msg in failures.items():
        res = pd.concat([res, pd.DataFrame([dict(model=name, error=msg)])], ignore_index=True)
    for name, msg in skipped.items():
        res = pd.concat([res, pd.DataFrame([dict(model=name, error="SKIPPED: " + msg)])], ignore_index=True)
    if skipped:
        log.warning("%d run(s) skipped (%s). Set HF_TOKEN and call the same command again: finished runs are kept, only these are trained.", len(skipped), ", ".join(skipped))
    if not runs and (skipped or failures):
        log.error("no run of suite '%s' finished (%d skipped, %d failed); see %s", tag, len(skipped), len(failures), out / "status.csv")
    res.to_csv(out / "results.csv", index=False)
    (out / "results.md").write_text(res.to_markdown(index=False, floatfmt=".4f") if hasattr(res, "to_markdown") else res.to_string())
    return res


def _load(runs: dict[str, Path], split: str) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    preds, thr = {}, {}
    for name, d in runs.items():
        preds[name], thr[name] = load_run(d, split)
    return preds, thr


def summarize(runs: dict[str, Path], reference: str | None, out: Path | None = None) -> pd.DataFrame:
    """One row per model on the shared test pairs (DeLong CI, ΔAUROC vs the reference, McNemar); fusion-gain table when both unimodal runs exist."""
    preds, thr = _load(runs, "test")
    if not preds:
        return pd.DataFrame()
    if "unimodal_a" in preds and "unimodal_b" in preds:  # late fusion from two independent unimodal runs
        preds["late_fusion"] = late_fusion(preds["unimodal_a"], preds["unimodal_b"])
        thr["late_fusion"] = 0.5 * (thr["unimodal_a"] + thr["unimodal_b"])
        if out is not None:
            val, _ = _load({k: v for k, v in runs.items() if k in ("unimodal_a", "unimodal_b")}, "val")
            gain = fusion_gain(preds, val)
            if not gain.empty:
                gain.to_csv(out / "fusion_gain.csv", index=False)
    ref = reference if reference in preds else next(iter(preds))
    return compare_runs(preds, thr, ref)


def summarize_seeds(runs: dict[int | None, dict[str, Path]], reference: str | None, out: Path) -> pd.DataFrame:
    """Per-seed tables (paired within a seed) plus mean +- sd across seeds and the AUROC of the seed ensemble."""
    per_seed = {}
    for sd, rr in runs.items():
        if rr:
            per_seed[sd] = summarize(rr, reference, out if sd == min(runs) else None)
            per_seed[sd].assign(seed=sd).to_csv(out / f"results_s{sd}.csv", index=False)
    names = list(dict.fromkeys(m for df in per_seed.values() for m in df["model"]))  # models in the order they first appear
    rows = []
    for m in names:
        a = np.array([float(df.loc[df["model"] == m, "auroc"].iloc[0]) for df in per_seed.values() if m in set(df["model"])])
        row = dict(model=m, n_seeds=len(a), auroc_mean=float(a.mean()), auroc_sd=float(a.std(ddof=1)) if len(a) > 1 else float("nan"), auroc_min=float(a.min()), auroc_max=float(a.max()))
        members = [rr[m] for rr in runs.values() if m in rr]
        if len(members) > 1:  # the ensemble of the seeds: usually the single most reliable improvement available
            ens = ensemble([load_run(d, "test")[0] for d in members])
            y, p = ens["y"].to_numpy(), ens["p"].to_numpy()
            thr = float(np.mean([load_run(d, "test")[1] for d in members]))
            _, lo, hi = delong_ci(y, p)
            row.update(ens_auroc=all_metrics(y, p, thr)["auroc"], ens_lo=lo, ens_hi=hi)
        rows.append(row)
    return pd.DataFrame(rows)

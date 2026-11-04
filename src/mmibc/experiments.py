"""Run the whole experiment programme on EVERY dataset that is available, with one command.

    mmibc experiments                   # configs/programs/all.yaml: prepare what exists, then every track in turn
    mmibc experiments --list            # what would run, what is already finished, what is blocked and why (touches nothing)
    mmibc experiments --suites          # also the multi-run suites (fusion ladders, backbone comparison)
    mmibc experiments --only cmmd       # tracks by name or group

A *program* (``configs/programs/*.yaml``) is a list of *tracks*; a track is either one run (a config plus overrides) or a suite. For each track
the datasets it needs are derived from its configuration (mammogram source, and the ultrasound sources when it pairs mammograms with
ultrasound), so the program never has to be told what you downloaded:

* datasets that are present are prepared once, up front (resumable; the ultrasound pool is built from the UNION of the sources the selected
  tracks need, so a later track never finds a source missing and an earlier one never trains on a source it did not list);
* a track whose dataset is not there is SKIPPED with the reason, and a track that fails is recorded as FAILED with its traceback; neither stops
  the others. Download or fix the cause and call the same command again: finished tracks are reused, only the missing ones are trained;
* an unavailable gated backbone (DINOv3 without a Hugging Face token) skips the rows that need it, as in ``mmibc suite``;
* ``runs/experiments_<program>/`` holds ``plan.csv`` (before), ``status.csv`` (after every track, so a Colab disconnect loses nothing),
  ``summary.csv`` / ``summary.md`` (one row per track: test AUROC with its DeLong interval, external-test AUROC, per-subset AUROC) and
  ``FAILED_<track>.txt``.

Datasets are never pooled into one training set: VinDr-Mammo labels are BI-RADS suspicion (with an official split), CMMD labels are biopsy
results (patient-level split we carve ourselves), so numbers on the two are separate results, not points on one curve.
"""

from __future__ import annotations

import gc
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from mmibc.pipeline import US, VINDR, finished_metrics, mammo_source_of, prepare, run_dir_for, run_experiment, run_xai
from mmibc.suite import load_suite, run_suite
from mmibc.utils.config import config_dir, load_config
from mmibc.utils.hf import unavailable_reason
from mmibc.utils.paths import ProjectPaths

log = logging.getLogger(__name__)

READY, RAW, MISSING = "ready", "raw", "missing"
US_PREFIX = "us:"


# ------------------------------------------------------------------------------------------------ the program
@dataclass
class Track:
    name: str
    config: str | None = None                      # a run track: config + overrides
    suite: str | None = None                       # a suite track: a configs/suites name or a path
    set: list[str] = field(default_factory=list)
    group: str = ""
    optional: bool = False                         # only with ``suites=True`` (or when named in ``only``)

    @property
    def kind(self) -> str:
        return "suite" if self.suite else "run"


def load_program(name_or_path: str | Path) -> tuple[dict, list[Track]]:
    p = Path(str(name_or_path))
    if not p.exists():
        stem = str(name_or_path) if str(name_or_path).endswith(".yaml") else f"{name_or_path}.yaml"
        p = config_dir() / "programs" / stem
    if not p.exists():
        raise FileNotFoundError(f"program '{name_or_path}' not found (looked in {p.parent}); available: {sorted(x.stem for x in p.parent.glob('*.yaml'))}")
    spec = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
    tracks = []
    for t in spec.get("tracks", []):
        if bool(t.get("config")) == bool(t.get("suite")):
            raise ValueError(f"track '{t.get('name')}' in {p.name} needs exactly one of 'config' (a run) or 'suite'")
        tracks.append(Track(name=t["name"], config=t.get("config"), suite=t.get("suite"), set=[str(x) for x in t.get("set", [])],
                            group=str(t.get("group", "")), optional=bool(t.get("optional", False))))
    names = [t.name for t in tracks]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate track names in {p.name}: {sorted({n for n in names if names.count(n) > 1})}")
    spec.setdefault("name", p.stem)
    return spec, tracks


def select_tracks(tracks: list[Track], only: list[str] | None = None, skip: list[str] | None = None, suites: bool = False) -> list[Track]:
    """``only``/``skip`` match a track name or a group. Optional (suite) tracks run with ``suites`` or when ``only`` names them."""
    known = {t.name for t in tracks} | {t.group for t in tracks if t.group}
    bad = [x for x in [*(only or []), *(skip or [])] if x not in known]
    if bad:
        raise ValueError(f"unknown track/group {bad}; available tracks: {[t.name for t in tracks]}; groups: {sorted(g for g in known if g not in {t.name for t in tracks})}")
    chosen = []
    for t in tracks:
        named = bool(only) and (t.name in only or t.group in only)
        if only and not named:
            continue
        if skip and (t.name in skip or t.group in skip):
            continue
        if t.optional and not (suites or named):
            continue
        chosen.append(t)
    return chosen


def track_configs(track: Track, overrides: list[str]) -> list[tuple[str, object]]:
    """[(label, cfg)] of everything a track trains: one entry for a run, one per suite row for a suite."""
    if track.kind == "run":
        return [(track.name, load_config(track.config, [*overrides, *track.set]))]
    spec = load_suite(track.suite)
    return [(f"{track.name}/{r['name']}", load_config(spec["config"], [*overrides, *track.set, *r.get("set", [])])) for r in spec["runs"]]


def needs_of(cfg) -> list[str]:
    """The datasets a configuration reads: its mammogram source, plus each ultrasound source when it pairs mammograms with ultrasound."""
    need = [mammo_source_of(cfg)]
    if cfg.task.kind == "synthetic_pairs":
        need += [f"{US_PREFIX}{k}" for k in cfg.data.us_sources]
    return need


# ------------------------------------------------------------------------------------------------ data state and preparation
def _non_empty(d: Path) -> bool:
    try:
        return d.is_dir() and any(True for _ in d.iterdir())
    except OSError:
        return False


def processed_us_sources(paths: ProjectPaths) -> set[str]:
    f = paths.processed / US / "manifest.csv"
    if not f.exists():
        return set()
    return set(pd.read_csv(f, usecols=["source"])["source"].astype(str))


def dataset_state(need: str, paths: ProjectPaths, us_have: set[str] | None = None) -> str:
    """``ready`` (processed), ``raw`` (downloaded, not prepared yet) or ``missing``."""
    if need.startswith(US_PREFIX):
        key = need[len(US_PREFIX):]
        have = processed_us_sources(paths) if us_have is None else us_have
        return READY if key in have else RAW if _non_empty(paths.raw / key) else MISSING
    if (paths.processed / need / "manifest.csv").exists():
        return READY
    raw = [paths.raw / "vindr_images", paths.raw / "vindr_annotations"] if need == VINDR else [paths.raw / need]
    return RAW if all(_non_empty(d) for d in raw) else MISSING


def prepare_datasets(needs: set[str], paths: ProjectPaths, overrides: list[str]) -> dict[str, str]:
    """Prepare every needed dataset that is downloaded but not processed. Returns {need: 'ready' | 'missing' | 'FAILED: ...'}.

    One dataset failing (a corrupt download, a missing optional package) never blocks the others.
    """
    out: dict[str, str] = {}
    cfg = load_config(None, overrides)
    for need in sorted(n for n in needs if not n.startswith(US_PREFIX)):
        state = dataset_state(need, paths)
        if state == RAW:
            log.info("preparing %s (first time: this reads the raw files and can take a while; it is resumable)", need)
            try:
                prepare(cfg, paths, (need,))
                state = READY
            except Exception as exc:  # noqa: BLE001
                log.error("preparing %s failed: %s", need, exc)
                out[need] = f"FAILED: {type(exc).__name__}: {exc}"
                continue
        out[need] = state
    us_need = sorted(n[len(US_PREFIX):] for n in needs if n.startswith(US_PREFIX))
    if us_need:
        out.update(_prepare_ultrasound(us_need, paths, overrides))
    return out


def _prepare_ultrasound(keys: list[str], paths: ProjectPaths, overrides: list[str]) -> dict[str, str]:
    have = processed_us_sources(paths)
    todo = [k for k in keys if k not in have and dataset_state(US_PREFIX + k, paths, have) == RAW]
    res = {US_PREFIX + k: (READY if k in have else MISSING) for k in keys}
    if not todo:
        return res
    # The manifest is rebuilt from the sources scanned in one call, so scan everything needed that has raw files (already-processed images are
    # not redone) and carry over the rows of processed sources whose raw folder is gone.
    scan = [k for k in dict.fromkeys([*keys, *sorted(have)]) if _non_empty(paths.raw / k)]
    old_f = paths.processed / US / "manifest.csv"
    old = pd.read_csv(old_f) if old_f.exists() else None
    log.info("preparing ultrasound sources %s (already processed: %s)", scan, sorted(have))
    try:
        prepare(load_config(None, [*overrides, f"data.us_sources=[{','.join(scan)}]"]), paths, (US,))
    except Exception as exc:  # noqa: BLE001  (the manifest is only replaced at the very end of a successful prepare, so the old pool is intact)
        log.error("preparing ultrasound failed: %s", exc)
        return {US_PREFIX + k: READY if k in have else f"FAILED: {type(exc).__name__}: {exc}" for k in keys}
    if old is not None:
        new = pd.read_csv(old_f)
        lost = old[~old["source"].astype(str).isin(set(new["source"].astype(str)))]
        if len(lost):
            log.info("keeping the processed rows of %s (their raw folders are not here any more)", sorted(set(lost["source"].astype(str))))
            pd.concat([new, lost], ignore_index=True).to_csv(old_f, index=False)
    have = processed_us_sources(paths)
    return {US_PREFIX + k: READY if k in have else MISSING for k in keys}


# ------------------------------------------------------------------------------------------------ plan
def _backbones(cfgs: list[tuple[str, object]]) -> list[str]:
    return list(dict.fromkeys(str(c.model.backbone) for _, c in cfgs))


def build_plan(tracks: list[Track], paths: ProjectPaths, overrides: list[str]) -> pd.DataFrame:
    """One row per track: the datasets it needs and their state, the backbone, and whether it is already finished. Touches nothing."""
    us_have = processed_us_sources(paths)
    rows = []
    for t in tracks:
        cfgs = track_configs(t, overrides)
        needs = list(dict.fromkeys(n for _, c in cfgs for n in needs_of(c)))
        state = {n: dataset_state(n, paths, us_have) for n in needs}
        blocked = [n for n, s in state.items() if s == MISSING]
        row = dict(track=t.name, kind=t.kind, group=t.group, needs=" ".join(needs), data=" ".join(f"{n}={s}" for n, s in state.items()), backbone=" ".join(_backbones(cfgs)))
        if t.kind == "run":
            cfg = cfgs[0][1]
            run = run_dir_for(cfg, paths)
            row["run_dir"] = run.name
            reason = unavailable_reason(str(cfg.model.backbone))
            finished = finished_metrics(cfg, run) is not None
            row["plan"] = "already finished (reused)" if finished else f"blocked: {', '.join(blocked)} not downloaded" if blocked else f"blocked: {reason}" if reason else "will run"
        else:
            row["plan"] = f"blocked: {', '.join(blocked)} not downloaded" if blocked else "will run (resumable; finished rows are reused)"
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ results
def _free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _delong(m: dict) -> tuple[float | None, float | None]:
    ci = m.get("auroc_delong_ci") or (m.get("ci") or {}).get("auroc")
    return (float(ci[0]), float(ci[1])) if ci else (None, None)


def result_row(metrics: dict) -> dict:
    """The numbers a reader wants from a finished run's metrics.json."""
    s = metrics["splits"]
    t = s["test"]
    lo, hi = _delong(t)
    row = dict(auroc=t["auroc"], auroc_lo=lo, auroc_hi=hi, sensitivity=t.get("sensitivity"), specificity=t.get("specificity"), n_test=t.get("n"))
    if "ext_test" in s:
        elo, ehi = _delong(s["ext_test"])
        row.update(ext_auroc=s["ext_test"]["auroc"], ext_lo=elo, ext_hi=ehi)
    for k, v in (t.get("by_stratum") or {}).items():
        row[f"auroc_{k}"] = v.get("auroc")
    return row


FRONT = ["track", "kind", "group", "state", "backbone", "auroc", "auroc_lo", "auroc_hi", "sensitivity", "specificity", "ext_auroc", "ext_lo", "ext_hi"]
BACK = ["n_test", "reused", "xai", "minutes", "run_dir", "results"]


def _pretty(df: pd.DataFrame) -> pd.DataFrame:
    """Readable column order and number formats for summary.md (summary.csv keeps the raw numbers)."""
    subsets = sorted(c for c in df.columns if c.startswith("auroc_") and c not in FRONT)
    cols = [c for c in [*FRONT, *subsets] if c in df.columns]
    cols += [c for c in df.columns if c not in cols and c not in BACK] + [c for c in BACK if c in df.columns]
    out = df[cols].copy().astype(object)
    for c in cols:
        if c in ("n_test",) or c.startswith(("rows_", "suite_rows")):
            out[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
        elif c == "minutes":
            out[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
        elif c == "reused":
            out[c] = df[c].map(lambda v: "" if pd.isna(v) else "yes" if bool(v) else "no")
        elif pd.api.types.is_float_dtype(df[c]):
            out[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return out.fillna("")


def _md(df: pd.DataFrame) -> str:
    df = _pretty(df)
    try:
        return df.to_markdown(index=False, disable_numparse=True)   # numparse would turn 0.540 back into 0.54
    except Exception:  # noqa: BLE001  (tabulate missing)
        return df.to_string(index=False)


CAVEAT = ("VinDr-Mammo labels are BI-RADS suspicion and CMMD labels are biopsy results: the two are separate results and are never pooled into one training set. "
          "CMMD's subset D2 is ~98 % malignant; read the `auroc_D1` / `auroc_D2` columns and prefer the D1-only tracks. Differences of 0.01-0.03 AUROC are inside the "
          "confidence intervals of test sets this size.")


def _write(out: Path, program: dict, status: list[dict], rows: list[dict]) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(status).to_csv(out / "status.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    head = f"# {program.get('name', 'experiments')}\n\n{program.get('description', '').strip()}\n\n{CAVEAT}\n\n"
    (out / "summary.md").write_text(head + (_md(summary) if len(summary) else "(nothing has finished yet)") + "\n")
    return summary


# ------------------------------------------------------------------------------------------------ the run
def run_program(
    program: str | Path, paths: ProjectPaths, overrides: list[str] = (), only: list[str] | None = None, skip: list[str] | None = None,
    suites: bool = False, xai: bool = False, force: bool = False, seeds: list[int] | None = None, dry_run: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Prepare what exists and run every selected track. Returns (summary table, exit code: 0 ok, 1 something FAILED or nothing ran, 2 interrupted)."""
    spec, tracks = load_program(program)
    tracks = select_tracks(tracks, only, skip, suites)
    overrides = list(overrides)
    plan = build_plan(tracks, paths, overrides)
    if dry_run:
        return plan, 0
    out = paths.runs / f"experiments_{Path(str(program)).stem}"
    out.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out / "plan.csv", index=False)

    needs = {n for t in tracks for _, c in track_configs(t, overrides) for n in needs_of(c)}
    prep = prepare_datasets(needs, paths, overrides)
    log.info("datasets: %s", ", ".join(f"{k}={'FAILED' if v.startswith('FAILED') else v}" for k, v in sorted(prep.items())))

    status: list[dict] = []
    rows: list[dict] = []
    rc = 0
    n_done = 0
    for i, t in enumerate(tracks, 1):
        log.info("=== track %d/%d: %s ===", i, len(tracks), t.name)
        cfgs = track_configs(t, overrides)
        bad = [(n, prep.get(n, MISSING)) for n in dict.fromkeys(n for _, c in cfgs for n in needs_of(c)) if prep.get(n, MISSING) != READY]
        base = dict(track=t.name, kind=t.kind, group=t.group)
        if bad:
            failed = [f"{n}: {s[len('FAILED: '):].splitlines()[0][:300]}" for n, s in bad if s.startswith("FAILED")]
            state = f"FAILED: dataset {failed[0]}" if failed else "SKIPPED: " + ", ".join(n for n, _ in bad) + " not downloaded (see `mmibc download`; rerun this command afterwards)"
            rc = 1 if failed else rc
            status.append(dict(**base, state=state))
            rows.append(dict(**base, state=state))
            log.warning("track %s: %s", t.name, state)
            _write(out, spec, status, rows)
            continue
        t0 = time.time()
        try:
            if t.kind == "run":
                srow, state = _run_track(t, cfgs[0][1], paths, force, xai, out)
            else:
                srow, state = _suite_track(t, paths, overrides, seeds, force)
        except KeyboardInterrupt:
            status.append(dict(**base, state="INTERRUPTED"))
            _write(out, spec, status, rows)
            raise
        except Exception as exc:  # noqa: BLE001
            state = f"FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
            srow = {}
            rc = 1
            (out / f"FAILED_{t.name}.txt").write_text(traceback.format_exc())
            log.error("track %s failed and was skipped: %s", t.name, state)
        else:
            (out / f"FAILED_{t.name}.txt").unlink(missing_ok=True)  # a retry that worked: the old traceback is stale
        row = dict(**base, state=state, **srow, minutes=round((time.time() - t0) / 60, 1))
        status.append(dict(**base, state=state))
        rows.append(row)
        _write(out, spec, status, rows)
        _free_memory()
        if state == "INTERRUPTED":
            rc = 2
            break
        n_done += state.startswith("done") or state.startswith("partial")
    summary = _write(out, spec, status, rows)
    if rc == 0 and not n_done and rows:
        rc = 1
        log.error("no track finished; see %s", out / "status.csv")
    skipped = [r["track"] for r in rows if str(r["state"]).startswith("SKIPPED")]
    if skipped:
        log.warning("%d track(s) skipped for missing data: %s. Download the datasets and call the same command again: finished tracks are kept.", len(skipped), ", ".join(skipped))
    return summary, rc


def _run_track(t: Track, cfg, paths: ProjectPaths, force: bool, xai: bool, out: Path) -> tuple[dict, str]:
    run = run_dir_for(cfg, paths)
    reason = unavailable_reason(str(cfg.model.backbone))
    if reason:
        return dict(run_dir=run.name, backbone=str(cfg.model.backbone)), "SKIPPED: " + reason
    reused = not force and cfg.training.resume and finished_metrics(cfg, run) is not None
    res = run_experiment(cfg, paths, force=force)
    if res.get("interrupted"):
        return dict(run_dir=run.name), "INTERRUPTED"
    row = dict(reused=bool(reused), backbone=str(cfg.model.backbone), **result_row(res), run_dir=run.name)
    if xai:
        if (run / "xai_summary_test.csv").exists() and not force:
            row["xai"] = "reused"
        else:
            try:
                run_xai(cfg, paths, "test")
                row["xai"] = "done"
            except Exception as exc:  # noqa: BLE001  (explanations failing never cost the trained model)
                row["xai"] = f"FAILED: {type(exc).__name__}"
                (out / f"FAILED_{t.name}_xai.txt").write_text(traceback.format_exc())
                log.error("explanations for %s failed: %s", t.name, exc)
    return row, "done" + (" (reused)" if reused else "")


def _suite_track(t: Track, paths: ProjectPaths, overrides: list[str], seeds: list[int] | None, force: bool) -> tuple[dict, str]:
    res = run_suite(t.suite, paths, [*overrides, *t.set], None, seeds or None, force)
    err = res["error"].astype(str) if "error" in res.columns else pd.Series([""] * len(res), index=res.index)
    bad = res["error"].notna() if "error" in res.columns else pd.Series([False] * len(res), index=res.index)
    n_skip, n_fail = int((bad & err.str.startswith("SKIPPED")).sum()), int((bad & ~err.str.startswith("SKIPPED")).sum())
    n_ok = int((~bad).sum())
    col = "auroc" if "auroc" in res.columns else "auroc_mean" if "auroc_mean" in res.columns else None
    best = res[~bad].sort_values(col, ascending=False).iloc[0] if col and n_ok else None
    row = dict(suite_rows=len(res), rows_ok=n_ok, rows_skipped=n_skip, rows_failed=n_fail, results=f"suite_{Path(str(t.suite)).stem}/results.md")
    if best is not None:
        row.update(best_model=str(best["model"]), auroc=float(best[col]))
    if not n_ok:
        return row, ("FAILED: no row of the suite finished" if n_fail else "SKIPPED: " + (err[bad].iloc[0] if bad.any() else "empty suite"))
    return row, "done" if not (n_skip or n_fail) else f"partial: {n_skip} row(s) skipped, {n_fail} failed"

"""Command line: ``mmibc <command>``.  Everything is also callable from Python (see notebooks/colab_quickstart.ipynb).

    download   fetch raw datasets          inspect   show the structure of a raw dataset
    manifest   build source_manifest.csv  prepare    raw -> processed (vindr | us | cmmd)   audit   split + merge-integrity audit (no training)
    train      train + evaluate one config evaluate  re-evaluate a trained run      xai   explanation metrics
    run        prepare -> train -> xai -> figures
    suite      run an ablation set (ladder | pool_ladder | recipe | backbones | pairing); resumable, optional --seeds
    experiments  the whole programme on every dataset that is available (VinDr, CMMD, ultrasound pools, external test); resumable, --list to preview
    literature compare a run with published results (protocol flags)
    figures    paper figures (data plots; --tex for the TikZ diagrams and the corrections table)
    smoke      the whole pipeline on synthetic data, CPU, ~1-2 min: run this first on a new machine
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("mmibc")


def _common(p: argparse.ArgumentParser, config: bool = True) -> None:
    if config:
        p.add_argument("-c", "--config", default="vindr_multiview", help="config name in configs/ or a path (default: vindr_multiview)")
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="dotted overrides, e.g. training.epochs=5 model.backbone=dinov3_vitb16")
    p.add_argument("--root", default=None, help="persistent root (default: $MMIBC_ROOT, Drive on Colab, ./workspace)")
    p.add_argument("--local", default=None, help="fast local scratch (default: /content/mmibc_local on Colab)")


def _paths(a, cfg=None):
    from mmibc.utils.paths import paths_from_cfg, resolve_paths

    if cfg is not None:
        if a.root:
            cfg.paths.root = a.root
        if a.local:
            cfg.paths.local = a.local
        return paths_from_cfg(cfg)
    import os

    paths = resolve_paths(a.root, a.local)
    os.environ.setdefault("HF_HOME", str(paths.hf_cache))
    return paths


def _cfg(a):
    from mmibc.utils.config import load_config

    return load_config(a.config, a.set)


# ----------------------------------------------------------------------------------------------- commands
def cmd_download(a):
    from mmibc.data.download import download_dataset, is_downloaded, setup_kaggle_credentials
    from mmibc.data.registry import REGISTRY

    paths = _paths(a, _cfg(a))  # same config as prepare/train, so every stage agrees on where raw data lives
    keys = list(REGISTRY) if a.all else a.keys
    if not keys:
        print("datasets:", ", ".join(f"{k} ({v.kind})" for k, v in REGISTRY.items()))
        return 0
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        print(f"unknown dataset(s): {unknown}. Known: {list(REGISTRY)}", file=sys.stderr)
        return 2
    where = "Google Drive" if paths.raw_on_drive else "local disk"
    print(f"saving raw downloads to {paths.raw} ({where})")
    # Check the Kaggle credentials once, up front: six identical failures would bury the one fix that is needed.
    wants_kaggle = [k for k in keys if REGISTRY[k].kind == "kaggle" and (a.force or not is_downloaded(k, paths))]
    use_kaggle = True
    if wants_kaggle:
        try:
            setup_kaggle_credentials()
        except RuntimeError as exc:
            use_kaggle = False
            print(f"\n!! Kaggle credentials are not available, so the Kaggle datasets will be skipped: {', '.join(wants_kaggle)}\n{exc}\n", file=sys.stderr)
    results: dict[str, str] = {}
    for i, k in enumerate(keys, 1):
        print(f"[{i}/{len(keys)}] {k}")
        try:
            print(f"{k}: {download_dataset(k, paths, a.force, use_kaggle)}")
            results[k] = "ok"
        except Exception as exc:  # noqa: BLE001  one failing dataset must not hide the others
            print(f"{k}: FAILED - {exc}", file=sys.stderr)
            results[k] = "FAILED"
    if len(keys) > 1:
        print("\nsummary:")
        for k, r in results.items():
            print(f"  {r:<7} {k}")
        failed = [k for k, r in results.items() if r != "ok"]
        if failed:
            print(f"\n{len(failed)} of {len(keys)} failed. Fix the cause shown above and re-run `mmibc download {' '.join(failed)}`; "
                  "datasets that already succeeded are skipped.", file=sys.stderr)
    return 0 if all(r == "ok" for r in results.values()) else 1


def cmd_inspect(a):
    from mmibc.data.download import inspect_dataset

    print(inspect_dataset(_paths(a, _cfg(a)).raw / a.key))
    return 0


def cmd_manifest(a):
    from mmibc.data.manifests import build_manifest

    out = build_manifest(a.key, _paths(a, _cfg(a)).raw / a.key, a.force)
    print(f"{out}  ({sum(1 for _ in open(out)) - 1} rows)")
    return 0


def cmd_prepare(a):
    from mmibc.pipeline import prepare

    cfg = _cfg(a)
    paths = _paths(a, cfg)
    which = {"all": ("vindr", "ultrasound"), "vindr": ("vindr",), "us": ("ultrasound",), "ultrasound": ("ultrasound",), "cmmd": ("cmmd",)}[a.what]
    if a.what == "all" and (paths.raw / "cmmd").exists():  # CMMD is optional: `all` includes it only when it has been downloaded
        which = (*which, "cmmd")
    prepare(cfg, paths, which, a.force)
    return 0


def cmd_audit(a):
    from mmibc.pipeline import build_pair_sets, build_tables

    cfg = _cfg(a)
    cfg.audit.stop_on_error = False
    paths = _paths(a, cfg)
    tables = build_tables(cfg, paths, stage=False, need_us=a.us or None)
    sets = build_pair_sets(cfg, tables, paths)
    bad = 0
    for name, rep in {**tables.audits, **sets.audits}.items():
        print(f"\n===== {name}: {'PASS' if rep.ok else 'FAIL'} =====\n{rep.to_markdown()}")
        bad += len(rep.errors)
    return 1 if bad else 0


def cmd_train(a):
    from mmibc.pipeline import run_experiment

    cfg = _cfg(a)
    res = run_experiment(cfg, _paths(a, cfg), evaluate=not a.no_eval, force=a.force)
    print(json.dumps({k: v for k, v in res.items() if k in ("threshold", "train", "backbone", "interrupted")}, indent=2, default=str))
    if "splits" in res:
        s = res["splits"]["test"]
        print(f"test AUROC {s['auroc']:.4f}  CI {s.get('auroc_delong_ci')}")
    return 0


def cmd_evaluate(a):
    from mmibc.evaluation.evaluate import evaluate_model
    from mmibc.pipeline import build_pair_sets, build_tables, load_trained, run_dir_for
    from mmibc.training.trainer import get_device

    cfg = _cfg(a)
    paths = _paths(a, cfg)
    run = run_dir_for(cfg, paths)
    if not (run / "best.pt").exists():
        print(f"no trained model at {run}: run `mmibc train` first", file=sys.stderr)
        return 1
    sets = build_pair_sets(cfg, build_tables(cfg, paths), paths)
    dev = get_device()
    res = evaluate_model(load_trained(cfg, run, dev, eager=False), sets.eval_sets(), cfg, run, dev)
    print(f"test AUROC {res['splits']['test']['auroc']:.4f}")
    return 0


def cmd_xai(a):
    from mmibc.pipeline import run_xai

    cfg = _cfg(a)
    print(run_xai(cfg, _paths(a, cfg), a.split).to_string(index=False))
    return 0


def cmd_run(a):
    from mmibc.pipeline import prepare, run_experiment, run_xai

    cfg = _cfg(a)
    paths = _paths(a, cfg)
    from mmibc.pipeline import mammo_source_of

    wanted = [mammo_source_of(cfg), *(["ultrasound"] if cfg.task.kind == "synthetic_pairs" else [])]
    need = [n for n in wanted if not (paths.processed / n / "manifest.csv").exists()]
    if need:
        log.info("preparing %s", need)
        prepare(cfg, paths, tuple(need))
    res = run_experiment(cfg, paths)
    if res.get("interrupted"):
        return 2
    if not a.no_xai:
        run_xai(cfg, paths, "test")
    if not a.no_figures:
        from mmibc.figures.make import make_figures
        from mmibc.pipeline import run_dir_for

        run = run_dir_for(cfg, paths)
        make_figures({run.name: run}, paths.root / "figures", a.template, splits_dir=paths.splits)
    return 0


def cmd_suite(a):
    from mmibc.suite import run_suite

    cfg_over = list(a.set)
    from mmibc.utils.config import load_config

    paths = _paths(a, load_config(None, cfg_over))
    res = run_suite(a.name, paths, cfg_over, a.only or None, a.seeds or None, a.force)
    print(res.to_string(index=False))
    return 0


def cmd_experiments(a):
    from mmibc.experiments import run_program
    from mmibc.utils.config import load_config

    paths = _paths(a, load_config(None, list(a.set)))
    summary, rc = run_program(a.program, paths, list(a.set), a.only or None, a.skip or None, a.suites, a.xai, a.force, a.seeds or None, a.list)
    cols = ["track", "plan", "data"] if a.list else ["track", "state", "auroc", "auroc_lo", "auroc_hi", "ext_auroc", "auroc_D1", "auroc_D2", "minutes"]
    shown = summary[[c for c in cols if c in summary.columns]] if len(summary) else summary
    print(shown.to_string(index=False, float_format=lambda x: f"{x:.3f}", max_colwidth=90) if len(shown) else "(no tracks selected)")
    if not a.list:
        print(f"\nresults: {paths.runs / ('experiments_' + Path(a.program).stem)}  (summary.md, status.csv, plan.csv)")
    return rc


def cmd_literature(a):
    import pandas as pd
    from omegaconf import OmegaConf

    from mmibc.benchmarks.compare import comparison_table, to_latex

    run = Path(a.run_dir)
    cfg = OmegaConf.load(run / "config.yaml")
    m = json.loads((run / "metrics.json").read_text())["splits"]["test"]
    lo, hi = m.get("auroc_delong_ci", m["ci"]["auroc"])
    df = comparison_table([dict(model=a.label or run.name, auroc=m["auroc"], lo=lo, hi=hi)], cfg, a.dataset)
    out = Path(a.out) if a.out else run
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "literature_comparison.csv", index=False)
    (out / "literature_comparison.tex").write_text(to_latex(df, "Comparison with published results. Only rows marked as protocol matched are directly comparable.", "tab:literature"))
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(df[["source", "model", "auroc", "comparable_with_published"]].to_string(index=False))
    return 0


def cmd_figures(a):
    paths = _paths(a)
    out = Path(a.out) if a.out else paths.root / "figures"
    made = []
    if a.tex:
        from mmibc.figures import tex

        if not tex.have_tex():
            print("TeX not found. Colab: apt-get install -y texlive-latex-extra texlive-fonts-recommended poppler-utils\n"
                  "(the pre-built diagrams are in paper/figures/pdf)", file=sys.stderr)
            return 1
        tex.write_corrections_table()
        made += tex.build_all(Path(a.tex_out) if a.tex_out else None)
    else:
        from mmibc.figures.make import make_figures

        runs = dict(x.split("=", 1) for x in a.runs) if a.runs else {d.name: d for d in sorted(paths.runs.glob("*")) if (d / "metrics.json").exists()}
        if not runs:
            print(f"no finished runs under {paths.runs}", file=sys.stderr)
            return 1
        made = make_figures({k: Path(v) for k, v in runs.items()}, out, a.template, a.split, a.reference, paths.splits, a.xai_run)
    print("\n".join(str(m) for m in made))
    return 0


def cmd_smoke(a):
    import tempfile

    from mmibc.data.synthetic import make_fake_busi, make_fake_extra_us, make_fake_vindr
    from mmibc.figures.make import make_figures
    from mmibc.pipeline import prepare, run_dir_for, run_experiment, run_xai
    from mmibc.utils.config import load_config
    from mmibc.utils.paths import ProjectPaths

    work = Path(a.workdir or tempfile.mkdtemp(prefix="mmibc_smoke_"))
    paths = ProjectPaths(work / "drive", work / "local").ensure()
    make_fake_vindr(paths.raw, 40), make_fake_busi(paths.raw), make_fake_extra_us(paths.raw)
    base = ["audit.min_per_class_split=1", "training.num_workers=0", "data.us_sources=[busi,extra_us]", "data.us_external_source=extra_us", "training.epochs=1", "eval.controls=[none,drop_a,shuffle_b]", "xai.n_samples=6", "xai.steps=4"]
    prepare(load_config("smoke.yaml", base), paths)
    runs = {}
    for kind, extra in (("multiview", []), ("synthetic_pairs", ["task.pair_mode=label_matched"])):
        cfg = load_config("smoke.yaml", [*base, f"task.kind={kind}", f"run_name=smoke_{kind}", *extra])
        res = run_experiment(cfg, paths)
        assert res["splits"]["test"]["auroc"] == res["splits"]["test"]["auroc"], "NaN test AUROC"
        run_xai(cfg, paths, "test")
        runs[kind] = run_dir_for(cfg, paths)
        print(f"[ok] {kind}: test AUROC {res['splits']['test']['auroc']:.3f}")
    if a.cmmd:  # CMMD on fake DICOMs written in the real layout (needs pydicom): biopsy-labelled mammograms, multiview and paired with ultrasound
        from mmibc.data.synthetic import make_fake_cmmd

        make_fake_cmmd(paths.raw)
        prepare(load_config("smoke.yaml", base), paths, ("cmmd",))
        cm = [*base, "data.mammo_source=cmmd", "data.us_sources=[busi]", "data.us_external_source=null"]
        for kind, extra in (("multiview", []), ("synthetic_pairs", ["task.pair_mode=label_matched"])):
            cfg = load_config("smoke.yaml", [*cm, f"task.kind={kind}", f"run_name=smoke_cmmd_{kind}", *extra])
            res = run_experiment(cfg, paths)
            assert res["splits"]["test"]["auroc"] == res["splits"]["test"]["auroc"], "NaN test AUROC"
            assert "by_stratum" in res["splits"]["test"], "per-subset metrics missing"
            run_xai(cfg, paths, "test")
            runs[f"cmmd_{kind}"] = run_dir_for(cfg, paths)
            print(f"[ok] cmmd {kind}: test AUROC {res['splits']['test']['auroc']:.3f}")
    made = make_figures(runs, work / "figures", splits_dir=paths.splits)
    print(f"[ok] {len(made)} figures -> {work / 'figures'}")
    print(f"SMOKE OK  (workdir {work})")
    return 0


# ----------------------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mmibc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="fetch raw datasets (no argument: list them)")
    _common(p)
    p.add_argument("keys", nargs="*")
    p.add_argument("--all", action="store_true", help="download every registered dataset, one after the other, and print a summary")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("inspect", help="structure report of a raw dataset")
    _common(p)
    p.add_argument("key")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("manifest", help="build raw/<key>/source_manifest.csv for bus_bra or breast_lesions_usg (prepare does this by itself when it is missing)")
    _common(p)
    p.add_argument("key", choices=["bus_bra", "breast_lesions_usg"])
    p.add_argument("--force", action="store_true", help="rebuild an existing manifest")
    p.set_defaults(fn=cmd_manifest)

    p = sub.add_parser("prepare", help="raw -> processed (crop, downscale, manifests); resumable")
    _common(p)
    p.add_argument("what", nargs="?", default="all", choices=["all", "vindr", "us", "ultrasound", "cmmd"])
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("audit", help="splits + merge-integrity audit; prints every finding, trains nothing")
    _common(p)
    p.add_argument("--us", action="store_true", help="include the ultrasound pool even for a multiview task")
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("train", help="train (resumable) and evaluate one configuration")
    _common(p)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--force", action="store_true", help="re-evaluate even if this exact configuration already finished (training always resumes)")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("evaluate", help="re-evaluate a trained run (frozen val threshold, CIs, controls)")
    _common(p)
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("xai", help="explanation maps scored against lesion annotations")
    _common(p)
    p.add_argument("--split", default="test")
    p.set_defaults(fn=cmd_xai)

    p = sub.add_parser("run", help="prepare -> train -> evaluate -> xai -> figures")
    _common(p)
    p.add_argument("--no-xai", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--template", default="lncs")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("suite", help="run an ablation set on identical frozen test pairs")
    _common(p, False)
    p.add_argument("name", help="ladder | pool_ladder | recipe | backbones | pairing | path.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--seeds", nargs="*", type=int, default=[], help="repeat every run for these training seeds on one fixed partition; adds mean/sd and the seed ensemble")
    p.add_argument("--force", action="store_true", help="re-evaluate finished runs (training always resumes)")
    p.set_defaults(fn=cmd_suite)

    p = sub.add_parser("experiments", help="run every experiment on every dataset that is available (prepares what exists, skips what is missing); resumable")
    _common(p, False)
    p.add_argument("program", nargs="?", default="all", help="configs/programs/<name>.yaml or a path (default: all)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="overrides applied to every track, e.g. training.epochs=1 for a quick dry run")
    p.add_argument("--only", nargs="*", default=[], help="track or group names (vindr | robust | cmmd | external | suites)")
    p.add_argument("--skip", nargs="*", default=[], help="track or group names to leave out")
    p.add_argument("--suites", action="store_true", help="also run the suite tracks (fusion ladders, backbone comparison)")
    p.add_argument("--seeds", nargs="*", type=int, default=[], help="for suite tracks: repeat every row for these seeds")
    p.add_argument("--xai", action="store_true", help="also compute explanation metrics for every run track (slower)")
    p.add_argument("--force", action="store_true", help="re-evaluate finished runs (training always resumes)")
    p.add_argument("--list", action="store_true", help="show the plan (datasets found, runs finished, tracks blocked) and exit; prepares and trains nothing")
    p.set_defaults(fn=cmd_experiments)

    p = sub.add_parser("literature", help="comparison table against published results, with protocol-comparability flags")
    _common(p, False)
    p.add_argument("run_dir")
    p.add_argument("--dataset", default="VinDr")
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_literature)

    p = sub.add_parser("figures", help="paper figures: data plots from finished runs, or --tex for the TikZ diagrams")
    _common(p, False)
    p.add_argument("--tex", action="store_true", help="build the TikZ architecture/algorithm diagrams and the corrections table")
    p.add_argument("--tex-out", default=None)
    p.add_argument("--runs", nargs="*", default=[], metavar="NAME=RUN_DIR")
    p.add_argument("--out", default=None)
    p.add_argument("--template", default="lncs", choices=["lncs", "ieee", "elsevier", "generic"])
    p.add_argument("--split", default="test")
    p.add_argument("--reference", default=None)
    p.add_argument("--xai-run", default=None)
    p.set_defaults(fn=cmd_figures)

    p = sub.add_parser("smoke", help="end-to-end check on synthetic data (CPU)")
    p.add_argument("--workdir", default=None)
    p.add_argument("--cmmd", action="store_true", help="also run the CMMD path on fake DICOMs (needs pydicom): proves DICOM reading, biopsy-labelled splits and per-subset metrics work here")
    p.set_defaults(fn=cmd_smoke)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())

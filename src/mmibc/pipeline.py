"""End-to-end orchestration: raw -> processed -> audited splits -> frozen pairs -> datasets -> train -> evaluate."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mmibc.data.datasets import PairDataset, build_transforms, stream_size
from mmibc.data.merge_audit import AuditConfig, AuditReport, audit_pairs, audit_ultrasound, audit_vindr
from mmibc.data.pairing import SyntheticResampler, load_real_pairs, make_native_multiview_pairs, make_synthetic_pairs
from mmibc.data.splits import add_size_quantiles, assign_us_splits, assign_vindr_splits
from mmibc.data.manifests import BUILDERS, build_manifest
from mmibc.data.ultrasound import load_source_manifest, prepare_ultrasound, scan_busi, scan_class_folders
from mmibc.data.vindr import prepare_vindr
from mmibc.utils.paths import ProjectPaths, stage_to_local

log = logging.getLogger(__name__)
VINDR, US = "vindr", "ultrasound"


# ----------------------------------------------------------------------------- prepare
def scan_us_sources(cfg, paths: ProjectPaths) -> pd.DataFrame:
    frames = []
    for key in cfg.data.us_sources:
        raw = paths.raw / key
        if not raw.exists():
            raise FileNotFoundError(f"ultrasound source '{key}' not found at {raw}. Run `mmibc download {key}` (or place it there by hand).")
        csv = raw / "source_manifest.csv"
        if key in BUILDERS and not csv.exists():  # bus_bra, breast_lesions_usg: labels live in a spreadsheet; build the manifest from it
            build_manifest(key, raw)
        if key == "busi":
            df = scan_busi(raw, "busi")
        elif csv.exists():
            df = load_source_manifest(csv, raw, key)
        else:
            df = scan_class_folders(raw, key)
        if df.empty:
            raise RuntimeError(f"source '{key}': no images found under {raw}. Run `mmibc inspect {key}`, then write {csv} (columns: uid, raw_path, class_name[, mask_paths, patient_id]).")
        log.info("source %-20s %5d images  %s", key, len(df), df["class_name"].value_counts().to_dict())
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def prepare(cfg, paths: ProjectPaths, which: tuple[str, ...] = (VINDR, US), force: bool = False) -> None:
    d = cfg.data
    if VINDR in which:
        prepare_vindr(paths.raw / "vindr_images", paths.raw / "vindr_annotations", paths.processed / VINDR, int(d.store_long_side), bool(d.vindr.crop_breast), d.vindr.label_scheme, int(d.num_prepare_workers), force)
    if US in which:
        prepare_ultrasound(scan_us_sources(cfg, paths), paths.processed / US, d.busi.label_scheme, int(d.store_long_side), bool(d.busi.inpaint_overlay), int(d.num_prepare_workers), force)


# ----------------------------------------------------------------------------- tables
@dataclass
class Tables:
    vindr: pd.DataFrame
    us: pd.DataFrame | None
    findings: pd.DataFrame
    vindr_root: Path
    us_root: Path | None
    audits: dict[str, AuditReport] = field(default_factory=dict)


def build_tables(cfg, paths: ProjectPaths, stage: bool = True, need_us: bool | None = None) -> Tables:
    """Load processed manifests, assign leakage-safe splits, audit the merge, persist everything under ``splits/``."""
    need_us = (cfg.task.kind == "synthetic_pairs") if need_us is None else need_us
    acfg = AuditConfig.from_cfg(cfg.get("audit"))
    out = paths.splits
    out.mkdir(parents=True, exist_ok=True)

    vsrc = paths.processed / VINDR
    vroot = stage_to_local(vsrc, paths.local_processed / VINDR) if stage else vsrc
    split_seed = int(cfg.seed if cfg.data.get("split_seed") is None else cfg.data.split_seed)
    vind = assign_vindr_splits(pd.read_csv(vsrc / "manifest.csv"), float(cfg.data.vindr.val_fraction), split_seed)
    vind = add_size_quantiles(vind)
    findings = pd.read_csv(vsrc / "findings.csv")
    audits = {"vindr": audit_vindr(vind, vroot, findings, acfg)}
    vind.to_csv(out / "vindr_split.csv", index=False)

    us = us_root = None
    if need_us:
        usrc = paths.processed / US
        us_root = stage_to_local(usrc, paths.local_processed / US) if stage else usrc
        ext = cfg.data.us_external_source
        us = assign_us_splits(
            pd.read_csv(usrc / "manifest.csv"), float(cfg.data.busi.val_fraction), float(cfg.data.busi.test_fraction), split_seed,
            bool(cfg.data.busi.dedupe), int(cfg.data.busi.dup_hamming), ext,
        )
        us = add_size_quantiles(us)
        audits["ultrasound"] = audit_ultrasound(us, us_root, acfg, ext, cfg.data.busi.label_scheme)
        us.to_csv(out / "ultrasound_split.csv", index=False)

    for name, rep in audits.items():
        rep.save(out)
        log.info("audit %-11s %s (%d errors, %d warnings) -> %s", name, "PASS" if rep.ok else "FAIL", len(rep.errors), len(rep.warnings), out / f"audit_{name}.md")
        if cfg.audit.stop_on_error:
            rep.raise_if_errors()
    return Tables(vind, us, findings, vroot, us_root, audits)


# ----------------------------------------------------------------------------- pairs and datasets
@dataclass
class PairSets:
    train: PairDataset
    val: PairDataset
    test: PairDataset
    extra: dict[str, PairDataset]  # ext_test (external ultrasound), real_test (genuine paired data), ...
    pair_tables: dict[str, pd.DataFrame]
    audits: dict[str, AuditReport]

    def eval_sets(self) -> dict[str, PairDataset]:
        return {"val": self.val, "test": self.test, **self.extra}


def build_pair_sets(cfg, tables: Tables, paths: ProjectPaths) -> PairSets:
    kind, mode = cfg.task.kind, cfg.task.pair_mode
    sz_a, sz_b = stream_size(cfg, "a"), stream_size(cfg, "b")  # (H, W) per stream; a mammogram can be read at higher resolution than an ultrasound
    tr_a, tr_b = build_transforms(cfg.input, True, sz_a), build_transforms(cfg.input, True, sz_b)
    ev_a, ev_b = build_transforms(cfg.input, False, sz_a), build_transforms(cfg.input, False, sz_b)
    seed_eval = int(cfg.task.eval_pair_seed)
    extra: dict[str, PairDataset] = {}
    audits: dict[str, AuditReport] = {}
    if kind == "multiview":
        pairs = make_native_multiview_pairs(tables.vindr)
        tabs = {s: pairs[pairs["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
        ra = rb = tables.vindr_root
        a_df = b_df = tables.vindr
        train = PairDataset(tabs["train"], ra, rb, tr_a, tr_b)
        matched, native = True, True
    elif kind == "synthetic_pairs":
        if tables.us is None:
            raise ValueError("synthetic_pairs needs the ultrasound pool: run `mmibc prepare us` first")
        a_df, b_df, ra, rb = tables.vindr, tables.us, tables.vindr_root, tables.us_root
        tabs = {s: make_synthetic_pairs(a_df, b_df, s, mode, seed_eval, replace=False) for s in ("val", "test")}
        tabs["train"] = make_synthetic_pairs(a_df, b_df, "train", mode, int(cfg.seed), replace=True)
        resampler = SyntheticResampler(a_df, b_df, "train", mode, int(cfg.seed) * 7919, True) if cfg.task.resample_each_epoch else None
        train = PairDataset(tabs["train"], ra, rb, tr_a, tr_b, resampler)
        matched, native = mode in ("label_matched", "size_conditioned"), False
        if (b_df["split"] == "ext_test").any():  # external ultrasound source x the held-out mammograms
            a_ext = a_df[a_df["split"] == "test"].assign(split="ext_test")
            tabs["ext_test"] = make_synthetic_pairs(a_ext, b_df, "ext_test", mode, seed_eval, replace=False)
            extra["ext_test"] = PairDataset(tabs["ext_test"], ra, rb, ev_a, ev_b)
            audits["pairs_ext_test"] = audit_pairs(tabs["ext_test"], a_ext, b_df, True, matched)
    else:
        raise ValueError(f"unknown task.kind '{kind}'")

    audits["pairs_train"] = audit_pairs(tabs["train"], a_df, b_df, False, matched, native)
    for s in ("val", "test"):
        audits[f"pairs_{s}"] = audit_pairs(tabs[s], a_df, b_df, not native, matched, native)
    tag = f"{kind}_{mode}" if kind == "synthetic_pairs" else kind
    out = paths.splits / tag
    out.mkdir(parents=True, exist_ok=True)
    for s, df in tabs.items():
        df.to_csv(out / f"pairs_{s}.csv", index=False)
    for name, rep in audits.items():
        rep.save(out)
        if cfg.audit.stop_on_error:
            rep.raise_if_errors()

    real = cfg.task.get("real_pairs_csv")
    if real:  # genuine same-patient mammogram+ultrasound pairs (Tier 1b): evaluation only
        rp = load_real_pairs(real)
        extra["real_test"] = PairDataset(rp, Path(cfg.task.real_root_a), Path(cfg.task.real_root_b), ev_a, ev_b)
        tabs["real_test"] = rp
    return PairSets(train, PairDataset(tabs["val"], ra, rb, ev_a, ev_b), PairDataset(tabs["test"], ra, rb, ev_a, ev_b), extra, tabs, audits)


# ----------------------------------------------------------------------------- experiment
def run_dir_for(cfg, paths: ProjectPaths) -> Path:
    tag = cfg.run_name if cfg.task.kind == "multiview" else f"{cfg.run_name}_{cfg.task.pair_mode}"
    return paths.runs / f"{tag}_{cfg.model.backbone}"


def finished_metrics(cfg, run: Path) -> dict | None:
    """The saved metrics of ``run`` if it was trained AND evaluated with exactly this configuration, else None."""
    from mmibc.evaluation.evaluate import eval_fingerprint
    from mmibc.training.trainer import config_fingerprint

    f = Path(run) / "metrics.json"
    if not f.exists():
        return None
    try:
        prev = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    if prev.get("train", {}).get("config_fingerprint") == config_fingerprint(cfg) and prev.get("eval_fingerprint") == eval_fingerprint(cfg) and "splits" in prev:
        return prev
    return None


def run_experiment(cfg, paths: ProjectPaths, evaluate: bool = True, stage: bool = True, force: bool = False) -> dict:
    """Train (resumable) and evaluate one configuration. Returns the metrics dict.

    A run that already finished with the same training AND evaluation configuration is not repeated (``force=True`` redoes the evaluation;
    training itself always resumes from ``last.pt``). A finished run whose evaluation settings changed is re-evaluated, not retrained.
    """
    from mmibc.evaluation.evaluate import evaluate_model
    from mmibc.models.factory import build_model
    from mmibc.training.trainer import Trainer, get_device
    from mmibc.utils.config import save_config
    from mmibc.utils.seed import seed_everything

    seed_everything(int(cfg.seed))
    run = run_dir_for(cfg, paths)
    if evaluate and not force and cfg.training.resume:
        done = finished_metrics(cfg, run)
        if done is not None:
            log.info("run %s already finished with this configuration: skipping (use --force to re-evaluate)", run.name)
            return done
    run.mkdir(parents=True, exist_ok=True)
    old = run / "metrics.json"
    if old.exists() and "calibration" not in json.loads(old.read_text() or "{}") and not (run / "metrics.uncalibrated.json").exists():
        (run / "metrics.uncalibrated.json").write_text(old.read_text())  # keep the numbers reported before recalibration existed
    save_config(cfg, run / "config.yaml")
    tables = build_tables(cfg, paths, stage)
    sets = build_pair_sets(cfg, tables, paths)
    model = build_model(cfg.model)
    for stream in ("a", "b"):
        bb = getattr(model, f"backbone_{stream}", None)
        if bb is not None:
            bb.check_input_size(stream_size(cfg, stream))
    if cfg.model.fusion == "early" and stream_size(cfg, "a") != stream_size(cfg, "b"):
        raise ValueError(f"fusion='early' stacks both images into one tensor: input sizes must match, got {stream_size(cfg, 'a')} and {stream_size(cfg, 'b')}")
    trainer = Trainer(model, cfg, run, sets.train, sets.val, get_device(), int(cfg.model.n_classes))
    summary = trainer.fit()
    if summary.get("interrupted") or not evaluate:
        return summary
    metrics = evaluate_model(model, sets.eval_sets(), cfg, run, trainer.device)
    metrics["train"] = summary
    metrics["backbone"] = cfg.model.backbone
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log.info("test AUROC %.4f  [%s]", metrics["splits"]["test"]["auroc"], run)
    return metrics


# ----------------------------------------------------------------------------- explanation
def load_trained(cfg, run: Path, device, eager: bool = True):
    """Rebuild the model of a finished run and load its best weights. ``eager`` enables attention rollout for ViT backbones."""
    import torch

    from mmibc.models.factory import build_model
    from mmibc.training.checkpoint import load_model_state

    model = build_model(cfg.model, eager_attention=eager)
    load_model_state(model, torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)["model"])
    return model.to(device).eval()


def run_xai(cfg, paths: ProjectPaths, split: str = "test", stage: bool = True):
    from mmibc.training.trainer import get_device
    from mmibc.xai.ground_truth import GroundTruth
    from mmibc.xai.report import explain_split

    run = run_dir_for(cfg, paths)
    tables = build_tables(cfg, paths, stage)
    sets = build_pair_sets(cfg, tables, paths)
    ds = sets.eval_sets()[split]
    device = get_device()
    model = load_trained(cfg, run, device, eager=True)
    gt = GroundTruth(stream_size(cfg, "a"), tables.vindr, tables.findings, tables.vindr_root, tables.us, tables.us_root, us_size=stream_size(cfg, "b"))
    return explain_split(model, ds, gt, cfg, run, device, split)

"""Integrity audit for merged datasets and pair tables.

Merging sources is where silent errors enter: colliding ids, a class name mapped to the wrong label, the same scan in two
sources, an "external" source that leaks into training, a split with too few positives to measure anything, files that
are missing / blank / the wrong size, and - the subtle one - a SOURCE that predicts the label (a model then learns the
scanner instead of the lesion). Every check yields a Finding; ``error`` findings stop the pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from mmibc.data.audit import leakage_report, near_duplicate_groups
from mmibc.data.pairing import assert_pairs_clean

log = logging.getLogger(__name__)
SPLITS = {"train", "val", "test", "ext_test"}


@dataclass
class Finding:
    level: str  # error | warn | info
    code: str
    message: str


@dataclass
class AuditConfig:
    min_per_class_split: int = 20     # warn below this many images of a class in a split
    max_prevalence_gap: float = 0.25  # warn when sources differ this much in positive rate
    max_meta_auc: float = 0.70        # warn when metadata/size/intensity alone predicts the label better than this
    dup_hamming: int = 6
    check_files: str = "all"          # all | exists | sample:<N>
    box_tolerance_px: float = 3.0
    max_boxes_outside: float = 0.05

    @staticmethod
    def from_cfg(section) -> "AuditConfig":
        return AuditConfig(**{k: v for k, v in dict(section or {}).items() if k in AuditConfig.__dataclass_fields__})


@dataclass
class AuditReport:
    name: str
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, level: str, code: str, message: str) -> None:
        self.findings.append(Finding(level, code, message))
        getattr(log, {"error": "error", "warn": "warning", "info": "info"}[level])("[%s] %s: %s", self.name, code, message)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warn"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_markdown(self) -> str:
        lines = [f"# Audit: {self.name}", "", f"**{'PASS' if self.ok else 'FAIL'}** - {len(self.errors)} errors, {len(self.warnings)} warnings", ""]
        for lvl, title in (("error", "Errors"), ("warn", "Warnings"), ("info", "Info")):
            rows = [f for f in self.findings if f.level == lvl]
            if rows:
                lines += [f"## {title}", ""] + [f"- `{f.code}` {f.message}" for f in rows] + [""]
        if self.stats:
            lines += ["## Statistics", "", "```json", json.dumps(self.stats, indent=2, default=str), "```"]
        return "\n".join(lines)

    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"audit_{self.name}.md").write_text(self.to_markdown())
        (out_dir / f"audit_{self.name}.json").write_text(json.dumps({"ok": self.ok, "findings": [asdict(f) for f in self.findings], "stats": self.stats}, indent=2, default=str))
        return out_dir / f"audit_{self.name}.md"

    def raise_if_errors(self) -> None:
        if self.errors:
            raise RuntimeError(f"Audit '{self.name}' failed:\n" + "\n".join(f"  [{f.code}] {f.message}" for f in self.errors))


# ----------------------------------------------------------------------------- shared checks
def _require(rep: AuditReport, df: pd.DataFrame, cols: list[str]) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        rep.add("error", "schema", f"missing columns {missing}")
    return not missing


def _class_counts(rep: AuditReport, df: pd.DataFrame, cfg: AuditConfig, label_col: str = "label") -> None:
    table = df.groupby(["split", label_col]).size().unstack(fill_value=0)
    rep.stats["per_split_class_counts"] = table.to_dict(orient="index")
    for split, row in table.iterrows():
        if split == "ext_test" and len(row) == 0:
            continue
        for lab in df[label_col].dropna().unique():
            n = int(row.get(lab, 0))
            if n == 0:
                rep.add("error", "empty_class", f"split '{split}' has no images of label {lab}: metrics are undefined")
            elif n < cfg.min_per_class_split:
                rep.add("warn", "small_class", f"split '{split}' has only {n} images of label {lab} (< {cfg.min_per_class_split}): confidence intervals will be very wide")


def _check_files(rep: AuditReport, df: pd.DataFrame, root: Path, cfg: AuditConfig, with_masks: bool = False) -> None:
    mode = cfg.check_files
    rows = df
    if mode.startswith("sample:"):
        rows = df.sample(n=min(len(df), int(mode.split(":")[1])), random_state=0)
    bad_missing, bad_decode, bad_size, bad_blank, bad_mask = [], [], [], [], []
    for r in rows.itertuples():
        p = Path(root) / r.proc_path
        if not p.exists() or p.stat().st_size == 0:
            bad_missing.append(r.uid)
            continue
        if mode == "exists":
            continue
        try:
            with Image.open(p) as im:
                im.load()
                size = im.size
                arr = np.asarray(im.convert("L"))
        except Exception:  # noqa: BLE001 - any decode failure is a finding
            bad_decode.append(r.uid)
            continue
        if hasattr(r, "proc_w") and not pd.isna(r.proc_w) and (size[0], size[1]) != (int(r.proc_w), int(r.proc_h)):
            bad_size.append(r.uid)
        if arr.std() < 1.0:
            bad_blank.append(r.uid)
        if with_masks and isinstance(getattr(r, "mask_path", ""), str) and r.mask_path:
            mp = Path(root) / r.mask_path
            try:
                with Image.open(mp) as mi:
                    if mi.size != size:
                        bad_mask.append(r.uid)
            except Exception:  # noqa: BLE001
                bad_mask.append(r.uid)
    for lst, code, msg in (
        (bad_missing, "file_missing", "image file missing or empty"), (bad_decode, "file_decode", "image cannot be decoded"),
        (bad_size, "size_mismatch", "stored size differs from the manifest (processed twice? stale cache?)"),
        (bad_blank, "blank_image", "image is (nearly) constant"), (bad_mask, "mask_mismatch", "mask missing or its size differs from the image"),
    ):
        if lst:
            rep.add("error", code, f"{len(lst)} files: {msg}; e.g. {lst[:3]}")
    rep.stats["files_checked"] = len(rows)


def _cross_split_duplicates(rep: AuditReport, df: pd.DataFrame, cfg: AuditConfig) -> None:
    if "dhash" not in df.columns or df["dhash"].isna().any():
        rep.add("warn", "no_hash", "perceptual hashes unavailable: cross-split duplicate check skipped")
        return
    gid = near_duplicate_groups(df["dhash"].tolist(), cfg.dup_hamming)
    n_splits = pd.Series(df["split"].to_numpy()).groupby(gid).nunique()
    leaky = n_splits[n_splits > 1]
    rep.stats["near_duplicate_groups_spanning_splits"] = int(len(leaky))
    if len(leaky):
        rep.add("error", "duplicate_leak", f"{len(leaky)} near-duplicate image groups (Hamming <= {cfg.dup_hamming}) span several splits: test images were seen in training")
    labs = pd.Series(df["label"].to_numpy()).groupby(gid).nunique()
    if (labs > 1).any():
        rep.add("error", "duplicate_label_conflict", f"{int((labs > 1).sum())} near-duplicate groups carry conflicting labels")


def source_confounding(rep: AuditReport, df: pd.DataFrame, cfg: AuditConfig, root: Path | None) -> None:
    """Does the source (or trivial image metadata) predict the label? If yes, merged-data accuracy is partly a scanner detector."""
    from scipy.stats import chi2_contingency
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    internal = df[df["split"] != "ext_test"]
    if internal["label"].nunique() != 2:
        return
    prev = internal.groupby("source")["label"].agg(["mean", "size"])
    rep.stats["prevalence_by_source"] = {k: dict(positive_rate=float(v["mean"]), n=int(v["size"])) for k, v in prev.iterrows()}
    if len(prev) > 1:
        gap = float(prev["mean"].max() - prev["mean"].min())
        p = float(chi2_contingency(pd.crosstab(internal["source"], internal["label"]))[1])
        rep.stats["source_label_chi2_p"], rep.stats["prevalence_gap"] = p, gap
        if gap > cfg.max_prevalence_gap and p < 0.01:
            rep.add("warn", "source_prevalence", f"positive rate differs by {gap:.2f} between sources (chi2 p={p:.1e}): the source partly determines the label. Report per-source metrics and consider source-balanced sampling")
    feats = pd.DataFrame({k: internal[c] for k, c in (("w", "proc_w"), ("h", "proc_h"), ("overlay", "overlay_frac")) if c in internal})
    if {"w", "h"} <= set(feats):
        feats["aspect"] = feats["w"] / feats["h"]
    feats["src"] = internal["source"].astype("category").cat.codes
    if root is not None and len(internal) <= 6000:
        stats = []
        for p in internal["proc_path"]:
            try:
                a = np.asarray(Image.open(Path(root) / p).convert("L").resize((64, 64)), dtype=np.float32)
                stats.append((a.mean(), a.std()))
            except Exception:  # noqa: BLE001
                stats.append((np.nan, np.nan))
        feats[["px_mean", "px_std"]] = np.array(stats)
    y, groups = internal["label"].to_numpy(), internal["group"].to_numpy()
    if min(np.bincount(y.astype(int))) < 10:
        return
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(feats, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=60, max_depth=3).fit(feats.iloc[tr], y[tr])
        oof[te] = clf.predict_proba(feats.iloc[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    rep.stats["metadata_only_label_auc"] = auc
    if auc > cfg.max_meta_auc:
        rep.add("warn", "metadata_shortcut", f"image size / intensity / source alone predict the label with AUC {auc:.2f} (> {cfg.max_meta_auc}): a model can score well without looking at the lesion. Compare against this floor and inspect per-source results")


# ----------------------------------------------------------------------------- ultrasound merge
def audit_ultrasound(manifest: pd.DataFrame, root: Path | None, cfg: AuditConfig | None = None, external_source: str | None = None, label_scheme: str = "binary_no_normal") -> AuditReport:
    cfg = cfg or AuditConfig()
    rep = AuditReport("ultrasound")
    df = manifest
    if not _require(rep, df, ["uid", "source", "class_name", "label", "proc_path", "split", "group"]):
        return rep
    rep.stats["n_images"], rep.stats["by_source"] = int(len(df)), df["source"].value_counts().to_dict()
    if df["uid"].duplicated().any():
        rep.add("error", "uid_collision", f"{int(df['uid'].duplicated().sum())} duplicate uids (two sources used the same id?)")
    if df["proc_path"].duplicated().any():
        rep.add("error", "path_collision", "several rows write to the same processed file; one overwrote the other")
    if df["label"].isna().any():
        rep.add("error", "label_nan", f"{int(df['label'].isna().sum())} rows without a label")
    if label_scheme == "binary_no_normal":
        wrong = df[df["label"].astype(int) != (df["class_name"] == "malignant").astype(int)]
        if len(wrong):
            rep.add("error", "label_mapping", f"{len(wrong)} rows whose label contradicts their class_name; e.g. {wrong['uid'].head(3).tolist()}")
        if (df["class_name"] == "normal").any():
            rep.add("error", "normal_in_binary", "'normal' images present under the binary_no_normal scheme")
    unknown = set(df["class_name"]) - {"benign", "malignant", "normal"}
    if unknown:
        rep.add("warn", "unknown_class", f"unrecognised class names {sorted(unknown)}")
    bad_split = set(df["split"]) - SPLITS
    if bad_split:
        rep.add("error", "split_value", f"unknown split values {sorted(bad_split)}")
    leak = leakage_report(df, keys=("uid", "group"))
    if any(leak.values()):
        rep.add("error", "split_leak", f"identifiers shared across splits: {leak}")
    if external_source is not None:
        ext = df["split"] == "ext_test"
        if not (df.loc[ext, "source"] == external_source).all() or (df.loc[~ext, "source"] == external_source).any():
            rep.add("error", "external_isolation", f"source '{external_source}' must be exactly the ext_test split; it is not")
        if not ext.any():
            rep.add("warn", "no_external", f"external source '{external_source}' has no images")
    _class_counts(rep, df[df["split"].isin(["train", "val", "test"]) | (df["split"] == "ext_test")], cfg)
    comp = df.groupby(["source", "split"]).size().unstack(fill_value=0)
    rep.stats["source_by_split"] = comp.to_dict(orient="index")
    for src, row in comp.iterrows():
        if src != external_source and row.get("train", 0) == 0:
            rep.add("warn", "source_not_in_train", f"source '{src}' contributes nothing to training")
    if "lesion_area_frac" in df and ((df["lesion_area_frac"] < 0) | (df["lesion_area_frac"] > 1)).any():
        rep.add("error", "lesion_frac_range", "lesion_area_frac outside [0, 1]")
    if "overlay_frac" in df:
        rep.stats["images_with_colour_overlay"] = int((df["overlay_frac"] > 0).sum())
    if "mask_path" in df:
        rep.stats["mask_coverage_by_source"] = {s: float(g["mask_path"].fillna("").astype(str).ne("").mean()) for s, g in df.groupby("source")}
    _cross_split_duplicates(rep, df, cfg)
    if root is not None:
        _check_files(rep, df, Path(root), cfg, with_masks=True)
    try:
        source_confounding(rep, df, cfg, Path(root) if root is not None else None)
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        rep.add("warn", "confounding_skipped", f"source-confounding check could not run: {exc}")
    return rep


# ----------------------------------------------------------------------------- VinDr
def audit_vindr(manifest: pd.DataFrame, root: Path | None, findings: pd.DataFrame | None = None, cfg: AuditConfig | None = None,
                name: str = "vindr", official: bool = True) -> AuditReport:
    """Audit of a mammogram table. ``official=False`` (CMMD) has no official test split to protect."""
    cfg = cfg or AuditConfig()
    rep = AuditReport(name)
    df = manifest
    if not _require(rep, df, ["uid", "study_id", "laterality", "view", "label", "split", "proc_path", "group", *(["official_split"] if official else [])]):
        return rep
    rep.stats["n_images"], rep.stats["n_studies"] = int(len(df)), int(df["study_id"].nunique())
    if df["uid"].duplicated().any():
        rep.add("error", "uid_collision", f"{int(df['uid'].duplicated().sum())} duplicate uids")
    if df["label"].isna().any():
        rep.add("error", "label_nan", f"{int(df['label'].isna().sum())} rows without a label")
    if set(df["view"]) - {"CC", "MLO"}:
        rep.add("warn", "view_values", f"unexpected views {sorted(set(df['view']) - {'CC', 'MLO'})}")
    if official:
        moved = df[(df["official_split"] == "test") != (df["split"] == "test")]
        if len(moved):
            rep.add("error", "official_test_changed", f"{len(moved)} images moved into/out of the official test split")
    leak = leakage_report(df, keys=("uid", "group"))
    if any(leak.values()):
        rep.add("error", "split_leak", f"studies/images shared across splits: {leak}")
    per_breast = df.groupby(["study_id", "laterality"])["label"].nunique()
    if (per_breast > 1).any():
        rep.add("error", "breast_label_conflict", f"{int((per_breast > 1).sum())} breasts have different labels on different views")
    dup_views = df.groupby(["study_id", "laterality", "view"]).size()
    if (dup_views > 1).any():
        rep.add("warn", "repeated_view", f"{int((dup_views > 1).sum())} (study, breast, view) combinations have several images; only the first is paired")
    _class_counts(rep, df, cfg)
    if findings is not None and len(findings) and "bx0" in findings:
        m = findings.merge(df[["uid", "proc_w", "proc_h"]], left_on="image_id", right_on="uid", how="inner")
        tol = cfg.box_tolerance_px
        out = ((m.bx0 < -tol) | (m.by0 < -tol) | (m.bx1 > m.proc_w + tol) | (m.by1 > m.proc_h + tol)).mean()
        rep.stats["fraction_boxes_outside_image"] = float(out)
        if out > cfg.max_boxes_outside:
            rep.add("error", "box_mapping", f"{out:.0%} of finding boxes fall outside the stored image: the box -> stored-pixel mapping is wrong")
    if root is not None:
        _check_files(rep, df, Path(root), cfg)
    return rep


def audit_cmmd(manifest: pd.DataFrame, root: Path | None, findings: pd.DataFrame | None = None, cfg: AuditConfig | None = None) -> AuditReport:
    """The mammogram checks plus what is specific to CMMD: breasts without both views, and the D1 / D2 subset shortcut."""
    cfg = cfg or AuditConfig()
    rep = audit_vindr(manifest, root, findings, cfg, name="cmmd", official=False)
    if any(f.code == "schema" for f in rep.errors):
        return rep
    df = manifest
    views = df.groupby(["study_id", "laterality"])["view"].agg(set)
    incomplete = int((~views.map(lambda v: {"CC", "MLO"} <= v)).sum())
    rep.stats["breasts"] = int(len(views))
    if incomplete:
        rep.add("warn", "incomplete_breast", f"{incomplete} of {len(views)} labelled breasts lack a CC or an MLO image and form no native pair")
    if "subset" in df:
        prev = df.drop_duplicates(["study_id", "laterality"]).groupby("subset")["label"].agg(["mean", "size"])
        rep.stats["malignant_rate_by_subset"] = {str(k): dict(malignant_rate=float(v["mean"]), breasts=int(v["size"])) for k, v in prev.iterrows()}
        if len(prev) > 1 and float(prev["mean"].max() - prev["mean"].min()) > cfg.max_prevalence_gap:
            rep.add("warn", "subset_shortcut", "the CMMD subsets differ strongly in malignant rate (" + ", ".join(f"{k} {v['mean']:.0%}" for k, v in prev.iterrows())
                    + "): a model can partly learn the subset instead of the lesion. Use data.cmmd.subsets=[D1] for a shortcut-free protocol, "
                      "or read the per-subset AUROC (`by_stratum` in metrics.json)")
    if "n_study_images" in df:
        by_n = df.drop_duplicates(["study_id", "laterality"]).groupby("n_study_images")["label"].agg(["mean", "size"])
        rep.stats["malignant_rate_by_study_images"] = {int(k): dict(malignant_rate=float(v["mean"]), breasts=int(v["size"])) for k, v in by_n.iterrows()}
    try:
        source_confounding(rep, df, cfg, Path(root) if root is not None else None)
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        rep.add("warn", "confounding_skipped", f"source-confounding check could not run: {exc}")
    return rep


# ----------------------------------------------------------------------------- pair tables
def audit_pairs(pairs: pd.DataFrame, a_df: pd.DataFrame | None, b_df: pd.DataFrame | None, frozen: bool, matched: bool, same_group: bool = False) -> AuditReport:
    """``frozen``: validation/test tables must be 1-to-1; ``matched``: both partners share a label; ``same_group``: native pairs."""
    rep = AuditReport(f"pairs_{pairs['split'].iloc[0] if len(pairs) else 'empty'}")
    if pairs.empty:
        rep.add("error", "empty_pairs", "no pairs")
        return rep
    try:
        assert_pairs_clean(pairs, a_df, b_df)
    except AssertionError as exc:
        rep.add("error", "pair_split", str(exc))
    if pairs["pair_id"].duplicated().any() and frozen:
        rep.add("error", "pair_repeat", "a frozen evaluation table repeats a pair")
    if frozen:
        for col in ("uid_a", "uid_b"):
            if pairs[col].duplicated().any() and (same_group is False):
                rep.add("error", "partner_reuse", f"{col} appears in several evaluation pairs: test samples are not independent")
    if matched and (pairs["label_a"] != pairs["label_b"]).any():
        rep.add("error", "label_mismatch", "label-matched pairs with different labels")
    if same_group and (pairs["group_a"] != pairs["group_b"]).any():
        rep.add("error", "cross_patient", "native pairs span different patients")
    if not set(pairs["label"].unique()) <= {0, 1}:
        rep.add("warn", "label_values", f"labels outside {{0,1}}: {sorted(pairs['label'].unique())}")
    rep.stats.update(
        n_pairs=int(len(pairs)), unique_a=int(pairs["uid_a"].nunique()), unique_b=int(pairs["uid_b"].nunique()),
        max_b_reuse=int(pairs["uid_b"].value_counts().max()), positive_rate=float(pairs["label"].mean()),
    )
    return rep

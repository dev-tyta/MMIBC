"""VinDr-Mammo: parse annotations, build the per-image table, process images, extract lesion boxes.

Important label semantics: VinDr-Mammo has BI-RADS assessments, NOT pathology. ``birads_4plus`` therefore yields a
*radiological suspicion* label (BI-RADS 1-3 -> 0, 4-5 -> 1), which is not comparable to BUSI's biopsy-derived labels.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mmibc.data._resume import run_resumable
from mmibc.data.preprocess import orig_box_to_processed, process_mammogram

log = logging.getLogger(__name__)

IMG_EXT = {".png", ".jpg", ".jpeg"}
REQUIRED_BREAST_COLS = ["study_id", "image_id", "laterality", "view_position", "breast_birads", "split"]
VINDR_SPLIT_MAP = {"training": "train", "train": "train", "test": "test", "testing": "test"}


def parse_birads(value) -> float:
    m = re.search(r"(\d)", str(value))
    return float(m.group(1)) if m else np.nan


def birads_to_label(birads: float, scheme: str) -> float:
    if np.isnan(birads) or birads < 1:
        return np.nan
    if scheme == "birads_4plus":
        return 0.0 if birads <= 3 else 1.0
    if scheme == "birads_3plus":  # Normal = BI-RADS 1-2, Abnormal = 3-5 (the grouping of Petrini et al., arXiv 2503.19945)
        return 0.0 if birads <= 2 else 1.0
    raise ValueError(f"Unknown VinDr label scheme '{scheme}'")


def _find_one(root: Path, name: str) -> Path:
    hits = sorted(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f"'{name}' not found under {root}")
    return hits[0]


def index_images(img_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in Path(img_dir).rglob("*") if p.suffix.lower() in IMG_EXT}


def load_breast_table(ann_dir: Path, img_dir: Path, label_scheme: str = "birads_4plus") -> pd.DataFrame:
    """One row per image with label, study, laterality, view and raw path. Rows without an image or label are dropped (and counted)."""
    df = pd.read_csv(_find_one(ann_dir, "breast-level_annotations.csv"))
    missing_cols = [c for c in REQUIRED_BREAST_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"breast-level_annotations.csv lacks {missing_cols}; found {list(df.columns)}")
    n0 = len(df)
    df = df.drop_duplicates("image_id").copy()
    df["birads"] = df["breast_birads"].map(parse_birads)
    df["label"] = df["birads"].map(lambda b: birads_to_label(b, label_scheme))
    df["laterality"] = df["laterality"].astype(str).str.strip().str.upper().str[0]
    df["view"] = df["view_position"].astype(str).str.strip().str.upper()
    df["official_split"] = df["split"].astype(str).str.strip().str.lower().map(VINDR_SPLIT_MAP)
    index = index_images(img_dir)
    df["src_path"] = df["image_id"].astype(str).map(lambda i: str(index[i]) if i in index else "")
    bad_img, bad_label, bad_split = df["src_path"] == "", df["label"].isna(), df["official_split"].isna()
    log.info("VinDr: %d rows (%d duplicates dropped); missing image=%d, unlabelled=%d, unknown split=%d", n0, n0 - len(df), bad_img.sum(), bad_label.sum(), bad_split.sum())
    df = df[~(bad_img | bad_label | bad_split)].copy()
    df["label"] = df["label"].astype(int)
    df["uid"] = df["image_id"].astype(str)
    df["study_id"] = df["study_id"].astype(str)
    df["csv_w"] = df.get("width", np.nan)
    df["csv_h"] = df.get("height", np.nan)
    df["density"] = df.get("breast_density", np.nan)
    cols = ["uid", "study_id", "laterality", "view", "birads", "density", "label", "official_split", "src_path", "csv_w", "csv_h"]
    return df[cols].reset_index(drop=True)


def load_findings(ann_dir: Path) -> pd.DataFrame:
    f = pd.read_csv(_find_one(ann_dir, "finding_annotations.csv"))
    need = ["image_id", "xmin", "ymin", "xmax", "ymax"]
    if any(c not in f.columns for c in need):
        raise ValueError(f"finding_annotations.csv lacks {[c for c in need if c not in f.columns]}; found {list(f.columns)}")
    f = f.dropna(subset=need).copy()
    f["image_id"] = f["image_id"].astype(str)
    keep = [c for c in ["image_id", "finding_categories", "finding_birads", "xmin", "ymin", "xmax", "ymax"] if c in f.columns]
    return f[keep].reset_index(drop=True)


def _process_row(row: dict) -> dict:
    out_dir = Path(row["_out"])
    dst = out_dir / "images" / row["study_id"] / f"{row['uid']}.png"
    geom = process_mammogram(row["src_path"], dst, row["_long_side"], row["_crop"])
    geom.update(uid=row["uid"], proc_path=str(dst.relative_to(out_dir)))
    return geom


def prepare_vindr(
    img_dir: Path, ann_dir: Path, out_dir: Path, long_side: int = 672, crop: bool = True,
    label_scheme: str = "birads_4plus", workers: int = 8, force: bool = False,
) -> pd.DataFrame:
    """Crop + downscale every image into ``out_dir`` and write ``manifest.csv`` and ``findings.csv`` (boxes in stored-image pixels).

    Resumable: per-image geometry is appended to ``progress.jsonl``; reruns skip finished images.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = load_breast_table(ann_dir, img_dir, label_scheme)
    items = [dict(r, _out=str(out_dir), _long_side=long_side, _crop=crop) for r in table.to_dict("records")]
    done = run_resumable(
        items, _process_row, out_dir / "progress.jsonl", workers=workers, desc="vindr", force=force,
        validate=lambda rec: (out_dir / rec["proc_path"]).exists(),
    )
    geom = pd.DataFrame(list(done.values()))
    geom = geom[geom["uid"].isin(table["uid"])]
    manifest = table.merge(geom, on="uid", how="inner").drop(columns=["src_path"])
    # Image dimensions in the CSV may differ from the PNG (e.g. DICOM->PNG resizing): rescale boxes by the ratio.
    manifest["sx"] = (manifest["orig_w"] / manifest["csv_w"]).where(manifest["csv_w"] > 0, 1.0).fillna(1.0)
    manifest["sy"] = (manifest["orig_h"] / manifest["csv_h"]).where(manifest["csv_h"] > 0, 1.0).fillna(1.0)

    findings = load_findings(ann_dir).merge(manifest[["uid", "crop_x0", "crop_y0", "scale", "sx", "sy", "orig_w", "orig_h"]], left_on="image_id", right_on="uid", how="inner")
    boxes = [
        orig_box_to_processed((r.xmin * r.sx, r.ymin * r.sy, r.xmax * r.sx, r.ymax * r.sy), (r.crop_x0, r.crop_y0), r.scale)
        for r in findings.itertuples()
    ]
    findings[["bx0", "by0", "bx1", "by1"]] = np.array(boxes, dtype=float).reshape(-1, 4)
    area = ((findings["xmax"] - findings["xmin"]) * findings["sx"] * (findings["ymax"] - findings["ymin"]) * findings["sy"]).clip(lower=0)
    findings["area_frac"] = area / (findings["orig_w"] * findings["orig_h"])
    lesion_frac = findings.groupby("uid")["area_frac"].sum().clip(upper=1.0)
    manifest["lesion_area_frac"] = manifest["uid"].map(lesion_frac).fillna(0.0)
    findings = findings.drop(columns=["uid", "crop_x0", "crop_y0", "scale", "sx", "sy"])

    manifest.to_csv(out_dir / "manifest.csv", index=False)
    findings.to_csv(out_dir / "findings.csv", index=False)
    log.info("VinDr manifest: %d images, %d findings -> %s", len(manifest), len(findings), out_dir)
    return manifest

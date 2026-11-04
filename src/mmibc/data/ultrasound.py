"""Ultrasound sources (BUSI and any number of additional ones) -> one unified, de-duplicated table.

A source is described by a table with columns:
    uid, source, raw_path, class_name (benign|malignant|normal), mask_paths ('|'-joined, may be empty), patient_id (may be NaN)
``scan_busi`` and ``scan_class_folders`` build that table from common layouts; ``load_source_manifest`` accepts a
hand-written CSV for any other dataset (e.g. BUS-BRA, BrEaST-Lesions-USG) once its layout has been inspected.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from mmibc.data._resume import run_resumable
from mmibc.data.audit import bits_to_hex, color_overlay_fraction, dhash_bits, inpaint_color_overlay
from mmibc.data.preprocess import downscale_long_side

log = logging.getLogger(__name__)

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ("benign", "malignant", "normal")
MASK_RE = re.compile(r"^(?P<base>.+)_mask(?:_\d+)?$")
REQUIRED_SOURCE_COLS = ["uid", "source", "raw_path", "class_name"]


def _find_busi_root(raw_dir: Path) -> Path:
    for d in [raw_dir, *sorted(p for p in raw_dir.rglob("*") if p.is_dir())]:
        if (d / "benign").is_dir() and (d / "malignant").is_dir():
            return d
    raise FileNotFoundError(f"No BUSI-style folder (benign/ malignant/ [normal/]) found under {raw_dir}")


def scan_busi(raw_dir: Path, source: str = "busi") -> pd.DataFrame:
    root = _find_busi_root(Path(raw_dir))
    rows = []
    for cls in CLASS_NAMES:
        d = root / cls
        if not d.is_dir():
            continue
        masks: dict[str, list[str]] = {}
        files = []
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            m = MASK_RE.match(p.stem)
            if m:
                masks.setdefault(m.group("base"), []).append(str(p))
            else:
                files.append(p)
        for p in files:
            num = re.search(r"\((\d+)\)", p.stem)
            uid = f"{source}_{cls}_{num.group(1)}" if num else f"{source}_{cls}_{re.sub(r'[^A-Za-z0-9]+', '_', p.stem)}"
            rows.append(dict(uid=uid, source=source, raw_path=str(p), class_name=cls, mask_paths="|".join(masks.get(p.stem, [])), patient_id=np.nan))
    df = pd.DataFrame(rows)
    log.info("BUSI scan: %s", df["class_name"].value_counts().to_dict())
    return df


def scan_class_folders(raw_dir: Path, source: str, class_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Generic layout: any folder named benign / malignant / normal (case-insensitive, or mapped via class_map) holding images."""
    class_map = {k.lower(): v for k, v in (class_map or {}).items()}
    rows = []
    for p in sorted(Path(raw_dir).rglob("*")):
        if p.suffix.lower() not in IMG_EXT or MASK_RE.match(p.stem):
            continue
        cls = next((class_map.get(part.lower(), part.lower()) for part in reversed(p.parts) if class_map.get(part.lower(), part.lower()) in CLASS_NAMES), None)
        if cls is None:
            continue
        rows.append(dict(uid=f"{source}_{len(rows):06d}", source=source, raw_path=str(p), class_name=cls, mask_paths="", patient_id=np.nan))
    return pd.DataFrame(rows)


def load_source_manifest(csv_path: Path, raw_dir: Path, source: str) -> pd.DataFrame:
    """CSV with: uid, raw_path (relative to raw_dir or absolute), class_name, optional mask_paths ('|'-joined), optional patient_id."""
    df = pd.read_csv(csv_path)
    df["source"] = source
    missing = [c for c in ["uid", "raw_path", "class_name"] if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} lacks columns {missing}")
    for col, default in (("mask_paths", ""), ("patient_id", np.nan)):
        if col not in df.columns:
            df[col] = default
    df["mask_paths"] = df["mask_paths"].fillna("")
    absolutise = lambda p: str(p) if Path(p).is_absolute() else str(Path(raw_dir) / p)
    df["raw_path"] = df["raw_path"].map(absolutise)
    df["mask_paths"] = df["mask_paths"].map(lambda s: "|".join(absolutise(x) for x in s.split("|") if x))
    df["uid"] = source + "_" + df["uid"].astype(str)
    return df[REQUIRED_SOURCE_COLS + ["mask_paths", "patient_id"]]


def apply_label_scheme(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    df = df.copy()
    if scheme == "binary_no_normal":
        df = df[df["class_name"].isin(["benign", "malignant"])]
        df["label"] = (df["class_name"] == "malignant").astype(int)
    elif scheme == "binary_normal_as_benign":
        log.warning("Merging 'normal' into 'benign' inflates specificity; use only to reproduce the legacy protocol.")
        df["label"] = (df["class_name"] == "malignant").astype(int)
    elif scheme == "three_class":
        df["label"] = df["class_name"].map({"normal": 0, "benign": 1, "malignant": 2}).astype(int)
    else:
        raise ValueError(f"Unknown ultrasound label scheme '{scheme}'")
    return df.reset_index(drop=True)


def merge_masks(paths: list[str], size: tuple[int, int]) -> np.ndarray:
    """Union of several binary masks (BUSI has up to three per image) as uint8 0/255, resized to ``size`` = (w, h)."""
    out = np.zeros((size[1], size[0]), dtype=bool)
    for p in paths:
        m = Image.open(p).convert("L")
        if m.size != size:
            m = m.resize(size, Image.NEAREST)
        out |= np.asarray(m) > 127
    return out.astype(np.uint8) * 255


def _process_us(row: dict) -> dict:
    out_dir = Path(row["_out"])
    img = Image.open(row["raw_path"])
    overlay = color_overlay_fraction(img)
    proc = inpaint_color_overlay(img) if (row["_inpaint"] and overlay > 0) else img.convert("L")
    orig_w, orig_h = proc.size
    proc, scale = downscale_long_side(proc, row["_long_side"])
    rel_img = Path("images") / f"{row['uid']}.png"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    proc.save(out_dir / rel_img)
    rec = dict(
        uid=row["uid"], proc_path=str(rel_img), overlay_frac=overlay, dhash=bits_to_hex(dhash_bits(proc)),
        orig_w=orig_w, orig_h=orig_h, scale=scale, proc_w=proc.width, proc_h=proc.height, mask_path="", lesion_area_frac=float("nan"),
    )
    mask_paths = [p for p in str(row["mask_paths"]).split("|") if p]
    if mask_paths:
        m = merge_masks(mask_paths, (orig_w, orig_h))
        mimg, _ = downscale_long_side(Image.fromarray(m), row["_long_side"])
        rel_mask = Path("masks") / f"{row['uid']}.png"
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
        mimg.save(out_dir / rel_mask)
        rec["mask_path"] = str(rel_mask)
        rec["lesion_area_frac"] = float((np.asarray(mimg) > 127).mean())
    elif row["class_name"] == "normal":
        rec["lesion_area_frac"] = 0.0
    return rec


def prepare_ultrasound(
    scan: pd.DataFrame, out_dir: Path, label_scheme: str = "binary_no_normal", long_side: int = 672, inpaint: bool = True,
    workers: int = 8, force: bool = False,
) -> pd.DataFrame:
    """Process every image of a (possibly multi-source) scan table and write ``manifest.csv``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = apply_label_scheme(scan, label_scheme)
    items = [dict(r, _out=str(out_dir), _long_side=long_side, _inpaint=inpaint) for r in table.to_dict("records")]
    done = run_resumable(
        items, _process_us, out_dir / "progress.jsonl", workers=workers, desc="ultrasound", force=force,
        validate=lambda rec: (out_dir / rec["proc_path"]).exists(),
    )
    geom = pd.DataFrame(list(done.values()))
    manifest = table.drop(columns=["raw_path", "mask_paths"]).merge(geom[geom["uid"].isin(table["uid"])], on="uid", how="inner")
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    log.info("Ultrasound manifest: %d images (%s) -> %s", len(manifest), manifest["source"].value_counts().to_dict(), out_dir)
    return manifest

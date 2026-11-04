"""Source manifests for ultrasound datasets whose layout is not "benign/ malignant/ normal/ folders".

``prepare us`` needs, for every ultrasound source, a table ``uid, raw_path, class_name[, mask_paths, patient_id]`` (paths relative to the
source's raw folder). Two sources need a builder because their labels live in a spreadsheet:

* ``bus_bra``            ``BUSBRA/bus_data.csv`` (ID, Case, Pathology, ...), ``BUSBRA/Images/<ID>.png``, ``BUSBRA/Masks/mask_<rest of ID>.png``.
                         The layout was confirmed on the first real Colab run (1,875 images: 1,268 benign / 607 malignant, 1,064 patients;
                         the left and right breast of a patient share ``Case``, which becomes ``patient_id`` so they never straddle splits).
* ``breast_lesions_usg`` the TCIA clinical spreadsheet (CaseID, Image_filename, Mask_tumor_filename, Classification, ...) next to a flat folder of
                         ``caseNNN.png`` / ``caseNNN_tumor.png`` (256 cases: 154 benign / 98 malignant / 4 normal). One image per case.

``build_manifest`` is called by ``prepare us`` when ``source_manifest.csv`` is missing, and by ``mmibc manifest <source>``. A manifest that
already exists is never overwritten (you may have edited it) unless ``force`` is set. Every row is checked against the files on disk;
rows whose image is missing are dropped with a warning, and a source that loses more than 5 % of its rows is refused rather than silently shrunk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

log = logging.getLogger(__name__)

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MANIFEST_NAME = "source_manifest.csv"
MAX_DROPPED_FRACTION = 0.05


def _file_index(raw_dir: Path) -> dict[str, Path]:
    """File name -> path relative to ``raw_dir`` for every image under it (the first one wins; duplicates are reported)."""
    index: dict[str, Path] = {}
    dup = 0
    for p in sorted(Path(raw_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            if p.name in index:
                dup += 1
                continue
            index[p.name] = p.relative_to(raw_dir)
    if dup:
        log.warning("%s: %d image file names occur more than once; the first occurrence is used", raw_dir, dup)
    return index


def _find_one(raw_dir: Path, pattern: str, what: str) -> Path:
    hits = sorted(p for p in Path(raw_dir).rglob("*") if p.is_file() and p.name.lower() == pattern.lower())
    if not hits:
        raise FileNotFoundError(f"{what}: no file named '{pattern}' under {raw_dir}. Run `mmibc inspect {Path(raw_dir).name}` to see the layout.")
    return hits[0]


def _checked(df: pd.DataFrame, raw_dir: Path, source: str) -> pd.DataFrame:
    """Drop rows whose image does not exist; refuse the source if that is more than a few rows."""
    exists = df["raw_path"].map(lambda p: (Path(raw_dir) / p).is_file())
    lost = int((~exists).sum())
    if lost:
        frac = lost / max(len(df), 1)
        msg = f"{source}: {lost} of {len(df)} listed images are not on disk (e.g. {df.loc[~exists, 'raw_path'].iloc[0]})"
        if frac > MAX_DROPPED_FRACTION:
            raise RuntimeError(msg + f"; that is {frac:.0%}, more than {MAX_DROPPED_FRACTION:.0%}. The download is incomplete or the layout differs.")
        log.warning("%s; they are dropped", msg)
    return df[exists].reset_index(drop=True)


def _lower(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def build_bus_bra_manifest(raw_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    meta_path = _find_one(raw_dir, "bus_data.csv", "BUS-BRA metadata")
    meta = pd.read_csv(meta_path)
    missing = [c for c in ("ID", "Pathology", "Case") if c not in meta.columns]
    if missing:
        raise ValueError(f"{meta_path} lacks columns {missing}; found {list(meta.columns)}")
    index = _file_index(raw_dir)
    classes = _lower(meta["Pathology"])
    bad = sorted(set(classes) - {"benign", "malignant"})
    if bad:
        raise ValueError(f"{meta_path}: unexpected Pathology values {bad} (expected benign / malignant)")

    def mask_of(i: str) -> str:
        rest = str(i).split("_", 1)[1] if "_" in str(i) else str(i)
        hit = index.get(f"mask_{rest}.png")
        return hit.as_posix() if hit is not None else ""

    rows = pd.DataFrame(
        dict(
            uid=meta["ID"].astype(str),
            raw_path=[index[f"{i}.png"].as_posix() if f"{i}.png" in index else f"{i}.png" for i in meta["ID"].astype(str)],
            class_name=classes,
            mask_paths=[mask_of(i) for i in meta["ID"]],
            patient_id=meta["Case"],
        )
    )
    rows = _checked(rows, raw_dir, "bus_bra")
    no_mask = int((rows["mask_paths"] == "").sum())
    if no_mask:
        log.warning("bus_bra: %d images have no mask file (lesion-size matching and XAI ground truth skip them)", no_mask)
    return rows


def build_breast_lesions_usg_manifest(raw_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    sheet = None
    for x in sorted(raw_dir.rglob("*.xlsx")):
        try:
            for name, df in pd.read_excel(x, sheet_name=None).items():
                if {"Image_filename", "Classification"} <= set(df.columns):
                    sheet, sheet_src = df, f"{x.name} [{name}]"
                    break
        except ImportError as exc:
            raise ImportError("reading the BrEaST spreadsheet needs openpyxl: pip install openpyxl") from exc
        if sheet is not None:
            break
    if sheet is None:
        raise FileNotFoundError(f"BrEaST-Lesions-USG: no spreadsheet with columns Image_filename and Classification under {raw_dir}. "
                                f"Run `mmibc download breast_lesions_usg` (it fetches the clinical-data .xlsx as well as the images zip).")
    sheet = sheet.dropna(subset=["Image_filename"])
    index = _file_index(raw_dir)
    classes = _lower(sheet["Classification"])
    bad = sorted(set(classes) - {"benign", "malignant", "normal"})
    if bad:
        raise ValueError(f"{sheet_src}: unexpected Classification values {bad} (expected benign / malignant / normal)")
    names = sheet["Image_filename"].astype(str).str.strip()
    masks = sheet["Mask_tumor_filename"] if "Mask_tumor_filename" in sheet.columns else pd.Series([None] * len(sheet), index=sheet.index)

    def mask_of(m) -> str:
        if not isinstance(m, str) or not m.strip():
            return ""
        hit = index.get(m.strip())
        return hit.as_posix() if hit is not None else ""

    rows = pd.DataFrame(
        dict(
            uid=[Path(n).stem for n in names],
            raw_path=[index[n].as_posix() if n in index else n for n in names],
            class_name=classes.to_numpy(),
            mask_paths=[mask_of(m) for m in masks],
            patient_id=[f"breast_case_{c}" for c in sheet["CaseID"]] if "CaseID" in sheet.columns else None,
        )
    )
    rows = _checked(rows, raw_dir, "breast_lesions_usg")
    if "Verification" in sheet.columns:
        log.info("breast_lesions_usg labels: %s | verification: %s", rows["class_name"].value_counts().to_dict(), sheet["Verification"].astype(str).value_counts().to_dict())
    return rows


BUILDERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "bus_bra": build_bus_bra_manifest,
    "breast_lesions_usg": build_breast_lesions_usg_manifest,
}


def build_manifest(source: str, raw_dir: Path, force: bool = False) -> Path:
    """Write ``<raw_dir>/source_manifest.csv`` for ``source`` and return its path. An existing manifest is kept unless ``force``."""
    raw_dir = Path(raw_dir)
    out = raw_dir / MANIFEST_NAME
    if out.exists() and not force:
        log.info("%s already exists; kept (use --force to rebuild it)", out)
        return out
    if source not in BUILDERS:
        raise KeyError(f"no manifest builder for '{source}'; known: {sorted(BUILDERS)}. For other sources write {out} by hand "
                       "(columns: uid, raw_path, class_name[, mask_paths, patient_id]).")
    df = BUILDERS[source](raw_dir)
    if df.empty:
        raise RuntimeError(f"{source}: the manifest would be empty ({raw_dir})")
    df.to_csv(out, index=False)
    log.info("wrote %s: %d images %s", out, len(df), df["class_name"].value_counts().to_dict())
    return out

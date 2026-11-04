"""CMMD (Chinese Mammography Database, TCIA): DICOM mammograms with BIOPSY-confirmed benign / malignant labels.

Unlike VinDr-Mammo (BI-RADS = radiological suspicion), these labels are pathology, so they mean the same thing as BUSI / BUS-BRA /
BrEaST labels. That is what makes label-matched mammogram + ultrasound pairs meaningful. CMMD ships no lesion boxes: there is no
localisation ground truth for the mammogram stream and ``size_conditioned`` pairing is not available for it.

Layout (confirmed on a real copy; ``raw/cmmd/``)::

    CMMD_clinicaldata_revision.xlsx     Sheet1: ID1, LeftRight, Age, number, abnormality, classification, subtype
    metadata.csv                        TCIA download manifest (Subject ID, Number of Images, ...); optional, used as a completeness check
    CMMD/D1-0001/<date>/<series>/1-1.dcm ... 1-N.dcm

What the real files taught (so the code does not guess):

* DICOM: explicit VR little endian, MONOCHROME2, 8-bit, ~2294 x 1914, ``ImageLaterality`` present, ``Laterality`` / ``ViewPosition``
  absent; the view is only in ``ViewCodeSequence[0].CodeMeaning`` ("cranio-caudal" / "medio-lateral oblique").
* A study holds 2 images (CC + MLO of the biopsied breast) or 4 (both breasts). The clinical sheet labels ONE breast per row
  (``LeftRight``); 97 patients have two rows (one per breast, so 1,872 labelled breasts for 1,775 patients). The contralateral breast
  of a 4-image study has no recorded finding: it is NOT labelled benign, it is excluded.
* Two subsets, ``D1`` and ``D2``, differ enormously in prevalence (D1 51 % malignant, D2 98 %). A model can score well by recognising the
  subset, so the audit reports it, ``data.cmmd.subsets`` can restrict the data, and evaluation reports AUROC per subset.

Needs ``pydicom`` (``pip install pydicom``).
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from mmibc.data._resume import run_resumable
from mmibc.data.preprocess import process_mammogram_array, to_uint8

log = logging.getLogger(__name__)

PATIENT_RE = re.compile(r"D\d-\d{3,5}")
MAX_UNUSABLE_FRACTION = 0.05
CLINICAL_COLS = ("ID1", "LeftRight", "classification")
FINDINGS_COLUMNS = ["image_id", "finding_categories", "finding_birads", "xmin", "ymin", "xmax", "ymax", "bx0", "by0", "bx1", "by1", "area_frac"]


def _pydicom():
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("CMMD is DICOM: install pydicom (pip install pydicom  or  pip install 'mmibc[dicom]')") from exc
    return pydicom


# ----------------------------------------------------------------------------- DICOM
def view_from_text(text: str) -> str:
    """'cranio-caudal' / 'CC' -> 'CC'; 'medio-lateral oblique' / 'MLO' -> 'MLO'; anything else -> ''."""
    t = re.sub(r"[^a-z]", "", str(text).lower())
    if t in ("cc", "craniocaudal") or t.startswith("cranio"):
        return "CC"
    if t in ("mlo", "mediolateraloblique") or "oblique" in t:
        return "MLO"
    return ""


def patient_of(path: Path, header_id: str = "") -> str:
    """Patient id: the DICOM PatientID if it looks like D1-0001, else the folder name on the path."""
    if PATIENT_RE.fullmatch(str(header_id).strip()):
        return str(header_id).strip()
    for part in reversed(Path(path).parts):
        if PATIENT_RE.fullmatch(part):
            return part
    return str(header_id).strip()


def read_header(path: Path) -> dict:
    """Header-only read: patient, breast side, view (and where each came from), instance number and pixel dimensions."""
    pydicom = _pydicom()
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    side = str(ds.get("ImageLaterality") or ds.get("Laterality") or "").strip().upper()[:1]
    side = side if side in ("L", "R") else ""
    view, view_src = "", ""
    seq = ds.get("ViewCodeSequence")
    if seq:
        view = view_from_text(getattr(seq[0], "CodeMeaning", ""))
        view_src = "view_code_sequence" if view else ""
    if not view and ds.get("ViewPosition"):
        view = view_from_text(ds.ViewPosition)
        view_src = "view_position" if view else ""
    try:
        instance = int(ds.get("InstanceNumber"))
    except (TypeError, ValueError):
        instance = 0
    return dict(
        patient=patient_of(path, ds.get("PatientID", "")), laterality=side, view=view, view_src=view_src, instance=instance,
        rows=int(ds.get("Rows", 0) or 0), cols=int(ds.get("Columns", 0) or 0), series_uid=str(ds.get("SeriesInstanceUID", "")),
    )


def read_pixels(path: Path) -> np.ndarray:
    """Decode one mammogram to uint8, white = dense: 8-bit data is kept as is, deeper data is windowed to 0.5-99.5 %, MONOCHROME1 is inverted."""
    pydicom = _pydicom()
    ds = pydicom.dcmread(path)
    try:
        arr = ds.pixel_array
    except Exception as exc:  # noqa: BLE001 - compressed transfer syntaxes need an extra decoder
        ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "?")
        raise RuntimeError(f"{path}: cannot decode the pixel data (transfer syntax {ts}): {exc}. Compressed DICOM needs a decoder: "
                           "pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg  (or python-gdcm)") from exc
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise RuntimeError(f"{path}: expected a single-channel 2-D mammogram, got an array of shape {arr.shape}")
    if arr.dtype == np.uint8:
        gray = arr
    else:
        a = arr.astype(np.float32)
        slope, intercept = float(ds.get("RescaleSlope", 1) or 1), float(ds.get("RescaleIntercept", 0) or 0)
        gray = to_uint8(a * slope + intercept)
    if str(ds.get("PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        gray = 255 - gray
    return np.ascontiguousarray(gray)


# ----------------------------------------------------------------------------- clinical sheet
def load_clinical(raw_dir: Path) -> pd.DataFrame:
    """One row per (patient, breast): label 1 = malignant, 0 = benign. Several lesions in one breast: malignant wins."""
    raw_dir = Path(raw_dir)
    sheet, src = None, ""
    candidates = sorted([*raw_dir.rglob("*.xlsx"), *raw_dir.rglob("*.xls"), *raw_dir.rglob("*.csv")])
    for f in candidates:
        try:
            tables = {"": pd.read_csv(f)} if f.suffix == ".csv" else pd.read_excel(f, sheet_name=None)
        except ImportError as exc:
            raise ImportError("reading the CMMD clinical spreadsheet needs openpyxl: pip install openpyxl") from exc
        except Exception:  # noqa: BLE001 - an unrelated file that is not a table
            continue
        for name, df in tables.items():
            df.columns = [str(c).strip() for c in df.columns]
            if set(CLINICAL_COLS) <= set(df.columns):
                sheet, src = df, f"{f.name}{f' [{name}]' if name else ''}"
                break
        if sheet is not None:
            break
    if sheet is None:
        raise FileNotFoundError(f"CMMD: no clinical spreadsheet with columns {list(CLINICAL_COLS)} under {raw_dir}. "
                                "It is CMMD_clinicaldata_revision.xlsx on the TCIA page (labels are NOT inside the DICOM files).")
    cls = sheet["classification"].astype(str).str.strip().str.lower()
    bad = sorted(set(cls) - {"benign", "malignant"})
    if bad:
        raise ValueError(f"{src}: unexpected classification values {bad} (expected Benign / Malignant)")
    side = sheet["LeftRight"].astype(str).str.strip().str.upper().str[:1]
    if (~side.isin(["L", "R"])).any():
        raise ValueError(f"{src}: LeftRight must be L or R, found {sorted(set(sheet['LeftRight'].astype(str)))}")
    df = pd.DataFrame(dict(
        patient=sheet["ID1"].astype(str).str.strip(), laterality=side.to_numpy(), label=(cls == "malignant").astype(int).to_numpy(),
        age=sheet["Age"].to_numpy() if "Age" in sheet else np.nan,
        abnormality=sheet["abnormality"].astype(str).to_numpy() if "abnormality" in sheet else "",
        subtype=sheet["subtype"].astype(str).replace("nan", "").to_numpy() if "subtype" in sheet else "",
    ))
    g = df.groupby(["patient", "laterality"])
    mixed = int((g["label"].nunique() > 1).sum())
    if mixed:
        log.warning("CMMD: %d breasts list lesions with different pathology; the breast is labelled malignant", mixed)
    out = g.agg(label=("label", "max"), n_lesions=("label", "size"), age=("age", "first"), abnormality=("abnormality", "first"), subtype=("subtype", "first")).reset_index()
    log.info("CMMD clinical sheet %s: %d rows -> %d labelled breasts of %d patients (%s)", src, len(df), len(out), out["patient"].nunique(), out["label"].value_counts().rename({0: "benign", 1: "malignant"}).to_dict())
    return out


# ----------------------------------------------------------------------------- header scan + join
def scan_headers(raw_dir: Path, cache: Path | None = None, workers: int = 8, force: bool = False) -> pd.DataFrame:
    """Header of every ``.dcm`` under ``raw_dir`` (``src_path`` relative to it). Cached in ``cache`` while the file count is unchanged."""
    raw_dir = Path(raw_dir)
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".dcm")
    if not files:
        other = sum(1 for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        hint = (f" It holds {other} PNG/JPEG files, but their breast side and view cannot be recovered without the DICOM headers; "
                "use the TCIA DICOM download.") if other else ""
        raise FileNotFoundError(f"CMMD: no .dcm files under {raw_dir}.{hint} Run `mmibc inspect cmmd` to see what is there.")
    if cache is not None and Path(cache).exists() and not force:
        prev = pd.read_csv(cache, keep_default_na=False)
        if len(prev) == len(files) and set(prev["src_path"]) == {p.relative_to(raw_dir).as_posix() for p in files}:
            log.info("CMMD headers: %d files (cached in %s)", len(prev), cache)
            return prev
    _pydicom()

    def one(p: Path) -> dict:
        try:
            return dict(read_header(p), src_path=p.relative_to(raw_dir).as_posix(), error="")
        except Exception as exc:  # noqa: BLE001 - recorded, counted, and refused if there are many
            return dict(patient=patient_of(p), laterality="", view="", view_src="", instance=0, rows=0, cols=0, series_uid="",
                        src_path=p.relative_to(raw_dir).as_posix(), error=f"{type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        rows = list(tqdm(ex.map(one, files), total=len(files), desc="cmmd headers"))
    df = pd.DataFrame(rows)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    return df


def _completeness(headers: pd.DataFrame, raw_dir: Path) -> pd.Series | None:
    """Images per patient according to TCIA's metadata.csv (None when absent); warns when files are missing from disk."""
    hits = sorted(Path(raw_dir).rglob("metadata.csv"))
    if not hits:
        return None
    meta = pd.read_csv(hits[0])
    if not {"Subject ID", "Number of Images"} <= set(meta.columns):
        return None
    want = meta.groupby("Subject ID")["Number of Images"].sum()
    have = headers.groupby("patient").size().reindex(want.index).fillna(0).astype(int)
    short = want[have < want]
    if len(short):
        frac = len(short) / max(len(want), 1)
        msg = f"CMMD: {len(short)} of {len(want)} patients have fewer DICOM files than metadata.csv lists (e.g. {short.index[:3].tolist()})"
        if frac > MAX_UNUSABLE_FRACTION:
            raise RuntimeError(msg + f"; that is {frac:.0%}. The download is incomplete: rerun the NBIA Data Retriever with the same manifest (or delete metadata.csv if you meant to use a part of CMMD).")
        log.warning(msg)
    return want


def build_cmmd_table(raw_dir: Path, headers: pd.DataFrame | None = None, cache: Path | None = None, workers: int = 8, force: bool = False) -> pd.DataFrame:
    """One row per LABELLED image: uid, study_id (= patient), laterality, view, label, subset, n_study_images, ... and ``src_path``."""
    raw_dir = Path(raw_dir)
    h = scan_headers(raw_dir, cache, workers, force) if headers is None else headers.copy()
    if "error" not in h:
        h["error"] = ""
    h["error"] = h["error"].fillna("")
    n_all = len(h)
    n_images = _completeness(h, raw_dir)
    h["n_study_images"] = h["patient"].map(h.groupby("patient").size())
    if n_images is not None:
        h["n_study_images"] = h["patient"].map(n_images).fillna(h["n_study_images"]).astype(int)
    clinical = load_clinical(raw_dir)

    # fallbacks for headers that lack a tag; every use is counted and logged
    clin_sides = clinical.groupby("patient")["laterality"].agg(lambda s: sorted(set(s)))
    need_side = (h["laterality"] == "") & (h["error"] == "") & (h["n_study_images"] == 2)
    only_side = h["patient"].map(clin_sides).map(lambda s: s[0] if isinstance(s, list) and len(s) == 1 else "")
    h.loc[need_side & (only_side != ""), "laterality"] = only_side[need_side & (only_side != "")]
    n_side_fallback = int((need_side & (only_side != "")).sum())
    need_view = (h["view"] == "") & (h["error"] == "") & (h["instance"] > 0)
    h.loc[need_view, "view"] = np.where(h.loc[need_view, "instance"] % 2 == 1, "CC", "MLO")  # real files: instance 1 = CC, 2 = MLO, 3 = CC, 4 = MLO
    h.loc[need_view, "view_src"] = "instance_parity"
    if n_side_fallback or need_view.any():
        log.warning("CMMD: %d images had no breast side (taken from the clinical sheet) and %d no view (taken from the instance number)", n_side_fallback, int(need_view.sum()))

    unusable = (h["error"] != "") | (h["laterality"] == "") | (h["view"] == "")
    if unusable.any():
        frac = float(unusable.mean())
        first = h.loc[unusable, "error"].replace("", np.nan).dropna()
        msg = f"CMMD: {int(unusable.sum())} of {n_all} DICOM files have no usable side/view" + (f" (e.g. {first.iloc[0]})" if len(first) else "")
        if frac > MAX_UNUSABLE_FRACTION:
            raise RuntimeError(msg + f"; that is {frac:.0%}. Run `mmibc inspect cmmd` and check the headers.")
        log.warning("%s; they are dropped", msg)
    h = h[~unusable]

    # one image per (patient, breast, view): re-acquisitions keep the lowest instance number
    h = h.sort_values(["patient", "laterality", "view", "instance", "src_path"])
    dup = h.duplicated(["patient", "laterality", "view"], keep="first")
    if dup.any():
        log.warning("CMMD: %d extra images for a (patient, breast, view) were ignored (kept the lowest instance number)", int(dup.sum()))
    h = h[~dup]

    t = h.merge(clinical, on=["patient", "laterality"], how="inner")
    # an unlabelled contralateral breast is excluded, but a patient whose imaged sides match NO clinical side is a data problem
    imaged = h.groupby("patient")["laterality"].agg(set)
    labelled = clinical.groupby("patient")["laterality"].agg(set)
    common = imaged.index.intersection(labelled.index)
    mismatch = [p for p in common if not (imaged[p] & labelled[p])]
    n_pat = max(len(common), 1)
    if mismatch:
        msg = f"CMMD: for {len(mismatch)} of {n_pat} patients the breast in the DICOM headers differs from the clinical sheet (e.g. {mismatch[:3]})"
        if len(mismatch) / n_pat > MAX_UNUSABLE_FRACTION:
            raise RuntimeError(msg + ". The side tags or the sheet are not what this code expects.")
        log.warning(msg)
    n_unlabelled = len(h) - len(t)
    no_images = int(len(clinical) - clinical.merge(h[["patient", "laterality"]].drop_duplicates(), on=["patient", "laterality"]).shape[0])
    log.info("CMMD: %d labelled images from %d patients; %d images of unlabelled (contralateral) breasts excluded; %d labelled breasts have no image",
             len(t), t["patient"].nunique(), n_unlabelled, no_images)
    if t.empty:
        raise RuntimeError("CMMD: no image could be matched to a clinical label (patient id and breast side)")
    t["uid"] = t["patient"] + "_" + t["laterality"] + "_" + t["view"]
    t["study_id"] = t["patient"]
    t["subset"] = t["patient"].str.extract(r"^(D\d)", expand=False).fillna("D?")
    cols = ["uid", "study_id", "laterality", "view", "label", "subset", "n_study_images", "n_lesions", "age", "abnormality", "subtype", "view_src", "src_path"]
    return t[cols].sort_values("uid").reset_index(drop=True)


# ----------------------------------------------------------------------------- prepare
def _process_row(row: dict) -> dict:
    out_dir = Path(row["_out"])
    dst = out_dir / "images" / row["study_id"] / f"{row['uid']}.png"
    geom = process_mammogram_array(read_pixels(Path(row["_raw"]) / row["src_path"]), dst, row["_long_side"], row["_crop"])
    geom.update(uid=row["uid"], proc_path=str(dst.relative_to(out_dir)))
    return geom


def prepare_cmmd(raw_dir: Path, out_dir: Path, long_side: int = 672, crop: bool = True, workers: int = 8, force: bool = False) -> pd.DataFrame:
    """DICOM -> cropped, downscaled PNG for every labelled image, plus ``manifest.csv`` and an (empty) ``findings.csv``. Resumable.

    All subsets are processed; ``data.cmmd.subsets`` selects among them when the splits are built, so changing it needs no re-prepare.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_cmmd_table(raw_dir, cache=out_dir / "headers.csv", workers=workers, force=force)
    items = [dict(r, _out=str(out_dir), _raw=str(raw_dir), _long_side=long_side, _crop=crop) for r in table.to_dict("records")]
    done = run_resumable(items, _process_row, out_dir / "progress.jsonl", workers=workers, desc="cmmd", force=force,
                         validate=lambda rec: (out_dir / rec["proc_path"]).exists())
    geom = pd.DataFrame(list(done.values()))
    manifest = table.merge(geom, on="uid", how="inner").drop(columns=["src_path"])
    manifest["lesion_area_frac"] = np.nan  # no lesion boxes in CMMD
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    pd.DataFrame(columns=FINDINGS_COLUMNS).to_csv(out_dir / "findings.csv", index=False)
    log.info("CMMD manifest: %d images, %d breasts, %d patients, %s -> %s", len(manifest), manifest[["study_id", "laterality"]].drop_duplicates().shape[0],
             manifest["study_id"].nunique(), manifest["label"].value_counts().rename({0: "benign", 1: "malignant"}).to_dict(), out_dir)
    return manifest

"""Tiny fake datasets with the REAL on-disk schemas, for tests and the Colab smoke test (no downloads needed).

Images are small noisy fields with a brighter blob whose intensity depends on the label, so a model can actually
learn from them and pipeline bugs (leakage, mismatched labels) surface as failed assertions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _blob_image(rng: np.random.Generator, h: int, w: int, label: int, border: int = 0, color: bool = False) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    img = rng.normal(70, 12, (h, w)).clip(0, 255)
    bh, bw = int(h * rng.uniform(0.15, 0.3)), int(w * rng.uniform(0.15, 0.3))
    y0, x0 = rng.integers(h // 8, h - bh - h // 8), rng.integers(w // 8, w - bw - w // 8)
    img[y0 : y0 + bh, x0 : x0 + bw] += 60 + 70 * label
    out = img.astype(np.uint8)
    if border:
        out[:border] = 0
        out[-border:] = 0
        out[:, :border] = 0
        out[:, -border:] = 0
    return out, (x0, y0, x0 + bw, y0 + bh)


def make_fake_vindr(raw_root: Path, n_studies: int = 40, seed: int = 0, size: tuple[int, int] = (96, 144)) -> None:
    """Writes raw_root/vindr_images/images_png/<study>/<image>.png and raw_root/vindr_annotations/{breast-level,finding}_annotations.csv."""
    rng = np.random.default_rng(seed)
    img_root, ann_root = Path(raw_root) / "vindr_images" / "images_png", Path(raw_root) / "vindr_annotations"
    ann_root.mkdir(parents=True, exist_ok=True)
    breast, findings = [], []
    for s in range(n_studies):
        sid = f"study{s:03d}"
        split = "test" if s >= int(0.8 * n_studies) else "training"
        for lat in ("L", "R"):
            birads = int(rng.choice([1, 2, 3, 4, 5], p=[0.3, 0.25, 0.1, 0.2, 0.15]))
            label = int(birads >= 4)
            for view in ("CC", "MLO"):
                iid = f"{sid}_{lat}_{view}"
                w, h = size
                body, box = _blob_image(rng, h, w, label, border=6)
                (img_root / sid).mkdir(parents=True, exist_ok=True)
                Image.fromarray(body).save(img_root / sid / f"{iid}.png")
                breast.append(dict(study_id=sid, series_id="s", image_id=iid, laterality=lat, view_position=view, height=h, width=w,
                                   breast_birads=f"BI-RADS {birads}", breast_density="DENSITY B", split=split))
                if label:
                    findings.append(dict(study_id=sid, image_id=iid, laterality=lat, view_position=view, finding_categories="['Mass']",
                                         finding_birads=f"BI-RADS {birads}", xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3]))
    pd.DataFrame(breast).to_csv(ann_root / "breast-level_annotations.csv", index=False)
    pd.DataFrame(findings).to_csv(ann_root / "finding_annotations.csv", index=False)


def make_fake_busi(raw_root: Path, n_benign: int = 24, n_malignant: int = 14, n_normal: int = 8, seed: int = 1, with_traps: bool = True) -> None:
    """BUSI layout (Dataset_BUSI_with_GT/<class>/<class> (i).png + _mask.png). Optionally plants the known BUSI defects:
    an exact duplicate, a duplicate with a conflicting label, a coloured overlay, and a second mask file."""
    rng = np.random.default_rng(seed)
    root = Path(raw_root) / "busi" / "Dataset_BUSI_with_GT"
    arrays: dict[str, np.ndarray] = {}
    for cls, n in (("benign", n_benign), ("malignant", n_malignant), ("normal", n_normal)):
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            lab = int(cls == "malignant")
            body, box = _blob_image(rng, 100, 120, lab)
            if cls == "normal":
                body = rng.normal(70, 12, (100, 120)).clip(0, 255).astype(np.uint8)
            arrays[f"{cls} ({i})"] = body
            Image.fromarray(body).save(d / f"{cls} ({i}).png")
            if cls != "normal":
                m = np.zeros((100, 120), np.uint8)
                m[box[1] : box[3], box[0] : box[2]] = 255
                Image.fromarray(m).save(d / f"{cls} ({i})_mask.png")
    if with_traps:
        Image.fromarray(arrays["benign (1)"]).save(root / "benign" / f"benign ({n_benign + 1}).png")           # exact duplicate, same label
        (root / "benign" / f"benign ({n_benign + 1})_mask.png").write_bytes((root / "benign" / "benign (1)_mask.png").read_bytes())
        Image.fromarray(arrays["malignant (1)"]).save(root / "benign" / f"benign ({n_benign + 2}).png")        # duplicate, CONFLICTING label
        rgb = np.stack([arrays["benign (2)"]] * 3, axis=-1).copy()
        rgb[10:20, 10:30] = (255, 0, 0)                                                                       # colour overlay
        Image.fromarray(rgb).save(root / "benign" / "benign (2).png")
        Image.fromarray(np.zeros((100, 120), np.uint8)).save(root / "benign" / "benign (3)_mask_1.png")       # second mask file


def make_fake_extra_us(raw_root: Path, source: str = "extra_us", n: int = 20, seed: int = 2) -> None:
    """A second ultrasound source in plain class-folder layout (stands in for BUS-BRA etc.)."""
    rng = np.random.default_rng(seed)
    for cls in ("benign", "malignant"):
        d = Path(raw_root) / source / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            body, _ = _blob_image(rng, 90, 110, int(cls == "malignant"))
            Image.fromarray(body).save(d / f"{i}.png")


def make_fake_bus_bra(raw_root: Path, n_patients: int = 16, seed: int = 4) -> None:
    """BUS-BRA layout as observed on the first real run: bus_75/BUSBRA/{Images/bus_0001-l.png, Masks/mask_0001-l.png, bus_data.csv}
    (one row per breast; the left and right breast of a patient share ``Case``)."""
    rng = np.random.default_rng(seed)
    base = Path(raw_root) / "bus_bra" / "BUSBRA"
    (base / "Images").mkdir(parents=True, exist_ok=True)
    (base / "Masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for case in range(1, n_patients + 1):
        for side in ("l", "r"):
            lab = int(rng.random() < 0.35)
            body, box = _blob_image(rng, 100, 120, lab)
            i = f"bus_{case:04d}-{side}"
            Image.fromarray(body).save(base / "Images" / f"{i}.png")
            m = np.zeros((100, 120), np.uint8)
            m[box[1] : box[3], box[0] : box[2]] = 255
            Image.fromarray(m).save(base / "Masks" / f"mask_{case:04d}-{side}.png")
            rows.append(dict(ID=i, Case=case, Histology="x", Pathology="malignant" if lab else "benign", BIRADS=4 if lab else 3, Device="d",
                             Width=120, Height=100, Side="left" if side == "l" else "right", BBOX=str(list(box))))
    pd.DataFrame(rows).to_csv(base / "bus_data.csv", index=False)
    pd.DataFrame(rows)[["ID", "Pathology"]].assign(kFold=0).to_csv(base / "5-fold-cv.csv", index=False)


def make_fake_breast_lesions(raw_root: Path, n_benign: int = 12, n_malignant: int = 8, n_normal: int = 2, seed: int = 5) -> None:
    """BrEaST-Lesions-USG layout: a flat folder of caseNNN.png / caseNNN_tumor.png and a clinical spreadsheet whose first sheet has
    CaseID, Image_filename, Mask_tumor_filename, Mask_other_filename, ..., Verification, Diagnosis, Classification (+ a data-dictionary sheet)."""
    rng = np.random.default_rng(seed)
    root = Path(raw_root) / "breast_lesions_usg"
    d = root / "BrEaST-Lesions_USG-images_and_masks"
    d.mkdir(parents=True, exist_ok=True)
    rows, k = [], 0
    for cls, n in (("benign", n_benign), ("malignant", n_malignant), ("normal", n_normal)):
        for _ in range(n):
            k += 1
            body, box = _blob_image(rng, 100, 120, int(cls == "malignant"))
            Image.fromarray(body).save(d / f"case{k:03d}.png")
            mask = ""
            if cls != "normal":
                m = np.zeros((100, 120), np.uint8)
                m[box[1] : box[3], box[0] : box[2]] = 255
                Image.fromarray(m).save(d / f"case{k:03d}_tumor.png")
                mask = f"case{k:03d}_tumor.png"
            rows.append(dict(CaseID=k, Image_filename=f"case{k:03d}.png", Mask_tumor_filename=mask or np.nan, Mask_other_filename=np.nan, Pixel_size=0.007, Age=50,
                             Verification="not applicable" if cls == "normal" else ("confirmed by biopsy" if k % 3 else "confirmed by follow-up care"),
                             Diagnosis="not applicable", Classification=cls))
    with pd.ExcelWriter(root / "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx") as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="BrEaST-Lesions-USG clinical dat", index=False)
        pd.DataFrame(dict(type=["int", "str"], description=["id", "file"])).to_excel(xw, sheet_name="Data Dictionary", index=False)


def make_fake_real_pairs(out_root: Path, n_cases: int = 12, seed: int = 3) -> Path:
    """Stand-in for a processed real mammogram+ultrasound pair set (KAU-BCMD-like). Writes images + pairs_manifest.csv."""
    rng = np.random.default_rng(seed)
    out_root = Path(out_root)
    rows = []
    for c in range(n_cases):
        lab = int(c % 2)
        for tag, (h, w) in (("m", (144, 96)), ("u", (90, 110))):
            body, _ = _blob_image(rng, h, w, lab)
            p = out_root / "images" / f"case{c:03d}_{tag}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(body).save(p)
        rows.append(dict(pair_id=f"case{c:03d}", case_id=f"case{c:03d}", path_a=f"images/case{c:03d}_m.png", path_b=f"images/case{c:03d}_u.png", label=lab))
    path = out_root / "pairs_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

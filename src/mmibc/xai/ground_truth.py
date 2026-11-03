"""Ground-truth lesion masks in MODEL-INPUT space.

VinDr: finding boxes (stored-image pixels, from findings.csv) rasterised through the same letterbox the model sees.
Ultrasound: lesion masks (BUSI etc.) letterboxed identically. Images without annotation return None, so they are
excluded from localisation metrics rather than counted as failures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from mmibc.data.preprocess import letterbox_params, letterbox_square


def box_mask(boxes: np.ndarray, proc_w: int, proc_h: int, size: int) -> np.ndarray:
    aff = letterbox_params(proc_w, proc_h, size)
    m = np.zeros((size, size), bool)
    for x0, y0, x1, y1 in np.asarray(boxes, float).reshape(-1, 4):
        X0, Y0, X1, Y1 = aff.apply_box((x0, y0, x1, y1))
        m[max(0, int(np.floor(Y0))): min(size, int(np.ceil(Y1))), max(0, int(np.floor(X0))): min(size, int(np.ceil(X1)))] = True
    return m


def file_mask(path: Path, size: int) -> np.ndarray:
    img, _ = letterbox_square(Image.open(path).convert("L"), size)
    return np.asarray(img) > 127


class GroundTruth:
    def __init__(self, size: int, vindr: pd.DataFrame | None = None, findings: pd.DataFrame | None = None, vindr_root: Path | None = None, us: pd.DataFrame | None = None, us_root: Path | None = None):
        self.size, self.vindr_root, self.us_root = size, vindr_root, us_root
        self.boxes = {} if findings is None else {k: g[["bx0", "by0", "bx1", "by1"]].to_numpy() for k, g in findings.groupby("image_id")}
        self.dims = {} if vindr is None else dict(zip(vindr["uid"], zip(vindr["proc_w"], vindr["proc_h"])))
        self.us_masks = {} if us is None or "mask_path" not in us else {u: m for u, m in zip(us["uid"], us["mask_path"].fillna("")) if m}

    def mask(self, uid: str) -> np.ndarray | None:
        if uid in self.boxes and uid in self.dims:
            w, h = self.dims[uid]
            m = box_mask(self.boxes[uid], int(w), int(h), self.size)
            return m if m.any() else None
        if uid in self.us_masks and self.us_root is not None:
            m = file_mask(Path(self.us_root) / self.us_masks[uid], self.size)
            return m if m.any() else None
        return None

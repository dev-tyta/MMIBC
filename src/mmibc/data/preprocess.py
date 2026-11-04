"""Image geometry helpers: dtype conversion, breast cropping, downscaling and letterboxing.

Geometry is tracked explicitly so lesion boxes / masks can be mapped into model-input space exactly
(needed for quantitative XAI: pointing game, IoU).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def to_uint8(arr: np.ndarray, percentiles: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """Window a 16-bit / float image into uint8. uint8 input is returned unchanged."""
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, percentiles)
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def load_gray_uint8(path: str | Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode.startswith("I"):  # 16-bit / 32-bit integer PNGs
        return to_uint8(np.asarray(img))
    return np.asarray(img.convert("L"))


def breast_foreground_box(gray: np.ndarray, thresh: int = 10, margin_frac: float = 0.01, min_line_frac: float = 0.002) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of the non-black region. Falls back to the full frame if nothing is found."""
    h, w = gray.shape
    mask = gray > thresh
    rows = np.flatnonzero(mask.sum(axis=1) > min_line_frac * w)
    cols = np.flatnonzero(mask.sum(axis=0) > min_line_frac * h)
    if rows.size == 0 or cols.size == 0:
        return (0, 0, w, h)
    mx, my = int(margin_frac * w), int(margin_frac * h)
    return (max(0, int(cols[0]) - mx), max(0, int(rows[0]) - my), min(w, int(cols[-1]) + 1 + mx), min(h, int(rows[-1]) + 1 + my))


def downscale_long_side(img: Image.Image, long_side: int) -> tuple[Image.Image, float]:
    """Downscale (never upscale) so the longest side is <= long_side. Returns (image, scale)."""
    scale = min(1.0, long_side / max(img.size))
    if scale < 1.0:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    return img, scale


@dataclass(frozen=True)
class Affine:
    """model_xy = scale * processed_xy + (pad_x, pad_y)   (isotropic scale + translation)."""

    scale: float
    pad_x: float
    pad_y: float

    def apply_box(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = box
        return (x0 * self.scale + self.pad_x, y0 * self.scale + self.pad_y, x1 * self.scale + self.pad_x, y1 * self.scale + self.pad_y)


def as_hw(size) -> tuple[int, int]:
    """Model-input size as (height, width): an int means a square, a 2-sequence is (height, width)."""
    if isinstance(size, (int, float)):
        return int(size), int(size)
    h, w = (int(v) for v in size)
    return h, w


def letterbox_params(width: int, height: int, size) -> Affine:
    """Geometry of resizing a (width x height) image into the ``size`` canvas (int -> square, (H, W) -> rectangle), aspect preserved."""
    H, W = as_hw(size)
    s = min(W / width, H / height)
    nw, nh = max(1, round(width * s)), max(1, round(height * s))
    return Affine(s, (W - nw) // 2, (H - nh) // 2)


def letterbox(img: Image.Image, size, fill: int = 0) -> tuple[Image.Image, Affine]:
    """Resize preserving aspect ratio, then centre-pad to the ``size`` canvas (no distortion). ``size``: int (square) or (H, W)."""
    H, W = as_hw(size)
    aff = letterbox_params(img.width, img.height, (H, W))
    nw, nh = max(1, round(img.width * aff.scale)), max(1, round(img.height * aff.scale))
    canvas = Image.new(img.mode, (W, H), fill)
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (int(aff.pad_x), int(aff.pad_y)))
    return canvas, aff


letterbox_square = letterbox  # historical name; accepts rectangles too


def orig_box_to_processed(box: tuple[float, float, float, float], crop_xy: tuple[float, float], scale: float) -> tuple[float, float, float, float]:
    """Map a box from original-image pixels to the stored (cropped + downscaled) image."""
    x0, y0, x1, y1 = box
    cx, cy = crop_xy
    return ((x0 - cx) * scale, (y0 - cy) * scale, (x1 - cx) * scale, (y1 - cy) * scale)


def process_mammogram(src: str | Path, dst: str | Path, long_side: int, crop: bool = True) -> dict:
    gray = load_gray_uint8(src)
    h, w = gray.shape
    x0, y0, x1, y1 = breast_foreground_box(gray) if crop else (0, 0, w, h)
    img = Image.fromarray(gray[y0:y1, x0:x1])
    img, scale = downscale_long_side(img, long_side)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)
    return dict(orig_w=w, orig_h=h, crop_x0=x0, crop_y0=y0, crop_x1=x1, crop_y1=y1, scale=scale, proc_w=img.width, proc_h=img.height)

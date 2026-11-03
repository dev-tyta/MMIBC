"""Dataset integrity: near-duplicate detection, colour-overlay (Doppler / calipers) detection, leakage assertions."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from PIL import Image

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- hashing
def dhash_bits(img: Image.Image, size: int = 16) -> np.ndarray:
    """Difference hash: size*size bits, robust to rescaling / mild re-encoding."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).ravel()


def bits_to_hex(bits: np.ndarray) -> str:
    return np.packbits(bits.astype(np.uint8)).tobytes().hex()


def hex_to_bits(h: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)).astype(bool)


def near_duplicate_groups(hex_hashes: list[str], max_hamming: int = 6, chunk: int = 256) -> np.ndarray:
    """Union-find over pairs with Hamming distance <= max_hamming. Returns a group id per input."""
    n = len(hex_hashes)
    if n == 0:
        return np.zeros(0, dtype=int)
    # float32 (exact for integers < 2**24): int8 would overflow on the 256-bit dot products.
    bits = np.stack([hex_to_bits(h) for h in hex_hashes]).astype(np.float32)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s in range(0, n, chunk):
        blk = bits[s : s + chunk]
        # Hamming distance via dot products: d = |a| + |b| - 2 a.b
        ones = bits.sum(axis=1)
        dist = blk.sum(axis=1)[:, None] + ones[None, :] - 2 * (blk @ bits.T)
        ii, jj = np.nonzero(np.rint(dist) <= max_hamming)
        for i, j in zip(ii + s, jj):
            if i < j:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    return gid


def resolve_duplicates(df: pd.DataFrame, hash_col: str = "dhash", label_col: str = "label", max_hamming: int = 6) -> pd.DataFrame:
    """Adds ``dup_group`` and flags: ``dup_conflict`` (group has >1 label: drop all) and ``dup_extra`` (redundant copy: drop)."""
    out = df.copy()
    out["dup_group"] = near_duplicate_groups(out[hash_col].tolist(), max_hamming)
    nlab = out.groupby("dup_group")[label_col].transform("nunique")
    out["dup_conflict"] = nlab > 1
    out["dup_extra"] = out.duplicated("dup_group", keep="first") & ~out["dup_conflict"]
    log.info(
        "Duplicates: %d images in conflicting-label groups, %d redundant copies (of %d)",
        int(out["dup_conflict"].sum()), int(out["dup_extra"].sum()), len(out),
    )
    return out


# ----------------------------------------------------------------------------- colour overlays
def color_overlay_mask(rgb: np.ndarray, sat_thresh: int = 40) -> np.ndarray:
    """Pixels whose channels differ markedly (Doppler flow, coloured calipers / annotations). rgb: HxWx3 uint8."""
    a = rgb.astype(np.int16)
    spread = a.max(axis=2) - a.min(axis=2)
    return spread > sat_thresh


def color_overlay_fraction(img: Image.Image, sat_thresh: int = 40) -> float:
    if img.mode not in ("RGB", "RGBA"):
        return 0.0
    return float(color_overlay_mask(np.asarray(img.convert("RGB")), sat_thresh).mean())


def inpaint_color_overlay(img: Image.Image, sat_thresh: int = 40, dilate: int = 2) -> Image.Image:
    """Grayscale image with coloured overlay pixels inpainted. Uses OpenCV when available, else a median fill."""
    rgb = np.asarray(img.convert("RGB"))
    mask = color_overlay_mask(rgb, sat_thresh)
    gray = np.asarray(img.convert("L"))
    if not mask.any():
        return Image.fromarray(gray)
    try:
        import cv2

        m = mask.astype(np.uint8) * 255
        if dilate:
            m = cv2.dilate(m, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
        return Image.fromarray(cv2.inpaint(gray, m, 3, cv2.INPAINT_TELEA))
    except ImportError:
        filled = gray.copy()
        filled[mask] = int(np.median(gray[~mask])) if (~mask).any() else 0
        return Image.fromarray(filled)


# ----------------------------------------------------------------------------- leakage
def leakage_report(df: pd.DataFrame, split_col: str = "split", keys: tuple[str, ...] = ("uid", "group")) -> dict[str, int]:
    """Count identifiers shared between different splits. A clean split table returns all zeros."""
    report: dict[str, int] = {}
    splits = sorted(df[split_col].unique())
    for key in keys:
        if key not in df.columns:
            continue
        shared = 0
        for i, a in enumerate(splits):
            for b in splits[i + 1 :]:
                shared += len(set(df.loc[df[split_col] == a, key]) & set(df.loc[df[split_col] == b, key]))
        report[key] = shared
    return report


def assert_no_leakage(df: pd.DataFrame, split_col: str = "split", keys: tuple[str, ...] = ("uid", "group")) -> None:
    rep = leakage_report(df, split_col, keys)
    bad = {k: v for k, v in rep.items() if v}
    if bad:
        raise AssertionError(f"Split leakage detected (identifiers shared across splits): {bad}")

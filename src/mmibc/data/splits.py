"""Leakage-safe splitting. All splits are grouped (study / patient / near-duplicate cluster) and stratified by label."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from mmibc.data.audit import assert_no_leakage, resolve_duplicates

log = logging.getLogger(__name__)


def connected_group_ids(df: pd.DataFrame, key_cols: list[str]) -> np.ndarray:
    """Rows sharing ANY non-null value in ANY of ``key_cols`` end up in the same group (transitively)."""
    n = len(df)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for col in key_cols:
        first_seen: dict = {}
        for i, v in enumerate(df[col].tolist()):
            if pd.isna(v):  # None, NaN and pandas' NA must never link unrelated rows
                continue
            if v in first_seen:
                ri, rj = find(i), find(first_seen[v])
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
            else:
                first_seen[v] = i
    _, gid = np.unique([find(i) for i in range(n)], return_inverse=True)
    return gid


def _carve(df: pd.DataFrame, group_col: str, label_col: str, frac: float, seed: int) -> np.ndarray:
    """Positional indices of a ~``frac`` hold-out chosen by whole groups, stratified by label."""
    n_splits = max(2, round(1.0 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, hold = next(sgkf.split(df, df[label_col], df[group_col]))
    return hold


def group_stratified_split(df: pd.DataFrame, group_col: str, label_col: str, val_frac: float, test_frac: float, seed: int) -> pd.Series:
    """Return a Series of 'train' / 'val' / 'test'. No group appears in more than one split."""
    split = pd.Series("train", index=df.index, dtype="object")
    if test_frac > 0:
        test_pos = _carve(df, group_col, label_col, test_frac, seed)
        split.iloc[test_pos] = "test"
    if val_frac > 0:
        rest = df[split == "train"]
        val_pos = _carve(rest, group_col, label_col, val_frac / (1.0 - test_frac), seed + 1)
        split.loc[rest.index[val_pos]] = "val"
    return split


def assign_vindr_splits(manifest: pd.DataFrame, val_fraction: float, seed: int) -> pd.DataFrame:
    """Official VinDr test split is kept intact; validation is carved from the official training split by *study*."""
    df = manifest.copy()
    df["group"] = "vindr:" + df["study_id"].astype(str)
    df["source"] = "vindr"
    df["split"] = df["official_split"]
    train = df[df["official_split"] == "train"]
    val_pos = _carve(train, "group", "label", val_fraction, seed)
    df.loc[train.index[val_pos], "split"] = "val"
    assert_no_leakage(df, keys=("uid", "group"))
    return df


def assign_cmmd_splits(manifest: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int, subsets: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    """CMMD has no official split: carve train / val / test by PATIENT (both breasts of a patient stay together), stratified by label and subset.

    ``subsets`` (e.g. ``["D1"]``) restricts the data before splitting. ``source`` is ``cmmd_d1`` / ``cmmd_d2`` so the merge audit's
    source-confounding check sees the subsets, and ``stratum`` carries the subset into the pair tables for per-subset evaluation.
    """
    df = manifest.copy()
    if subsets:
        keep = {str(s).upper() for s in subsets}
        df = df[df["subset"].astype(str).str.upper().isin(keep)]
        if df.empty:
            raise ValueError(f"data.cmmd.subsets={sorted(keep)} matches nothing; the prepared CMMD data has subsets {sorted(manifest['subset'].astype(str).unique())}")
    df = df.reset_index(drop=True)
    df["group"] = "cmmd:" + df["study_id"].astype(str)
    df["source"] = "cmmd_" + df["subset"].astype(str).str.lower()
    df["stratum"] = df["subset"].astype(str)
    df["_strat"] = df["label"].astype(int).astype(str) + df["subset"].astype(str)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class")  # a rare subset x label cell is fine; it is still split by whole patients
        df["split"] = group_stratified_split(df, "group", "_strat", val_fraction, test_fraction, seed)
    df = df.drop(columns=["_strat"])
    assert_no_leakage(df, keys=("uid", "group"))
    return df


def assign_us_splits(
    manifest: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int,
    dedupe: bool = True, dup_hamming: int = 6, external_source: str | None = None,
) -> pd.DataFrame:
    """Unified ultrasound split.

    * near-duplicates (across ALL sources) are clustered by perceptual hash; conflicting-label clusters are dropped,
      redundant copies are dropped (``dedupe=True``);
    * the split group is the connected component of {patient_id, duplicate cluster};
    * ``external_source`` (e.g. a different institution) is held out entirely as split 'ext_test'.
    """
    df = manifest.copy()
    df = resolve_duplicates(df, "dhash", "label", dup_hamming)
    if dedupe:
        df = df[~(df["dup_conflict"] | df["dup_extra"])].reset_index(drop=True)
    pid = df["source"].astype(str) + ":" + df["patient_id"].astype("string")
    df["patient_key"] = pid.where(df["patient_id"].notna(), None)
    df["group"] = ["us:" + str(g) for g in connected_group_ids(df, ["patient_key", "dup_group"])]
    df["split"] = "train"
    if external_source is not None:
        ext = df["source"] == external_source
        df.loc[ext, "split"] = "ext_test"
    internal = df[df["split"] != "ext_test"]
    df.loc[internal.index, "split"] = group_stratified_split(internal, "group", "label", val_fraction, test_fraction, seed)
    assert_no_leakage(df, keys=("uid", "group"))
    return df.drop(columns=["patient_key"])


def add_size_quantiles(df: pd.DataFrame, size_col: str = "lesion_area_frac", by: tuple[str, ...] = ("split", "label")) -> pd.DataFrame:
    """Percentile rank (0..1) of lesion size within (split, label); NaN where lesion size is unknown or zero."""
    df = df.copy()
    sized = df[size_col].where(df[size_col] > 0)
    df["size_q"] = sized.groupby([df[c] for c in by]).rank(pct=True)
    return df

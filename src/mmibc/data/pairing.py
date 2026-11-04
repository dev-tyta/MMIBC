"""Pairing, redesigned.

Evidence tiers (strongest first):

  Tier 1 - REAL same-patient pairs
      1a. ``native_multiview``  CC <-> MLO of the SAME breast, SAME study (VinDr-Mammo). Thousands of pairs.
      1b. ``real_cross_modal``  mammogram <-> ultrasound of the SAME patient (KAU-BCMD subset). Too small to train
          on; used as the external real-pair test of whether anything learned from synthetic pairs transfers.
  Tier 2 - SYNTHETIC cross-modal pairs (VinDr x BUSI/BUS-BRA/...), always labelled as synthetic:
      ``label_matched``      random partner with the same label
      ``size_conditioned``   same label AND matched lesion-size percentile (the only attribute both modalities expose)
  Tier 3 - CONTROLS, which turn the synthetic-pairing limitation into a measurement:
      ``random``             partner ignores the label (target = mammogram label)
      ``shuffled_partner``   label-matched pairs whose partners are then permuted

Rules every synthetic pairing obeys: both images come from the SAME split (no leakage); training pairs use
sampling with replacement and are re-drawn each epoch (no truncation to the minority class, so the whole pool is
used); validation/test pairs are 1-to-1 without replacement, drawn once with a fixed seed and saved, so every
model is evaluated on exactly the same pairs and test samples are independent.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SYNTHETIC_MODES = ("label_matched", "size_conditioned", "random", "shuffled_partner")
PAIR_COLS = ["pair_id", "uid_a", "uid_b", "path_a", "path_b", "label", "label_a", "label_b", "split", "pair_type", "group_a", "group_b"]


# ----------------------------------------------------------------------------- Tier 1a
def make_native_multiview_pairs(table: pd.DataFrame) -> pd.DataFrame:
    """CC (stream a) <-> MLO (stream b) of the same (study, laterality). ``table`` needs uid, study_id, laterality, view, proc_path, label, split."""
    t = table[table["view"].isin(["CC", "MLO"])]
    first = t.sort_values("uid").drop_duplicates(["study_id", "laterality", "view"], keep="first")
    n_dropped = len(t) - len(first)
    if n_dropped:
        log.warning("%d extra images per (study, breast, view) ignored (kept first by uid)", n_dropped)
    cc = first[first["view"] == "CC"].add_suffix("_a")
    mlo = first[first["view"] == "MLO"].add_suffix("_b")
    m = cc.merge(mlo, left_on=["study_id_a", "laterality_a"], right_on=["study_id_b", "laterality_b"], how="inner")
    if not (m["label_a"] == m["label_b"]).all():
        raise AssertionError("CC and MLO of the same breast carry different labels: breast-level labelling is inconsistent")
    if not (m["split_a"] == m["split_b"]).all():
        raise AssertionError("CC and MLO of the same breast fall in different splits")
    out = pd.DataFrame(
        dict(
            pair_id=m["study_id_a"] + "_" + m["laterality_a"], uid_a=m["uid_a"], uid_b=m["uid_b"], path_a=m["proc_path_a"], path_b=m["proc_path_b"],
            label=m["label_a"].astype(int), label_a=m["label_a"].astype(int), label_b=m["label_b"].astype(int), split=m["split_a"],
            pair_type="native_multiview", group_a="vindr:" + m["study_id_a"], group_b="vindr:" + m["study_id_a"],
        )
    )
    return out[PAIR_COLS].reset_index(drop=True)


# ----------------------------------------------------------------------------- Tier 1b
def load_real_pairs(csv_path, root_a: str = "", root_b: str = "") -> pd.DataFrame:
    """Real same-patient mammogram<->ultrasound pairs from a manifest CSV.

    Required columns: pair_id, case_id, path_a (mammogram), path_b (ultrasound), label.
    Optional: split. Paths are relative to the processed root of each modality.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in ["pair_id", "case_id", "path_a", "path_b", "label"] if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} lacks columns {missing}")
    df["uid_a"] = df.get("uid_a", df["pair_id"].astype(str) + "_a")
    df["uid_b"] = df.get("uid_b", df["pair_id"].astype(str) + "_b")
    df["label"] = df["label"].astype(int)
    df["label_a"], df["label_b"] = df["label"], df["label"]
    df["group_a"] = df["group_b"] = "real:" + df["case_id"].astype(str)
    df["pair_type"] = "real_cross_modal"
    if "split" not in df.columns:
        df["split"] = "ext_test"
    return df[PAIR_COLS]


def assign_group_folds(df: pd.DataFrame, group_col: str = "group_a", label_col: str = "label", n_folds: int = 5, seed: int = 0) -> pd.Series:
    """Patient-level stratified folds for small real-pair sets (cross-validation of a light fusion head)."""
    from sklearn.model_selection import StratifiedGroupKFold

    fold = pd.Series(-1, index=df.index)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sgkf.split(df, df[label_col], df[group_col])):
        fold.iloc[te] = k
    return fold


# ----------------------------------------------------------------------------- Tier 2 / 3
def _nearest_by_rank(q: np.ndarray, pool_sorted_idx: np.ndarray, rng: np.random.Generator, jitter: int) -> np.ndarray:
    pos = np.rint(q * (len(pool_sorted_idx) - 1)).astype(int)
    if jitter:
        pos = pos + rng.integers(-jitter, jitter + 1, size=len(pos))
    return pool_sorted_idx[np.clip(pos, 0, len(pool_sorted_idx) - 1)]


def make_synthetic_pairs(a_df: pd.DataFrame, b_df: pd.DataFrame, split: str, mode: str, seed: int, replace: bool) -> pd.DataFrame:
    """Pair stream-a images (mammograms) with stream-b images (ultrasounds) WITHIN ``split``.

    replace=True  : every a-image is an anchor once; partners drawn with replacement (training).
    replace=False : 1-to-1 without replacement, n = min(count_a, count_b) per label (validation / test).
    The target ``label`` is the shared label (matched modes) or the mammogram label (random / shuffled_partner).
    """
    if mode not in SYNTHETIC_MODES:
        raise ValueError(f"mode must be one of {SYNTHETIC_MODES}")
    a = a_df[a_df["split"] == split].reset_index(drop=True)
    b = b_df[b_df["split"] == split].reset_index(drop=True)
    if a.empty or b.empty:
        raise ValueError(f"No images in split '{split}' (a={len(a)}, b={len(b)})")
    rng = np.random.default_rng(seed)
    has_q = "size_q" in a.columns and "size_q" in b.columns

    def matched(use_size: bool) -> tuple[np.ndarray, np.ndarray]:
        ia_all, ib_all = [], []
        for lab in sorted(set(a["label"]) & set(b["label"])):
            ia, ib = np.flatnonzero(a["label"] == lab), np.flatnonzero(b["label"] == lab)
            if replace:
                sel_a = ia
                sel_b = rng.choice(ib, size=len(ia), replace=True)
                if use_size and has_q:
                    pool = ib[~b.loc[ib, "size_q"].isna().to_numpy()]
                    pool = pool[np.argsort(b.loc[pool, "size_q"].to_numpy(), kind="stable")]
                    q = a.loc[ia, "size_q"].to_numpy(dtype=float)
                    ok = ~np.isnan(q) & (len(pool) > 0)
                    if ok.any():
                        sel_b[ok] = _nearest_by_rank(q[ok], pool, rng, jitter=max(1, int(0.02 * len(pool))))
            else:
                n = min(len(ia), len(ib))
                if use_size and has_q:
                    # Monotone coupling: pair equally spaced quantiles of both size distributions.
                    sa = ia[np.argsort(a.loc[ia, "size_q"].fillna(0.5).to_numpy(), kind="stable")]
                    sb = ib[np.argsort(b.loc[ib, "size_q"].fillna(0.5).to_numpy(), kind="stable")]
                    sel_a = sa[np.rint(np.linspace(0, len(sa) - 1, n)).astype(int)]
                    sel_b = sb[np.rint(np.linspace(0, len(sb) - 1, n)).astype(int)]
                else:
                    sel_a, sel_b = rng.choice(ia, n, replace=False), rng.choice(ib, n, replace=False)
            ia_all.append(sel_a)
            ib_all.append(sel_b)
        return np.concatenate(ia_all), np.concatenate(ib_all)

    if mode in ("label_matched", "size_conditioned"):
        sel_a, sel_b = matched(use_size=(mode == "size_conditioned"))
    elif mode == "shuffled_partner":
        sel_a, sel_b = matched(use_size=False)
        sel_b = sel_b[rng.permutation(len(sel_b))]
    else:  # random
        if replace:
            sel_a, sel_b = np.arange(len(a)), rng.integers(0, len(b), size=len(a))
        else:
            n = min(len(a), len(b))
            sel_a, sel_b = rng.choice(len(a), n, replace=False), rng.choice(len(b), n, replace=False)

    A, B = a.iloc[sel_a].reset_index(drop=True), b.iloc[sel_b].reset_index(drop=True)
    shared_label = mode in ("label_matched", "size_conditioned")
    out = pd.DataFrame(
        dict(
            pair_id=[f"{x}|{y}" for x, y in zip(A["uid"], B["uid"])], uid_a=A["uid"], uid_b=B["uid"], path_a=A["proc_path"], path_b=B["proc_path"],
            label=A["label"].astype(int), label_a=A["label"].astype(int), label_b=B["label"].astype(int), split=split, pair_type=mode,
            group_a=A["group"], group_b=B["group"],
        )
    )
    if shared_label:
        assert (out["label_a"] == out["label_b"]).all()
    return out[PAIR_COLS].sample(frac=1.0, random_state=seed).reset_index(drop=True)


class SyntheticResampler:
    """Picklable callable: epoch -> freshly drawn training pairs (used via ``PairDataset.set_epoch``)."""

    def __init__(self, a_df: pd.DataFrame, b_df: pd.DataFrame, split: str, mode: str, base_seed: int, replace: bool = True):
        self.a_df, self.b_df, self.split, self.mode, self.base_seed, self.replace = a_df, b_df, split, mode, base_seed, replace

    def __call__(self, epoch: int) -> pd.DataFrame:
        return make_synthetic_pairs(self.a_df, self.b_df, self.split, self.mode, self.base_seed + epoch, self.replace)


def assert_pairs_clean(pairs: pd.DataFrame, a_df: pd.DataFrame | None = None, b_df: pd.DataFrame | None = None) -> None:
    """Hard guarantees: single split per table; for synthetic pairs both partners' own split equals the pair's split."""
    if pairs["split"].nunique() != 1:
        raise AssertionError(f"pair table mixes splits: {pairs['split'].unique()}")
    split = pairs["split"].iloc[0]
    for df, col in ((a_df, "uid_a"), (b_df, "uid_b")):
        if df is not None:
            own = df.set_index("uid")["split"].reindex(pairs[col])
            if own.isna().any() or (own != split).any():
                raise AssertionError(f"{col}: found images whose own split differs from the pair split '{split}'")

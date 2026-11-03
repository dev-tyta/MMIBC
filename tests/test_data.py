from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mmibc.data import audit, preprocess
from mmibc.data.pairing import (
    SyntheticResampler, assert_pairs_clean, assign_group_folds, load_real_pairs, make_native_multiview_pairs, make_synthetic_pairs,
)
from mmibc.data.splits import add_size_quantiles, assign_us_splits, assign_vindr_splits, connected_group_ids, group_stratified_split
from mmibc.data.synthetic import make_fake_real_pairs


# ------------------------------------------------------------------ preprocess
def test_breast_crop_and_uint8():
    g = np.zeros((100, 80), np.uint8)
    g[20:70, 10:50] = 120
    x0, y0, x1, y1 = preprocess.breast_foreground_box(g, margin_frac=0)
    assert (x0, y0, x1, y1) == (10, 20, 50, 70)
    assert preprocess.to_uint8(np.arange(0, 65535, 100, dtype=np.uint16)).dtype == np.uint8
    assert preprocess.breast_foreground_box(np.zeros((10, 10), np.uint8)) == (0, 0, 10, 10)


def test_letterbox_maps_boxes_exactly():
    img = Image.new("L", (40, 80), 255)
    out, aff = preprocess.letterbox_square(img, 64)
    assert out.size == (64, 64)
    x0, y0, x1, y1 = aff.apply_box((0, 0, 40, 80))
    arr = np.asarray(out)
    assert arr[int(y0) + 2, int(x0) + 2] > 200 and arr[2, 2] < 50  # content inside mapped box, padding outside
    assert aff == preprocess.letterbox_params(40, 80, 64)


# ------------------------------------------------------------------ vindr
def test_vindr_prepare_and_boxes(vindr_manifest):
    out, m = vindr_manifest
    assert len(m) == 40 * 4
    assert set(m["label"].unique()) == {0, 1}
    assert (out / m["proc_path"].iloc[0]).exists()
    f = pd.read_csv(out / "findings.csv")
    assert len(f) > 0
    # every mapped box must lie (almost) inside its processed image
    j = f.merge(m[["uid", "proc_w", "proc_h"]], left_on="image_id", right_on="uid")
    assert (j["bx0"] >= -1).all() and (j["by0"] >= -1).all() and (j["bx1"] <= j["proc_w"] + 1).all() and (j["by1"] <= j["proc_h"] + 1).all()
    assert (m.loc[m["label"] == 1, "lesion_area_frac"] > 0).all()


def test_vindr_resume_is_idempotent(raw_root, vindr_manifest):
    from mmibc.data.vindr import prepare_vindr

    out, m = vindr_manifest
    m2 = prepare_vindr(raw_root / "vindr_images", raw_root / "vindr_annotations", out, long_side=112, workers=2)
    assert len(m2) == len(m)


def test_vindr_splits_and_native_pairs(vindr_manifest):
    _, m = vindr_manifest
    t = assign_vindr_splits(m, val_fraction=0.2, seed=0)
    assert set(t["split"]) == {"train", "val", "test"}
    assert t.groupby("group")["split"].nunique().max() == 1            # a study never straddles splits
    assert (t.loc[t["official_split"] == "test", "split"] == "test").all()
    pairs = make_native_multiview_pairs(t)
    assert len(pairs) == 80                                              # 40 studies x 2 breasts
    info = t.set_index("uid")
    for col in ("view", "laterality", "study_id"):
        pairs[col + "_a"] = pairs["uid_a"].map(info[col])
        pairs[col + "_b"] = pairs["uid_b"].map(info[col])
    assert (pairs["view_a"] == "CC").all() and (pairs["view_b"] == "MLO").all()
    assert (pairs["study_id_a"] == pairs["study_id_b"]).all() and (pairs["laterality_a"] == pairs["laterality_b"]).all()
    for s, p in pairs.groupby("split"):
        assert_pairs_clean(p, t, t)


# ------------------------------------------------------------------ ultrasound
def test_busi_scan_masks_and_label_schemes(raw_root):
    from mmibc.data.ultrasound import apply_label_scheme, scan_busi

    scan = scan_busi(raw_root / "busi")
    assert (scan["class_name"] == "benign").sum() == 24 + 2          # +2 planted duplicates; _mask_1 is NOT an image
    assert not scan["raw_path"].str.contains("_mask").any()
    assert scan.loc[scan["uid"] == "busi_benign_3", "mask_paths"].iloc[0].count("|") == 1   # two mask files merged
    assert set(apply_label_scheme(scan, "binary_no_normal")["class_name"]) == {"benign", "malignant"}
    assert apply_label_scheme(scan, "three_class")["label"].max() == 2
    assert len(apply_label_scheme(scan, "binary_normal_as_benign")) == len(scan)


def test_us_prepare_overlay_and_dedupe(us_manifest):
    out, m = us_manifest
    assert (m["source"].value_counts().to_dict().keys()) == {"busi", "extra_us"}
    ov = m.set_index("uid").loc["busi_benign_2", "overlay_frac"]
    assert ov > 0
    assert Image.open(out / m.set_index("uid").loc["busi_benign_2", "proc_path"]).mode == "L"   # inpainted to grayscale
    busi_benign = m.loc[(m["class_name"] == "benign") & (m["source"] == "busi"), "lesion_area_frac"]
    assert (busi_benign.dropna() > 0).all() and busi_benign.isna().sum() == 1   # only the planted mask-less duplicate lacks a mask
    assert m.loc[m["source"] == "extra_us", "lesion_area_frac"].isna().all()   # sources without masks have unknown lesion size
    t = assign_us_splits(m, val_fraction=0.2, test_fraction=0.2, seed=0, dedupe=True, dup_hamming=6)
    uids = set(t["uid"])
    assert "busi_benign_25" not in uids or "busi_benign_1" not in uids   # exact duplicate pair: at most one kept
    assert "busi_malignant_1" not in uids and "busi_benign_26" not in uids  # conflicting-label duplicates: both dropped
    assert t.groupby("group")["split"].nunique().max() == 1
    ext = assign_us_splits(m, 0.2, 0.2, 0, external_source="extra_us")
    assert (ext.loc[ext["source"] == "extra_us", "split"] == "ext_test").all()
    assert not (ext.loc[ext["source"] != "extra_us", "split"] == "ext_test").any()


def test_dedupe_hashing_and_groups():
    rng = np.random.default_rng(0)
    base = Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8))
    h_same = audit.bits_to_hex(audit.dhash_bits(base))
    h_noisy = audit.bits_to_hex(audit.dhash_bits(Image.fromarray(np.clip(np.asarray(base).astype(int) + rng.integers(-2, 3, (64, 64)), 0, 255).astype(np.uint8))))
    h_other = audit.bits_to_hex(audit.dhash_bits(Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8))))
    g = audit.near_duplicate_groups([h_same, h_noisy, h_other], max_hamming=30)
    assert g[0] == g[1] != g[2]


# ------------------------------------------------------------------ splits
def test_group_split_and_connected_groups():
    df = pd.DataFrame({"g": np.repeat(np.arange(60), 3), "y": np.repeat(np.tile([0, 1], 30), 3)})
    s = group_stratified_split(df, "g", "y", 0.2, 0.2, seed=0)
    assert df.assign(s=s).groupby("g")["s"].nunique().max() == 1
    assert set(s) == {"train", "val", "test"}
    assert 0.12 < (s == "test").mean() < 0.3
    d = pd.DataFrame({"p": ["a", "b", None, "c"], "dup": [1, 1, 2, 3]})
    gid = connected_group_ids(d, ["p", "dup"])
    assert gid[0] == gid[1] and len(set(gid)) == 3


# ------------------------------------------------------------------ pairing
@pytest.fixture()
def pools(vindr_manifest, us_manifest):
    _, vm = vindr_manifest
    _, um = us_manifest
    a = add_size_quantiles(assign_vindr_splits(vm, 0.2, 0))
    b = add_size_quantiles(assign_us_splits(um, 0.2, 0.2, 0))
    return a, b


@pytest.mark.parametrize("mode", ["label_matched", "size_conditioned"])
def test_matched_pairs_are_clean(pools, mode):
    a, b = pools
    for replace in (False, True):
        p = make_synthetic_pairs(a, b, "val", mode, seed=1, replace=replace)
        assert (p["label_a"] == p["label_b"]).all() and (p["label"] == p["label_a"]).all()
        assert_pairs_clean(p, a, b)
        if not replace:
            assert p["uid_a"].is_unique and p["uid_b"].is_unique


def _toy_pool(n, seed, prefix):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(dict(uid=[f"{prefix}{i}" for i in range(n)], label=1, split="train", group=[f"{prefix}g{i}" for i in range(n)],
                           proc_path="x.png", lesion_area_frac=rng.uniform(0.01, 0.3, n)))
    return add_size_quantiles(df)


@pytest.mark.parametrize("replace", [False, True])
def test_size_conditioning_correlates_sizes(replace):
    a, b = _toy_pool(80, 0, "a"), _toy_pool(60, 1, "b")
    sq = {**a.set_index("uid")["size_q"].to_dict(), **b.set_index("uid")["size_q"].to_dict()}
    corr = lambda p: np.corrcoef(p["uid_a"].map(sq), p["uid_b"].map(sq))[0, 1]
    assert corr(make_synthetic_pairs(a, b, "train", "size_conditioned", seed=0, replace=replace)) > 0.9
    assert abs(corr(make_synthetic_pairs(a, b, "train", "label_matched", seed=0, replace=replace))) < 0.5   # label matching alone does NOT couple sizes


def test_controls_break_label_agreement(pools):
    a, b = pools
    for mode in ("random", "shuffled_partner"):
        p = make_synthetic_pairs(a, b, "train", mode, seed=0, replace=False)
        assert (p["label"] == p["label_a"]).all()
        assert (p["label_a"] != p["label_b"]).any()
        assert_pairs_clean(p, a, b)


def test_pairs_never_cross_splits_and_resample_varies(pools):
    a, b = pools
    rs = SyntheticResampler(a, b, "train", "label_matched", base_seed=0, replace=True)
    p0, p1 = rs(0), rs(1)
    assert len(p0) == (a["split"] == "train").sum() - (a.loc[a["split"] == "train", "label"].isin(set(b["label"])) == False).sum()  # noqa: E712
    assert not p0["pair_id"].equals(p1["pair_id"])
    assert rs(0).equals(p0)                                              # deterministic per epoch
    bad = p0.copy()
    bad.loc[bad.index[0], "uid_b"] = b.loc[b["split"] == "test", "uid"].iloc[0]
    with pytest.raises(AssertionError):
        assert_pairs_clean(bad, a, b)


def test_real_pairs_manifest_and_folds(tmp_path):
    csv = make_fake_real_pairs(tmp_path, n_cases=12)
    rp = load_real_pairs(csv)
    assert (rp["pair_type"] == "real_cross_modal").all() and (rp["split"] == "ext_test").all()
    folds = assign_group_folds(rp, n_folds=3, seed=0)
    assert set(folds) == {0, 1, 2}
    with pytest.raises(ValueError):
        pd.DataFrame({"pair_id": [1]}).to_csv(tmp_path / "bad.csv", index=False)
        load_real_pairs(tmp_path / "bad.csv")

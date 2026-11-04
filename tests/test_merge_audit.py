from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mmibc.data.merge_audit import AuditConfig, audit_pairs, audit_ultrasound, audit_vindr
from mmibc.data.pairing import make_native_multiview_pairs, make_synthetic_pairs
from mmibc.data.splits import add_size_quantiles, assign_us_splits, assign_vindr_splits
from mmibc.data.synthetic import make_fake_busi, make_fake_extra_us
from mmibc.data.ultrasound import prepare_ultrasound, scan_busi, scan_class_folders

CFG = AuditConfig(check_files="all")


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw_big")
    make_fake_busi(raw, n_benign=70, n_malignant=50, n_normal=5)
    make_fake_extra_us(raw, n=40)
    out = tmp_path_factory.mktemp("proc_big")
    scan = pd.concat([scan_busi(raw / "busi"), scan_class_folders(raw / "extra_us", "extra_us")], ignore_index=True)
    m = prepare_ultrasound(scan, out, "binary_no_normal", long_side=112, workers=1)
    df = assign_us_splits(m, 0.15, 0.15, seed=0, external_source="extra_us")
    return out, add_size_quantiles(df)


def codes(rep, level):
    return {f.code for f in rep.findings if f.level == level}


def test_clean_merge_passes(merged):
    root, df = merged
    rep = audit_ultrasound(df, root, CFG, external_source="extra_us")
    assert rep.ok, rep.to_markdown()
    assert rep.stats["files_checked"] == len(df)
    assert set(df["split"]) == {"train", "val", "test", "ext_test"}


def test_uid_and_label_mapping_defects(merged):
    root, df = merged
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert "uid_collision" in codes(audit_ultrasound(dup, root, CFG, "extra_us"), "error")
    bad = df.copy()
    i = bad.index[bad["class_name"] == "malignant"][0]
    bad.loc[i, "label"] = 0
    assert "label_mapping" in codes(audit_ultrasound(bad, root, CFG, "extra_us"), "error")


def test_cross_split_duplicate_is_caught(merged):
    root, df = merged
    bad = df.copy()
    tr = bad[bad["split"] == "train"].iloc[0]
    j = bad.index[bad["split"] == "test"][0]
    bad.loc[j, "dhash"] = tr["dhash"]
    bad.loc[j, "label"], bad.loc[j, "class_name"] = tr["label"], tr["class_name"]
    assert "duplicate_leak" in codes(audit_ultrasound(bad, root, CFG, "extra_us"), "error")


def test_external_source_must_stay_external(merged):
    root, df = merged
    bad = df.copy()
    bad.loc[bad.index[bad["split"] == "ext_test"][0], "split"] = "train"
    assert "external_isolation" in codes(audit_ultrasound(bad, root, CFG, "extra_us"), "error")
    bad2 = df.copy()
    bad2.loc[bad2.index[bad2["split"] == "test"][0], "source"] = "extra_us"
    assert "external_isolation" in codes(audit_ultrasound(bad2, root, CFG, "extra_us"), "error")


def test_file_defects(merged, tmp_path):
    root, df = merged
    bad = df.copy()
    bad.loc[bad.index[0], "proc_path"] = "images/does_not_exist.png"
    assert "file_missing" in codes(audit_ultrasound(bad, root, CFG, "extra_us"), "error")
    r = df.iloc[1]
    blank = root / "images" / "blank_test.png"
    Image.fromarray(np.full((int(r.proc_h), int(r.proc_w)), 90, np.uint8)).save(blank)
    bad = df.copy()
    bad.loc[bad.index[1], "proc_path"] = "images/blank_test.png"
    assert "blank_image" in codes(audit_ultrasound(bad, root, CFG, "extra_us"), "error")
    blank.unlink()
    wrong = root / "images" / "wrong_size.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (10, 10), dtype=np.uint8)).save(wrong)
    bad = df.copy()
    bad.loc[bad.index[2], "proc_path"] = "images/wrong_size.png"
    assert "size_mismatch" in codes(audit_ultrasound(bad, root, CFG, "extra_us"), "error")
    wrong.unlink()


def test_empty_class_in_a_split_is_an_error(merged):
    root, df = merged
    bad = df[~((df["split"] == "test") & (df["label"] == 1))]
    assert "empty_class" in codes(audit_ultrasound(bad, root, AuditConfig(check_files="exists"), "extra_us"), "error")


def _hand_table(n=240, seed=0):
    rng = np.random.default_rng(seed)
    src = np.where(np.arange(n) < n // 2, "A", "B")
    return pd.DataFrame(
        dict(uid=[f"u{i}" for i in range(n)], source=src, class_name="benign", label=0, proc_path=[f"i{i}.png" for i in range(n)],
             split=rng.choice(["train", "val", "test"], n, p=[0.7, 0.15, 0.15]), group=[f"g{i}" for i in range(n)],
             proc_w=rng.integers(100, 120, n), proc_h=rng.integers(100, 120, n), overlay_frac=0.0)
    )


def test_source_label_confounding_is_flagged():
    df = _hand_table()
    df["label"] = (df["source"] == "B").astype(int)             # source B is all malignant
    df["class_name"] = np.where(df["label"] == 1, "malignant", "benign")
    rep = audit_ultrasound(df, None, CFG)
    assert "source_prevalence" in codes(rep, "warn")
    assert "metadata_shortcut" in codes(rep, "warn")           # source alone separates the labels perfectly
    assert rep.stats["metadata_only_label_auc"] > 0.9


def test_balanced_sources_are_not_flagged():
    df = _hand_table()
    rng = np.random.default_rng(1)
    df["label"] = rng.integers(0, 2, len(df))
    df["class_name"] = np.where(df["label"] == 1, "malignant", "benign")
    rep = audit_ultrasound(df, None, CFG)
    assert "source_prevalence" not in codes(rep, "warn") and "metadata_shortcut" not in codes(rep, "warn")


# ----------------------------------------------------------------------------- VinDr and pairs
@pytest.fixture(scope="module")
def vindr_split(vindr_manifest):
    root, m = vindr_manifest
    return root, assign_vindr_splits(m, 0.2, seed=0)


def test_vindr_clean_and_defects(vindr_split):
    root, df = vindr_split
    findings = pd.read_csv(root / "findings.csv")
    rep = audit_vindr(df, root, findings, AuditConfig(check_files="all", min_per_class_split=1))
    assert rep.ok, rep.to_markdown()
    moved = df.copy()
    moved.loc[moved.index[moved["official_split"] == "test"][0], "split"] = "train"
    assert "official_test_changed" in codes(audit_vindr(moved, None, None, CFG), "error")
    conflict = df.copy()
    i = conflict.index[0]
    conflict.loc[i, "label"] = 1 - conflict.loc[i, "label"]
    assert "breast_label_conflict" in codes(audit_vindr(conflict, None, None, CFG), "error")
    shifted = findings.copy()
    shifted[["bx0", "bx1"]] += 5000
    assert "box_mapping" in codes(audit_vindr(df, None, shifted, CFG), "error")


def test_pair_audits(vindr_split):
    root, df = vindr_split
    native = make_native_multiview_pairs(df)
    for split in native["split"].unique():
        sub = native[native["split"] == split]
        assert audit_pairs(sub, df, df, frozen=False, matched=True, same_group=True).ok
    # frozen synthetic pairs: build from the same table as both pools just to exercise the checks
    a = df.copy()
    b = df.assign(uid="b_" + df["uid"])
    val = make_synthetic_pairs(a, b, "val", "label_matched", seed=0, replace=False)
    assert audit_pairs(val, a, b, frozen=True, matched=True).ok
    reused = pd.concat([val, val.iloc[[0]].assign(pair_id="dup")], ignore_index=True)
    assert "partner_reuse" in codes(audit_pairs(reused, a, b, frozen=True, matched=True), "error")
    mism = val.copy()
    mism.loc[0, "label_b"] = 1 - mism.loc[0, "label_a"]
    assert "label_mismatch" in codes(audit_pairs(mism, a, b, frozen=True, matched=True), "error")
    crossed = pd.concat([val, make_synthetic_pairs(a, b, "test", "label_matched", seed=0, replace=False)], ignore_index=True)
    assert "pair_split" in codes(audit_pairs(crossed, a, b, frozen=True, matched=True), "error")
    ctrl = make_synthetic_pairs(a, b, "val", "random", seed=0, replace=False)
    assert audit_pairs(ctrl, a, b, frozen=True, matched=False).ok

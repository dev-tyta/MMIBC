"""Manifest builders for the ultrasound sources whose labels live in a spreadsheet (BUS-BRA, BrEaST-Lesions-USG).

The layouts are the ones seen on the first real Colab run; before this, `prepare us` stopped with "no images found" and the manifest had to be
written by hand in the notebook.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mmibc.cli import main
from mmibc.data.manifests import build_manifest
from mmibc.data.synthetic import make_fake_breast_lesions, make_fake_bus_bra, make_fake_busi, make_fake_vindr
from mmibc.data.ultrasound import load_source_manifest
from mmibc.pipeline import prepare, scan_us_sources
from mmibc.utils.config import load_config
from mmibc.utils.paths import ProjectPaths


def test_bus_bra_manifest_matches_what_was_written_by_hand(tmp_path):
    make_fake_bus_bra(tmp_path, n_patients=10)
    raw = tmp_path / "bus_bra"
    out = build_manifest("bus_bra", raw)
    m = pd.read_csv(out)
    assert list(m.columns) == ["uid", "raw_path", "class_name", "mask_paths", "patient_id"]
    assert len(m) == 20 and set(m["class_name"]) <= {"benign", "malignant"}
    meta = pd.read_csv(raw / "BUSBRA" / "bus_data.csv")
    # exactly the mapping of the notebook cell the user wrote for the real dataset
    assert (m["uid"] == meta["ID"]).all() and (m["patient_id"] == meta["Case"]).all()
    assert (m["raw_path"] == meta["ID"].map(lambda x: f"BUSBRA/Images/{x}.png")).all()
    assert (m["mask_paths"] == meta["ID"].map(lambda x: f"BUSBRA/Masks/mask_{x.split('_')[1]}.png")).all()
    assert m.groupby("patient_id")["uid"].count().eq(2).all()                       # left + right breast share a patient id
    loaded = load_source_manifest(out, raw, "bus_bra")
    assert loaded["uid"].iloc[0] == "bus_bra_bus_0001-l" and (loaded["raw_path"].map(lambda p: __import__("pathlib").Path(p).is_file())).all()


def test_breast_lesions_manifest_uses_the_spreadsheet_labels(tmp_path):
    make_fake_breast_lesions(tmp_path, n_benign=6, n_malignant=4, n_normal=2)
    raw = tmp_path / "breast_lesions_usg"
    m = pd.read_csv(build_manifest("breast_lesions_usg", raw))
    assert m["class_name"].value_counts().to_dict() == {"benign": 6, "malignant": 4, "normal": 2}
    assert m["uid"].iloc[0] == "case001" and m["raw_path"].iloc[0].endswith("case001.png")
    assert m["mask_paths"].fillna("").map(bool).sum() == 10                         # normal cases have no tumour mask
    assert m["patient_id"].is_unique
    out = load_source_manifest(raw / "source_manifest.csv", raw, "breast_lesions_usg")
    assert out["mask_paths"].map(lambda s: all(__import__("pathlib").Path(p).is_file() for p in s.split("|") if p)).all()


def test_existing_manifest_is_kept_unless_forced(tmp_path):
    make_fake_bus_bra(tmp_path, n_patients=4)
    raw = tmp_path / "bus_bra"
    (raw / "source_manifest.csv").write_text("uid,raw_path,class_name\nhand_written,BUSBRA/Images/bus_0001-l.png,benign\n")
    assert len(pd.read_csv(build_manifest("bus_bra", raw))) == 1                    # the user's file stands
    assert len(pd.read_csv(build_manifest("bus_bra", raw, force=True))) == 8


def test_missing_files_are_dropped_few_but_refused_many(tmp_path):
    make_fake_bus_bra(tmp_path, n_patients=20)
    raw = tmp_path / "bus_bra"
    (raw / "BUSBRA" / "Images" / "bus_0001-l.png").unlink()                         # 1 of 40: dropped with a warning
    assert len(pd.read_csv(build_manifest("bus_bra", raw, force=True))) == 39
    for i in range(2, 8):
        (raw / "BUSBRA" / "Images" / f"bus_{i:04d}-l.png").unlink()                  # 7 of 40 (> 5 %): the download is incomplete
    with pytest.raises(RuntimeError, match="incomplete or the layout differs"):
        build_manifest("bus_bra", raw, force=True)


def test_unexpected_labels_and_missing_metadata_fail_loudly(tmp_path):
    make_fake_bus_bra(tmp_path, n_patients=3)
    raw = tmp_path / "bus_bra"
    meta = pd.read_csv(raw / "BUSBRA" / "bus_data.csv")
    meta.loc[0, "Pathology"] = "indeterminate"
    meta.to_csv(raw / "BUSBRA" / "bus_data.csv", index=False)
    with pytest.raises(ValueError, match="unexpected Pathology"):
        build_manifest("bus_bra", raw, force=True)
    (raw / "BUSBRA" / "bus_data.csv").unlink()
    with pytest.raises(FileNotFoundError, match="bus_data.csv"):
        build_manifest("bus_bra", raw, force=True)
    with pytest.raises(FileNotFoundError, match="clinical-data"):
        build_manifest("breast_lesions_usg", tmp_path / "empty")                      # nothing there: says what to download
    with pytest.raises(KeyError, match="no manifest builder"):
        build_manifest("busi", raw)


def test_prepare_us_builds_the_manifests_by_itself(tmp_path):
    paths = ProjectPaths(tmp_path / "drive", tmp_path / "local").ensure()
    make_fake_busi(paths.raw)
    make_fake_bus_bra(paths.raw, n_patients=12)
    make_fake_breast_lesions(paths.raw)
    cfg = load_config("smoke.yaml", ["data.us_sources=[busi,bus_bra,breast_lesions_usg]", "data.us_external_source=breast_lesions_usg", "audit.min_per_class_split=1", "training.num_workers=0"])
    scanned = scan_us_sources(cfg, paths)
    assert (paths.raw / "bus_bra" / "source_manifest.csv").exists() and (paths.raw / "breast_lesions_usg" / "source_manifest.csv").exists()
    assert {"busi", "bus_bra", "breast_lesions_usg"} == set(scanned["source"])
    prepare(cfg, paths, ("ultrasound",))
    done = pd.read_csv(paths.processed / "ultrasound" / "manifest.csv")
    assert {"bus_bra", "breast_lesions_usg"} <= set(done["source"]) and done["uid"].is_unique


def test_manifest_command(tmp_path, capsys):
    paths = ProjectPaths(tmp_path / "drive", tmp_path / "local").ensure()
    make_fake_breast_lesions(paths.raw)
    args = ["manifest", "breast_lesions_usg", "--root", str(tmp_path / "drive"), "--local", str(tmp_path / "local")]
    assert main(args) == 0
    assert "source_manifest.csv" in capsys.readouterr().out and (paths.raw / "breast_lesions_usg" / "source_manifest.csv").exists()


def test_adding_the_external_source_does_not_move_the_internal_splits(tmp_path):
    """BrEaST is listed last and held out: the BUSI/BUS-BRA train/val/test assignment must be the one of the pool-only run, so the two stay comparable."""
    from mmibc.data.splits import assign_us_splits

    paths = ProjectPaths(tmp_path / "drive", tmp_path / "local").ensure()
    make_fake_busi(paths.raw)
    make_fake_bus_bra(paths.raw, n_patients=40)
    make_fake_breast_lesions(paths.raw, 30, 20, 2)
    cfg = load_config("smoke.yaml", ["data.us_sources=[busi,bus_bra,breast_lesions_usg]", "audit.min_per_class_split=1", "training.num_workers=0"])
    prepare(cfg, paths, ("ultrasound",))
    m = pd.read_csv(paths.processed / "ultrasound" / "manifest.csv")
    pool = assign_us_splits(m[m["source"] != "breast_lesions_usg"], 0.15, 0.15, 42).set_index("uid")["split"]
    both = assign_us_splits(m, 0.15, 0.15, 42, external_source="breast_lesions_usg")
    assert (both.loc[both["source"] == "breast_lesions_usg", "split"] == "ext_test").all() and (both["split"] == "ext_test").sum() > 0
    internal = both[both["source"] != "breast_lesions_usg"].set_index("uid")["split"]
    assert set(pool.index) == set(internal.index) and (pool == internal.reindex(pool.index)).all()

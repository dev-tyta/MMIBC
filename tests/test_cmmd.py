"""CMMD (DICOM mammograms with biopsy labels): reading, joining the clinical sheet, splits, audit, and the pipeline running on it.

The fake data follows what the REAL files showed (ImageLaterality present, view only in ViewCodeSequence, 2- and 4-image studies, a label for the
biopsied breast only, D1 / D2 subsets of very different malignant rate). Needs pydicom, which is an optional dependency.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

pydicom = pytest.importorskip("pydicom")

from mmibc.cli import main  # noqa: E402
from mmibc.data.cmmd import build_cmmd_table, load_clinical, prepare_cmmd, read_header, read_pixels, view_from_text  # noqa: E402
from mmibc.data.merge_audit import AuditConfig, audit_cmmd  # noqa: E402
from mmibc.data.splits import assign_cmmd_splits  # noqa: E402
from mmibc.data.synthetic import _write_dcm, make_fake_cmmd  # noqa: E402
from mmibc.evaluation.evaluate import by_stratum  # noqa: E402
from mmibc.utils.config import load_config  # noqa: E402
from mmibc.utils.paths import ProjectPaths  # noqa: E402


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    root = tmp_path_factory.mktemp("cmmd_raw")
    make_fake_cmmd(root)
    return root / "cmmd"


@pytest.fixture(scope="module")
def prepared(raw, tmp_path_factory):
    out = tmp_path_factory.mktemp("cmmd_proc")
    return out, prepare_cmmd(raw, out, long_side=112, workers=2)


# ------------------------------------------------------------------------------------------------- DICOM reading
@pytest.mark.parametrize("text,view", [("cranio-caudal", "CC"), ("medio-lateral oblique", "MLO"), ("CC", "CC"), ("MLO", "MLO"), ("Medio-Lateral Oblique", "MLO"), ("latero-medial", ""), ("", "")])
def test_view_text_is_normalised(text, view):
    assert view_from_text(text) == view


def test_header_reading_uses_image_laterality_and_the_view_code_sequence(raw):
    f = next((raw / "CMMD" / "D2-0001").rglob("1-3.dcm"))
    h = read_header(f)
    assert (h["patient"], h["laterality"], h["view"], h["view_src"], h["instance"]) == ("D2-0001", "R", "CC", "view_code_sequence", 3)
    ds = pydicom.dcmread(f, stop_before_pixels=True)
    assert "Laterality" not in ds and "ViewPosition" not in ds                      # exactly what the real files lack


def test_monochrome1_is_inverted_and_deep_pixels_are_windowed(tmp_path):
    ramp = np.tile(np.linspace(0, 4095, 64).astype(np.uint16), (32, 1))             # 12-bit, bright on the right
    _write_dcm(tmp_path / "pos.dcm", ramp, "D1-0001", "L", "cranio-caudal", 1, monochrome1=False)
    _write_dcm(tmp_path / "neg.dcm", ramp, "D1-0001", "L", "cranio-caudal", 1, monochrome1=True)
    pos, neg = read_pixels(tmp_path / "pos.dcm"), read_pixels(tmp_path / "neg.dcm")
    assert pos.dtype == np.uint8 and pos[:, -1].mean() > pos[:, 0].mean() + 200     # windowed to the full 8-bit range
    assert neg[:, 0].mean() > neg[:, -1].mean() + 200                               # MONOCHROME1: low value = white, so it is flipped
    assert np.abs(pos.astype(int) + neg.astype(int) - 255).max() <= 1


# ------------------------------------------------------------------------------------------------- clinical sheet
def test_clinical_sheet_gives_one_label_per_breast_and_malignant_wins(tmp_path):
    rows = pd.DataFrame(dict(ID1=["D1-0001", "D1-0001", "D1-0001", "D1-0002"], LeftRight=["L", "L", "R", "L"], Age=[40] * 4, number=[2] * 4,
                             abnormality=["mass"] * 4, classification=["Benign", "Malignant", "Benign", "Benign"], subtype=[np.nan] * 4))
    rows.to_excel(tmp_path / "CMMD_clinicaldata_revision.xlsx", sheet_name="Sheet1", index=False)
    c = load_clinical(tmp_path).set_index(["patient", "laterality"])
    assert len(c) == 3 and c.loc[("D1-0001", "L"), "label"] == 1 and c.loc[("D1-0001", "L"), "n_lesions"] == 2     # two lesions in one breast
    assert c.loc[("D1-0001", "R"), "label"] == 0                                    # the other breast of the same patient keeps its own label
    rows.assign(classification="Maybe").to_excel(tmp_path / "CMMD_clinicaldata_revision.xlsx", sheet_name="Sheet1", index=False)
    with pytest.raises(ValueError, match="unexpected classification"):
        load_clinical(tmp_path)
    (tmp_path / "CMMD_clinicaldata_revision.xlsx").unlink()
    with pytest.raises(FileNotFoundError, match="clinical spreadsheet"):
        load_clinical(tmp_path)


# ------------------------------------------------------------------------------------------------- the table and prepare
def test_manifest_labels_only_the_biopsied_breast_with_both_views(raw, prepared):
    out, m = prepared
    clin = load_clinical(raw)
    assert len(m) == 2 * len(clin) and m["uid"].is_unique                           # CC + MLO of every labelled breast, nothing else
    assert set(m["view"]) == {"CC", "MLO"} and m.groupby(["study_id", "laterality"])["view"].nunique().eq(2).all()
    key = m.drop_duplicates(["study_id", "laterality"]).set_index(["study_id", "laterality"])["label"].sort_index()
    want = clin.set_index(["patient", "laterality"])["label"].reindex(key.index)
    assert want.notna().all() and (key.to_numpy() == want.to_numpy()).all()         # the sheet's label, for the breast the sheet names
    # a 4-image study contributes only its labelled breast(s): more files on disk than images in the manifest
    n_files = len(list(raw.rglob("*.dcm")))
    assert n_files > len(m) and m["n_study_images"].isin([2, 4]).all()
    assert (m["subset"].isin(["D1", "D2"])).all() and (m["subset"] == m["study_id"].str[:2]).all()
    assert m["lesion_area_frac"].isna().all()                                       # no lesion boxes in CMMD
    assert pd.read_csv(out / "findings.csv").empty and {"image_id", "bx0"} <= set(pd.read_csv(out / "findings.csv").columns)
    assert all((out / p).exists() for p in m["proc_path"])


def test_prepare_resumes_and_a_header_cache_is_reused(raw, tmp_path):
    first = prepare_cmmd(raw, tmp_path, long_side=112, workers=2)
    stamp = {p: p.stat().st_mtime_ns for p in (tmp_path / "images").rglob("*.png")}
    again = prepare_cmmd(raw, tmp_path, long_side=112, workers=2)
    pd.testing.assert_frame_equal(first, again, check_dtype=False)
    assert stamp == {p: p.stat().st_mtime_ns for p in (tmp_path / "images").rglob("*.png")}   # no image was rewritten
    assert (tmp_path / "headers.csv").exists()


def test_missing_view_tag_falls_back_to_the_instance_number_and_is_flagged(raw, tmp_path):
    copy = tmp_path / "cmmd"
    shutil.copytree(raw, copy)
    f = next((copy / "CMMD" / "D2-0005").rglob("1-4.dcm"))                          # D2-0005 has both breasts labelled; instance 4 = right MLO
    ds = pydicom.dcmread(f)
    del ds.ViewCodeSequence
    ds.save_as(f)
    t = build_cmmd_table(copy, cache=None, workers=1).set_index("uid")
    assert t.loc["D2-0005_R_MLO", "view_src"] == "instance_parity"                  # instance 4 is even: MLO (true of the real study D2-0746)
    assert t.loc["D2-0005_R_CC", "view_src"] == "view_code_sequence"


def test_unreadable_files_are_dropped_few_but_refused_many(raw, tmp_path):
    copy = tmp_path / "cmmd"
    shutil.copytree(raw, copy)
    victim = next((copy / "CMMD" / "D1-0001").rglob("1-1.dcm"))
    victim.write_bytes(b"not a dicom")
    t = build_cmmd_table(copy, cache=None, workers=1)                               # 1 of 130 files: dropped with a warning
    assert "D1-0001_R_CC" not in set(t["uid"]) and "D1-0001_L_CC" not in set(t["uid"])
    for f in sorted(copy.rglob("*.dcm"))[:15]:
        f.write_bytes(b"garbage")
    with pytest.raises(RuntimeError, match="no usable side/view"):
        build_cmmd_table(copy, cache=None, workers=1)


def test_an_incomplete_download_is_caught_by_the_tcia_metadata(raw, tmp_path):
    copy = tmp_path / "cmmd"
    shutil.copytree(raw, copy)
    for pid in ("D1-0001",):                                                        # 1 of 44 patients (2 %): warning only
        shutil.rmtree(copy / "CMMD" / pid)
    assert len(build_cmmd_table(copy, cache=None, workers=1)) > 0
    for pid in ("D1-0004", "D1-0005", "D1-0006", "D1-0007"):                        # 5 of 44 (11 %): refuse
        shutil.rmtree(copy / "CMMD" / pid)
    with pytest.raises(RuntimeError, match="download is incomplete"):
        build_cmmd_table(copy, cache=None, workers=1)
    (copy / "metadata.csv").unlink()                                                # without the manifest there is nothing to compare with
    assert len(build_cmmd_table(copy, cache=None, workers=1)) > 0


def test_png_only_downloads_are_refused_with_an_explanation(tmp_path):
    from PIL import Image

    (tmp_path / "x").mkdir()
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / "x" / "a.png")
    with pytest.raises(FileNotFoundError, match="breast side and view cannot be recovered"):
        build_cmmd_table(tmp_path, cache=None, workers=1)


# ------------------------------------------------------------------------------------------------- splits and audit
def test_splits_are_by_patient_and_subsets_restrict(prepared):
    _, m = prepared
    s = assign_cmmd_splits(m, 0.2, 0.2, seed=1)
    assert s.groupby("study_id")["split"].nunique().eq(1).all()                     # both breasts / all views of a patient stay together
    assert set(s["split"]) == {"train", "val", "test"} and set(s["stratum"]) == {"D1", "D2"} and set(s["source"]) == {"cmmd_d1", "cmmd_d2"}
    assert (assign_cmmd_splits(m, 0.2, 0.2, seed=1)["split"] == s["split"]).all()   # deterministic
    d1 = assign_cmmd_splits(m, 0.2, 0.2, seed=1, subsets=["d1"])
    assert set(d1["subset"]) == {"D1"} and len(d1) < len(s)
    with pytest.raises(ValueError, match="matches nothing"):
        assign_cmmd_splits(m, 0.2, 0.2, seed=1, subsets=["D9"])


def _table(n1=60, n2=60, p1=0.5, p2=0.98, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for subset, n, p in (("D1", n1, p1), ("D2", n2, p2)):
        for i in range(n):
            lab = int(rng.random() < p)
            for v in ("CC", "MLO"):
                pid = f"{subset}-{i:04d}"
                rows.append(dict(uid=f"{pid}_L_{v}", study_id=pid, laterality="L", view=v, label=lab, subset=subset, n_study_images=2, proc_path=f"{pid}_{v}.png",
                                 proc_w=100, proc_h=140, lesion_area_frac=np.nan))
    return pd.DataFrame(rows)


def test_audit_flags_the_subset_shortcut_and_reports_the_rates():
    df = assign_cmmd_splits(_table(), 0.15, 0.15, seed=0)
    rep = audit_cmmd(df, None, None, AuditConfig(min_per_class_split=1))
    codes = {f.code for f in rep.warnings}
    assert "subset_shortcut" in codes and rep.ok                                    # a warning, not a stop: the user decides
    rates = rep.stats["malignant_rate_by_subset"]
    assert rates["D2"]["malignant_rate"] > 0.9 > 0.7 > rates["D1"]["malignant_rate"]
    balanced = audit_cmmd(assign_cmmd_splits(_table(p2=0.5), 0.15, 0.15, seed=0), None, None, AuditConfig(min_per_class_split=1))
    assert "subset_shortcut" not in {f.code for f in balanced.warnings}


def test_audit_catches_a_patient_in_two_splits_and_a_breast_with_two_labels():
    df = assign_cmmd_splits(_table(), 0.15, 0.15, seed=0)
    bad = df.copy()
    bad.loc[bad["uid"] == bad["uid"].iloc[0], "split"] = "test" if bad["split"].iloc[0] != "test" else "val"
    assert "split_leak" in {f.code for f in audit_cmmd(bad, None, None, AuditConfig(min_per_class_split=1)).errors}
    bad = df.copy()
    bad.loc[bad.index[0], "label"] = 1 - bad.loc[bad.index[0], "label"]
    assert "breast_label_conflict" in {f.code for f in audit_cmmd(bad, None, None, AuditConfig(min_per_class_split=1)).errors}


def test_by_stratum_reports_a_single_class_stratum_without_an_auroc():
    pairs = pd.DataFrame(dict(pair_id=list("abcdef"), stratum=["D1", "D1", "D1", "D1", "D2", "D2"]))
    pred = pd.DataFrame(dict(pair_id=list("abcdef"), y=[0, 1, 0, 1, 1, 1], p=[0.2, 0.9, 0.4, 0.7, 0.8, 0.6]))
    out = by_stratum(pred, pairs, 0.5)
    assert out["D1"]["auroc"] == 1.0 and out["D1"]["n"] == 4 and out["D1"]["specificity"] == 1.0
    assert out["D2"]["auroc"] is None and out["D2"]["n_neg"] == 0 and out["D2"]["sensitivity"] == 1.0
    assert by_stratum(pred, pairs.drop(columns="stratum"), 0.5) is None             # a source without strata (VinDr) adds nothing


# ------------------------------------------------------------------------------------------------- the pipeline on CMMD
@pytest.fixture(scope="module")
def cmmd_project(raw, tmp_path_factory):
    root = tmp_path_factory.mktemp("cmmd_proj")
    paths = ProjectPaths(root / "drive", root / "local", raw_on_drive=False).ensure()
    shutil.copytree(raw, paths.raw / "cmmd")
    over = ["audit.min_per_class_split=1", "training.num_workers=0", "data.mammo_source=cmmd", "data.store_long_side=112"]
    from mmibc.pipeline import prepare

    prepare(load_config("smoke.yaml", over), paths, ("cmmd",))
    return paths, over


TINY = ["model.backbone=dinov2_tiny_test", "model.pretrained=false", "model.freeze=none", "model.xattn_heads=2", "model.head_hidden=32", "training.epochs=1",
        "input.image_size=56", "eval.controls=[none]", "eval.bootstrap_samples=20", "training.batch_size=8"]


def test_multiview_trains_on_cmmd_and_reports_per_subset_metrics(cmmd_project):
    from mmibc.pipeline import run_experiment

    paths, over = cmmd_project
    cfg = load_config("cmmd_multiview.yaml", [*over, *TINY, "run_name=cmmd_mv_test"])
    res = run_experiment(cfg, paths)
    assert res["splits"]["test"]["auroc"] == res["splits"]["test"]["auroc"]
    assert set(res["splits"]["test"]["by_stratum"]) <= {"D1", "D2"} and "by_stratum" in res["splits"]["val"]
    assert (paths.splits / "cmmd_split.csv").exists() and (paths.splits / "audit_cmmd.md").exists()
    pairs = pd.read_csv(paths.splits / "multiview" / "pairs_test.csv")
    assert pairs["pair_type"].eq("native_multiview").all() and pairs["group_a"].str.startswith("cmmd:").all() and pairs["group_a"].eq(pairs["group_b"]).all()
    assert pairs["pair_id"].is_unique and set(pairs["stratum"]) <= {"D1", "D2"}


def test_cmmd_mammograms_pair_with_ultrasound_but_not_by_lesion_size(cmmd_project, raw_root):
    from mmibc.pipeline import build_pair_sets, build_tables, prepare

    paths, over = cmmd_project
    for name in ("busi", "extra_us"):
        if not (paths.raw / name).exists():
            shutil.copytree(raw_root / name, paths.raw / name)
    us = ["data.us_sources=[busi,extra_us]", "data.us_external_source=extra_us"]
    prepare(load_config("smoke.yaml", [*over, *us]), paths, ("ultrasound",))
    cfg = load_config("cmmd_us_pool.yaml", [*over, *us, *TINY, "task.pair_mode=label_matched"])
    sets = build_pair_sets(cfg, build_tables(cfg, paths), paths)
    test = sets.pair_tables["test"]
    assert (test["label_a"] == test["label_b"]).all() and set(test["stratum"]) <= {"D1", "D2"} and "ext_test" in sets.extra
    bad = load_config("cmmd_us_pool.yaml", [*over, *us, *TINY, "task.pair_mode=size_conditioned"])
    with pytest.raises(ValueError, match="CMMD ships no lesion boxes"):
        build_pair_sets(bad, build_tables(bad, paths), paths)


def test_selecting_cmmd_changes_the_fingerprint_but_the_default_does_not():
    from mmibc.training.trainer import config_fingerprint

    assert config_fingerprint(load_config("vindr_multiview.yaml")) == "48d389f3dde7"          # the first real run still resumes / is skipped
    assert config_fingerprint(load_config("cmmd_multiview.yaml")) != config_fingerprint(load_config("vindr_multiview.yaml"))
    assert config_fingerprint(load_config("cmmd_multiview.yaml", ["data.cmmd.subsets=[D1]"])) != config_fingerprint(load_config("cmmd_multiview.yaml"))
    assert config_fingerprint(load_config("vindr_multiview.yaml", ["data.mammo_source=vindr"])) == "48d389f3dde7"
    with pytest.raises(ValueError, match="mammo_source"):
        from mmibc.pipeline import mammo_source_of

        mammo_source_of(load_config(None, ["data.mammo_source=foo"]))


def test_prepare_all_includes_cmmd_only_when_it_has_been_downloaded(tmp_path, monkeypatch):
    import mmibc.pipeline as pipeline

    seen = []
    monkeypatch.setattr(pipeline, "prepare", lambda cfg, paths, which, force=False: seen.append(tuple(which)))
    paths = ProjectPaths(tmp_path / "drive", tmp_path / "local").ensure()
    args = ["--root", str(tmp_path / "drive"), "--local", str(tmp_path / "local")]
    assert main(["prepare", "all", *args]) == 0 and seen[-1] == ("vindr", "ultrasound")
    (paths.raw / "cmmd").mkdir(parents=True)
    assert main(["prepare", "all", *args]) == 0 and seen[-1] == ("vindr", "ultrasound", "cmmd")
    assert main(["prepare", "cmmd", *args]) == 0 and seen[-1] == ("cmmd",)


def test_cmmd_configs_and_suites_are_well_formed():
    from mmibc.models.backbones import BACKBONES
    from mmibc.suite import load_suite

    for name, kind in (("cmmd_multiview.yaml", "multiview"), ("cmmd_us_pool.yaml", "synthetic_pairs")):
        cfg = load_config(name)
        assert cfg.data.mammo_source == "cmmd" and cfg.task.kind == kind and list(cfg.data.cmmd.subsets) == ["D1", "D2"]
    for suite in ("cmmd_pool_ladder", "backbones_cmmd"):
        spec = load_suite(suite)
        assert load_config(spec["config"]).data.mammo_source == "cmmd" and spec["reference"] in [r["name"] for r in spec["runs"]]
    runs = load_suite("backbones_cmmd")["runs"]
    assert load_suite("backbones_cmmd")["continue_on_error"] and {s.split("=")[1] for r in runs for s in r["set"] if s.startswith("model.backbone=")} <= set(BACKBONES)

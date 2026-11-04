"""``mmibc experiments``: every track on every dataset that is available, skipping what is missing and surviving what fails.

Fake VinDr, CMMD (DICOM) and ultrasound sources stand in for the real downloads; the backbone is the tiny random-weight test model, so this checks the
machinery (data discovery, up-front preparation, skip/fail/reuse, the summary) and not the quality of any model.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pydicom")

from mmibc.cli import main  # noqa: E402
from mmibc.data.synthetic import make_fake_cmmd  # noqa: E402
from mmibc.experiments import (  # noqa: E402
    build_plan, dataset_state, load_program, needs_of, prepare_datasets, processed_us_sources, run_program, select_tracks, track_configs,
)
from mmibc.models import backbones as bb  # noqa: E402
from mmibc.pipeline import build_tables, run_dir_for, select_us_sources  # noqa: E402
from mmibc.utils.config import load_config  # noqa: E402
from mmibc.utils.paths import ProjectPaths  # noqa: E402

GLOBAL = ["audit.min_per_class_split=1", "training.num_workers=0", "training.epochs=1", "eval.controls=[none]", "eval.bootstrap_samples=20"]

PROGRAM = """name: mini
description: tiny programme on fake data
tracks:
  - {name: t_vindr_mv, group: vindr, config: smoke, set: [task.kind=multiview, run_name=x_vindr_mv]}
  - {name: t_bad, group: vindr, config: smoke, set: [task.kind=multiview, input.image_size=57, run_name=x_bad]}
  - {name: t_cmmd_mv,  group: cmmd,  config: smoke, set: [task.kind=multiview, data.mammo_source=cmmd, run_name=x_cmmd_mv]}
  - {name: t_cmmd_pool, group: cmmd, config: smoke, set: [task.kind=synthetic_pairs, task.pair_mode=label_matched, data.mammo_source=cmmd, "data.us_sources=[busi]", data.us_external_source=null, run_name=x_cmmd_pool]}
  - {name: t_ext, group: external, config: smoke, set: [task.kind=synthetic_pairs, task.pair_mode=label_matched, "data.us_sources=[busi,extra_us]", data.us_external_source=extra_us, run_name=x_ext]}
  - {name: t_no_data, group: external, config: smoke, set: [task.kind=synthetic_pairs, "data.us_sources=[busi,not_downloaded]", run_name=x_nodata]}
  - {name: s_mini, group: suites, suite: SUITE_PATH, optional: true}
"""

SUITE = """config: smoke.yaml
reference: a
continue_on_error: true
runs:
  - {name: a, set: [task.kind=multiview]}
  - {name: b, set: [task.kind=multiview, model.fusion=concat]}
"""


@pytest.fixture(scope="module")
def exp(raw_root, tmp_path_factory):
    root = tmp_path_factory.mktemp("exp")
    paths = ProjectPaths(root / "drive", root / "local").ensure()
    for name in ("vindr_images", "vindr_annotations", "busi", "extra_us"):
        shutil.copytree(raw_root / name, paths.raw / name)
    make_fake_cmmd(paths.raw)
    suite = root / "mini_suite.yaml"
    suite.write_text(SUITE)
    program = root / "mini.yaml"
    program.write_text(PROGRAM.replace("SUITE_PATH", str(suite)))
    return paths, program, root


@pytest.fixture(scope="module")
def first(exp):
    paths, program, _ = exp
    summary, rc = run_program(program, paths, GLOBAL)
    return summary.set_index("track"), rc


# ----------------------------------------------------------------------------------------------- the shipped program
def test_the_shipped_program_is_consistent():
    spec, tracks = load_program("all")
    runs = [t for t in tracks if t.kind == "run"]
    dirs = {}
    for t in runs:
        (_, cfg), = track_configs(t, [])
        assert str(cfg.model.backbone) in bb.BACKBONES
        d = f"{cfg.run_name}" if cfg.task.kind == "multiview" else f"{cfg.run_name}_{cfg.task.pair_mode}"
        assert d not in dirs, f"tracks {t.name} and {dirs[d]} would write the same run directory"
        dirs[d] = t.name
    # run names that already exist on Drive are reused, so earlier results are picked up instead of retrained
    names = {t.name: run_dir_for(track_configs(t, [])[0][1], ProjectPaths(Path("/r"), Path("/l"))).name for t in runs}
    assert names["vindr_multiview"] == "multiview_xattn_s42_dinov2_base"
    assert names["vindr_pool_label_matched"] == "pool_busi_busbra_label_matched_dinov2_base"
    assert names["vindr_pool_shuffled_partner"] == "pool_busi_busbra_shuffled_partner_dinov2_base"
    for t in tracks:
        if t.kind == "suite":
            assert t.optional and track_configs(t, [])                              # suites are opt-in and every row's configuration loads


def test_needs_follow_the_configuration():
    spec, tracks = load_program("all")
    by = {t.name: t for t in tracks}
    need = lambda n: needs_of(track_configs(by[n], [])[0][1])  # noqa: E731
    assert need("vindr_multiview") == ["vindr"]
    assert need("vindr_pool_random") == ["vindr", "us:busi", "us:bus_bra"]
    assert need("cmmd_multiview_d1") == ["cmmd"]
    assert need("cmmd_pool_d1_label_matched") == ["cmmd", "us:busi", "us:bus_bra"]
    assert need("vindr_external") == ["vindr", "us:busi", "us:bus_bra", "us:breast_lesions_usg"]
    assert need("cmmd_external_d1") == ["cmmd", "us:busi", "us:bus_bra", "us:breast_lesions_usg"]


def test_track_selection():
    _, tracks = load_program("all")
    default = [t.name for t in select_tracks(tracks)]
    assert "vindr_multiview" in default and not any(n.startswith("suite_") for n in default)
    assert any(n.startswith("suite_") for n in (t.name for t in select_tracks(tracks, suites=True)))
    assert {t.group for t in select_tracks(tracks, only=["cmmd"])} == {"cmmd"}
    assert "vindr_multiview" not in [t.name for t in select_tracks(tracks, skip=["vindr"])]
    assert [t.name for t in select_tracks(tracks, only=["suite_backbones"])] == ["suite_backbones"]   # naming an optional track runs it
    with pytest.raises(ValueError, match="unknown track"):
        select_tracks(tracks, only=["nope"])


def test_a_track_needs_exactly_one_of_config_or_suite(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("tracks:\n  - {name: a, config: smoke, suite: ladder}\n")
    with pytest.raises(ValueError, match="exactly one"):
        load_program(f)
    f.write_text("tracks:\n  - {name: a, config: smoke}\n  - {name: a, config: smoke}\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_program(f)


# ----------------------------------------------------------------------------------------------- planning touches nothing
def test_the_plan_on_an_empty_project_blocks_everything_and_writes_nothing(tmp_path):
    paths = ProjectPaths(tmp_path / "d", tmp_path / "l").ensure()
    plan, rc = run_program("all", paths, [], dry_run=True)
    assert rc == 0 and len(plan) == 13 and plan["plan"].str.startswith("blocked").all()
    assert list(paths.runs.iterdir()) == [] and list(paths.processed.iterdir()) == []
    assert main(["experiments", "--list", "--root", str(tmp_path / "d"), "--local", str(tmp_path / "l"), "--only", "cmmd"]) == 0


# ----------------------------------------------------------------------------------------------- a whole programme on fake data
def test_available_datasets_are_prepared_once_and_missing_ones_are_reported(exp, first):
    paths, _, _ = exp
    summary, rc = first
    assert dataset_state("vindr", paths) == dataset_state("cmmd", paths) == "ready"
    assert processed_us_sources(paths) == {"busi", "extra_us"}                      # the union of what the tracks need AND exists
    nd = summary.loc["t_no_data", "state"]
    assert nd.startswith("SKIPPED") and "us:not_downloaded" in nd and "us:busi" not in nd


def test_finished_failed_and_skipped_tracks_are_recorded_and_do_not_stop_the_rest(exp, first):
    paths, program, _ = exp
    summary, rc = first
    for name in ("t_vindr_mv", "t_cmmd_mv", "t_cmmd_pool", "t_ext"):
        assert summary.loc[name, "state"] == "done", summary.loc[name].to_dict()
        assert 0.0 <= summary.loc[name, "auroc"] <= 1.0 and summary.loc[name, "auroc_lo"] <= summary.loc[name, "auroc_hi"]
        assert bool(summary.loc[name, "reused"]) is False
    assert summary.loc["t_bad", "state"].startswith("FAILED: ValueError")
    out = paths.runs / "experiments_mini"
    assert (out / "FAILED_t_bad.txt").exists() and "Traceback" in (out / "FAILED_t_bad.txt").read_text()
    assert rc == 1                                                                  # a failure is visible in the exit code
    assert {"plan.csv", "status.csv", "summary.csv", "summary.md"} <= {p.name for p in out.iterdir()}
    md = (out / "summary.md").read_text()
    assert "never pooled" in md and "t_ext" in md
    order = list(pd.read_csv(out / "status.csv")["track"])
    assert order.index("t_bad") < order.index("t_ext") and summary.loc["t_ext", "state"] == "done"      # the failure sat in the middle of the programme
    assert "t_no_data" in order


def test_external_and_per_subset_numbers_reach_the_summary(first):
    summary, _ = first
    assert pd.notna(summary.loc["t_ext", "ext_auroc"])                              # the held-out ultrasound source was scored
    assert pd.isna(summary.loc["t_vindr_mv"].get("ext_auroc", float("nan")))
    assert {"auroc_D1", "auroc_D2"} <= set(summary.columns)                         # CMMD tracks report each subset
    assert pd.isna(summary.loc["t_vindr_mv"].get("auroc_D1", float("nan")))


def test_a_pool_run_trains_only_on_the_sources_it_lists_even_though_the_pool_holds_more(exp, first):
    paths, _, _ = exp
    assert processed_us_sources(paths) == {"busi", "extra_us"}
    run = paths.runs / "x_cmmd_pool_label_matched_dinov2_tiny_test"
    used = set()
    for split in ("val", "test"):
        used |= set(pd.read_csv(run / f"predictions_{split}.csv")["uid_b"].astype(str).str.split("_").str[0])
    assert used == {"busi"}, used                                                   # extra_us (also in the pool) never leaked into this run


def test_a_second_call_reuses_finished_tracks_and_retrains_nothing(exp, first):
    paths, program, _ = exp
    stamp = {n: (paths.runs / n / "metrics.json").stat().st_mtime_ns for n in ("x_vindr_mv_dinov2_tiny_test", "x_cmmd_mv_dinov2_tiny_test")}
    summary, rc = run_program(program, paths, GLOBAL)
    summary = summary.set_index("track")
    for name in ("t_vindr_mv", "t_cmmd_mv", "t_cmmd_pool", "t_ext"):
        assert summary.loc[name, "state"].startswith("done") and bool(summary.loc[name, "reused"]) is True
    for n, t in stamp.items():
        assert (paths.runs / n / "metrics.json").stat().st_mtime_ns == t
    assert summary.loc["t_bad", "state"].startswith("FAILED") and rc == 1
    assert summary.loc["t_vindr_mv", "auroc"] == pytest.approx(first[0].loc["t_vindr_mv", "auroc"])


def test_the_plan_after_a_run_says_what_is_finished(exp, first):
    paths, program, _ = exp
    _, tracks = load_program(program)
    plan = build_plan(select_tracks(tracks), paths, GLOBAL).set_index("track")
    assert plan.loc["t_vindr_mv", "plan"].startswith("already finished")
    assert plan.loc["t_no_data", "plan"].startswith("blocked") and plan.loc["t_bad", "plan"] == "will run"


def test_a_suite_track_runs_when_asked_and_reports_its_rows(exp, first):
    paths, program, _ = exp
    summary, rc = run_program(program, paths, GLOBAL, only=["s_mini"])
    row = summary.set_index("track").loc["s_mini"]
    assert row["state"] == "done" and row["rows_ok"] == 2 and row["rows_failed"] == 0 and row["rows_skipped"] == 0
    assert row["results"] == "suite_mini_suite/results.md" and (paths.runs / "suite_mini_suite" / "results.csv").exists()
    assert rc == 0


def test_the_cli_runs_a_selection(exp, first, capsys):
    paths, program, root = exp
    assert main(["experiments", str(program), "--only", "t_vindr_mv", "t_cmmd_mv", "--root", str(root / "drive"), "--local", str(root / "local"), "--set", *GLOBAL]) == 0
    out = capsys.readouterr().out
    assert "t_vindr_mv" in out and "done" in out and "summary.md" in out


def test_a_track_with_an_unavailable_gated_backbone_is_skipped(exp, monkeypatch):
    paths, program, _ = exp
    monkeypatch.setattr("mmibc.experiments.unavailable_reason", lambda name: "gated and no token" if name == "dinov2_tiny_test" else None)
    summary, rc = run_program(program, paths, GLOBAL, only=["t_vindr_mv"], force=False)
    row = summary.set_index("track").loc["t_vindr_mv"]
    assert row["state"].startswith("SKIPPED") and "no token" in row["state"]
    assert rc == 1                                                                  # nothing finished in this call


def test_a_dataset_that_cannot_be_prepared_fails_its_tracks_with_the_reason(raw_root, tmp_path):
    paths = ProjectPaths(tmp_path / "d", tmp_path / "l").ensure()
    (paths.raw / "cmmd").mkdir(parents=True)
    (paths.raw / "cmmd" / "notes.txt").write_text("a download that holds no DICOM")      # present, but unusable
    prog = tmp_path / "p.yaml"
    prog.write_text("name: p\ntracks:\n  - {name: c, group: cmmd, config: smoke, set: [task.kind=multiview, data.mammo_source=cmmd]}\n")
    summary, rc = run_program(prog, paths, GLOBAL)
    state = summary.set_index("track").loc["c", "state"]
    assert state.startswith("FAILED: dataset cmmd") and rc == 1


# ----------------------------------------------------------------------------------------------- ultrasound pool handling
def _cfg(*extra):
    return load_config("smoke.yaml", [*GLOBAL, *extra])


def test_only_the_listed_sources_are_selected_from_the_shared_pool():
    m = pd.DataFrame(dict(uid=["a", "b", "c"], source=["busi", "extra_us", "breast_lesions_usg"], class_name=["benign"] * 3))
    sel = select_us_sources(m, _cfg("data.us_sources=[busi,extra_us]"), "pool")
    assert list(sel["source"]) == ["busi", "extra_us"]
    with pytest.raises(ValueError, match="not_here.*mmibc prepare us"):
        select_us_sources(m, _cfg("data.us_sources=[busi,not_here]"), "pool")
    with pytest.raises(ValueError, match="must also be listed"):
        select_us_sources(m, _cfg("data.us_sources=[busi]", "data.us_external_source=extra_us"), "pool")


def test_build_tables_uses_only_the_configured_ultrasound_sources(exp, first):
    paths, _, _ = exp
    t = build_tables(_cfg("task.kind=synthetic_pairs", "data.us_sources=[busi]", "data.us_external_source=null"), paths, stage=False)
    assert set(t.us["source"]) == {"busi"}                                          # the pool also holds extra_us


def test_a_processed_source_whose_raw_folder_is_gone_is_kept_when_another_source_is_added(raw_root, tmp_path):
    paths = ProjectPaths(tmp_path / "d", tmp_path / "l").ensure()
    for name in ("busi", "extra_us"):
        shutil.copytree(raw_root / name, paths.raw / name)
    over = [*GLOBAL, "data.store_long_side=112"]
    assert prepare_datasets({"us:busi", "us:extra_us"}, paths, over) == {"us:busi": "ready", "us:extra_us": "ready"}
    shutil.move(paths.raw / "extra_us", paths.raw / "extra_us2")                    # raw of extra_us is gone, a new source appears
    got = prepare_datasets({"us:busi", "us:extra_us", "us:extra_us2"}, paths, over)
    assert got == {"us:busi": "ready", "us:extra_us": "ready", "us:extra_us2": "ready"}
    m = pd.read_csv(paths.processed / "ultrasound" / "manifest.csv")
    assert set(m["source"]) == {"busi", "extra_us", "extra_us2"} and m["uid"].is_unique
    assert dataset_state("us:nowhere", paths) == "missing"

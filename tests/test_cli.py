from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmibc.cli import build_parser, main

OVER = ["audit.min_per_class_split=1", "training.num_workers=0", "data.us_sources=[busi,extra_us]", "data.us_external_source=extra_us", "training.epochs=1",
        "eval.controls=[none,drop_a,shuffle_b]", "xai.n_samples=6", "xai.steps=4"]


@pytest.fixture(scope="module")
def smoke_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli_smoke")
    assert main(["smoke", "--workdir", str(d)]) == 0
    return d


def _args(d: Path, name: str, kind: str) -> list[str]:
    return ["-c", "smoke", "--root", str(d / "drive"), "--local", str(d / "local"), "--set", *OVER, f"task.kind={kind}", f"run_name={name}"]


def test_every_command_is_registered():
    names = set(build_parser()._subparsers._group_actions[0].choices)
    assert names == {"download", "inspect", "manifest", "prepare", "audit", "train", "evaluate", "xai", "run", "suite", "literature", "figures", "smoke"}


def test_smoke_leaves_runs_audits_and_figures(smoke_dir):
    runs = sorted((smoke_dir / "drive" / "runs").glob("smoke_*"))
    assert len(runs) == 2
    for r in runs:
        m = json.loads((r / "metrics.json").read_text())
        assert m["threshold_chosen_on"] == "val" and "test" in m["splits"] and {c["control"] for c in m["controls"]} == {"none", "drop_a", "shuffle_b"}
        assert (r / "best.pt").exists() and (r / "config.yaml").exists() and list(r.glob("xai_summary_test.csv"))
    assert any((smoke_dir / "drive" / "splits").glob("audit_*.md"))
    assert len(list((smoke_dir / "figures").glob("*.pdf"))) >= 7


def test_evaluate_reuses_the_trained_weights_and_keeps_the_same_threshold_source(smoke_dir, capsys):
    run = next((smoke_dir / "drive" / "runs").glob("smoke_multiview*"))
    before = json.loads((run / "metrics.json").read_text())["splits"]["test"]["auroc"]
    assert main(["evaluate", *_args(smoke_dir, "smoke_multiview", "multiview")]) == 0
    after = json.loads((run / "metrics.json").read_text())["splits"]["test"]["auroc"]
    assert after == pytest.approx(before, abs=1e-6)                      # same weights, same frozen pairs -> same score


def test_audit_command_prints_every_audit_and_passes(smoke_dir, capsys):
    assert main(["audit", *_args(smoke_dir, "smoke_multiview", "multiview")]) == 0
    out = capsys.readouterr().out
    assert "===== vindr: PASS" in out and "pairs_test" in out


def test_literature_command_flags_protocol_comparability(smoke_dir, capsys, tmp_path):
    run = next((smoke_dir / "drive" / "runs").glob("smoke_multiview*"))
    assert main(["literature", str(run), "--dataset", "VinDr", "--out", str(tmp_path)]) == 0
    import pandas as pd

    df = pd.read_csv(tmp_path / "literature_comparison.csv", keep_default_na=False)
    assert df.iloc[0]["source"] == "this work" and (df["comparable_with_published"] != "").all()
    assert (tmp_path / "literature_comparison.tex").read_text().startswith(r"\begin{table}")


def test_figures_command_builds_from_finished_runs_and_refuses_when_there_are_none(smoke_dir, tmp_path, capsys):
    out = tmp_path / "figs"
    assert main(["figures", "--root", str(smoke_dir / "drive"), "--out", str(out), "--template", "ieee"]) == 0
    assert (out / "fig_roc.pdf").exists() and (out / "fig_roc.png").exists()
    assert main(["figures", "--root", str(tmp_path / "empty_root"), "--out", str(tmp_path / "x")]) == 1


def test_download_without_arguments_lists_the_registered_datasets(tmp_path, capsys):
    assert main(["download", "--root", str(tmp_path / "r"), "--local", str(tmp_path / "l")]) == 0
    listing = capsys.readouterr().out
    assert "busi (kaggle)" in listing and "breast_lesions_usg (url)" in listing and "cmmd (kaggle)" in listing

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from mmibc.figures import corrections as C
from mmibc.figures import tex
from mmibc.figures.style import WIDTHS, apply_style, fig_width, save_figure

needs_tex = pytest.mark.skipif(not tex.have_tex(), reason="pdflatex / poppler not installed")
FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"


# ------------------------------------------------------------------------------------------ corrections key
def test_key_is_complete_unique_and_ordered():
    assert C.IDS == [f"C{i}" for i in range(1, 15)]
    assert all(k.component and k.v1 and k.v2 and k.evidence for k in C.KEY)


def test_tex_escape_handles_identifiers_quotes_and_specials():
    s = C.tex_escape("'normal' mapped; v_proj(context); x 4 families; CC-MLO & 5% of #1")
    assert "``normal''" in s and r"\texttt{v\_\allowbreak{}proj}" in s and r"$\times$ 4" in s
    assert "CC--MLO" in s and r"\&" in s and r"5\%" in s and r"\#1" in s
    assert "_" not in re.sub(r"\\_", "", s)                       # no unescaped underscore survives


def test_committed_tables_are_current():
    assert (FIG / "tab_corrections.tex").read_text() == C.latex_table(False)
    assert (FIG / "tab_corrections_evidence.tex").read_text() == C.latex_table(True)


def test_badges_in_the_diagrams_match_the_key():
    arch = tex.badge_ids(FIG / "fig_architecture.tex")
    algo = tex.badge_ids(FIG / "fig_algorithm.tex")
    assert arch == set(C.IDS), f"architecture must reference every correction; missing {set(C.IDS) - arch}, unknown {arch - set(C.IDS)}"
    assert algo <= set(C.IDS) and algo, f"unknown badge in the algorithm figure: {algo - set(C.IDS)}"


# ------------------------------------------------------------------------------------------ TikZ diagrams
@needs_tex
def test_diagrams_compile_embed_fonts_and_match_the_text_width(tmp_path):
    from PIL import Image

    for name in tex.DIAGRAMS:
        made = tex.build_diagram(name, tmp_path, formats=("pdf", "svg", "png"), dpi=300)
        pdf = tmp_path / f"{name}.pdf"
        fonts = tex.pdf_fonts(pdf)
        assert fonts and all(f["embedded"] and "Type 3" not in f["type"] for f in fonts), fonts
        width_cm = tex.pdf_width_cm(pdf) - 6 / 72 * 2.54          # standalone adds a 3 pt border on each side
        assert width_cm == pytest.approx(tex.TEXT_WIDTH_CM, abs=0.02), f"{name}: {width_cm:.3f} cm"
        im = Image.open(tmp_path / f"{name}.png")
        assert im.size[0] == pytest.approx(tex.pdf_width_cm(pdf) / 2.54 * 300, abs=2)
        assert (tmp_path / f"{name}.svg").read_text()[:200].lstrip().startswith(("<?xml", "<svg"))
        assert len(made) == 3


@needs_tex
@pytest.mark.parametrize("table", ["tab_corrections", "tab_corrections_evidence"])
def test_corrections_tables_fit_the_text_width(table, tmp_path):
    pdf = tex.compile_tex("\\begin{minipage}{12.2cm}\\input{%s}\\end{minipage}" % (FIG / table).as_posix(), tmp_path / f"{table}.pdf")
    log = pdf.with_suffix(".log").read_text()
    assert "Overfull" not in log, [ln for ln in log.splitlines() if "Overfull" in ln][:3]


def test_committed_diagram_exports_exist():
    for name in tex.DIAGRAMS:
        for ext in ("pdf", "svg", "png"):
            assert (FIG / "pdf" / f"{name}.{ext}").stat().st_size > 10_000, f"{name}.{ext} missing: run `mmibc figures --tex`"


# ------------------------------------------------------------------------------------------ matplotlib style (data plots)
def test_template_widths_are_defined_for_every_venue():
    assert fig_width("lncs") == pytest.approx(4.8) and fig_width("ieee", "double") > fig_width("ieee", "single")
    assert set(WIDTHS) >= {"lncs", "ieee", "elsevier", "generic"}


@needs_tex
def test_matplotlib_pdf_has_embedded_non_type3_fonts_and_png_is_600_dpi(tmp_path):
    import matplotlib.pyplot as plt
    from PIL import Image

    apply_style("lncs")
    fig, ax = plt.subplots(figsize=(fig_width("lncs"), 2.2))
    ax.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20) ** 2, label=r"$\mathrm{AUROC}=0.91$")
    ax.set_xlabel("1 - specificity"), ax.legend()
    out = save_figure(fig, tmp_path / "roc", ("pdf", "png", "svg"))
    plt.close(fig)
    fonts = tex.pdf_fonts(tmp_path / "roc.pdf")
    assert fonts and all(f["embedded"] and "Type 3" not in f["type"] for f in fonts), fonts
    dpi = Image.open(tmp_path / "roc.png").info["dpi"]
    assert all(abs(d - 600) < 1 for d in dpi)
    assert len(out) == 3


def test_pdf_output_is_byte_stable(tmp_path):
    import matplotlib.pyplot as plt

    apply_style("lncs")
    blobs = []
    for i in range(2):
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([0, 1], [0, 1])
        save_figure(fig, tmp_path / f"f{i}", ("pdf",))
        plt.close(fig)
        blobs.append((tmp_path / f"f{i}.pdf").read_bytes())
    assert blobs[0] == blobs[1]


# ------------------------------------------------------------------------------------------ data plots
from conftest import make_fake_run  # noqa: E402

import logging  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from mmibc.figures import make, plots  # noqa: E402


@pytest.fixture()
def runs(tmp_path):
    r = {n: make_fake_run(tmp_path / "runs", n, sep, i + 1) for i, (n, sep) in enumerate([("unimodal_a", 0.9), ("unimodal_b", 1.1), ("concat", 1.3), ("xattn", 1.6)])}
    yield r
    plt.close("all")


def test_roc_legend_reports_the_stored_auroc_and_delong_interval(runs):
    fig = plots.roc_plot({k: runs[k] for k in ("concat", "xattn")}, "test")
    texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    import json

    for (name, run), t in zip({k: runs[k] for k in ("concat", "xattn")}.items(), texts):
        m = json.loads((run / "metrics.json").read_text())["splits"]["test"]
        lo, hi = m["auroc_delong_ci"]
        assert f"{m['auroc']:.3f}" in t and f"{lo:.3f}" in t and f"{hi:.3f}" in t, t
    ax = fig.axes[0]
    assert ax.get_xlim()[0] <= 0 and ax.get_xlim()[1] >= 1 and ax.get_aspect() in (1.0, "equal")


def test_ladder_orders_runs_and_marks_the_reference(runs):
    fig = plots.ladder_plot(runs, "test", reference="concat", sort=True)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    import json

    aucs = [json.loads((runs[k.replace(" ", "_")] / "metrics.json").read_text())["splits"]["test"]["auroc"] for k in labels]
    assert aucs == sorted(aucs) and len(labels) == 4
    assert any(abs(ln.get_xdata()[0] - aucs[labels.index("concat")]) < 1e-9 for ln in ax.lines if len(set(ln.get_xdata())) == 1)   # reference line


def test_controls_plot_shows_every_control_with_its_delta(runs):
    fig = plots.controls_plot(runs["xattn"])
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == [plots.CONTROL_LABELS[c] for c in ("none", "drop_a", "drop_b", "shuffle_b")]
    assert any("-0.080" in t.get_text() for t in ax.texts)


def test_calibration_plot_reports_ece_and_brier(runs):
    from mmibc.evaluation.metrics import calibration

    fig = plots.calibration_plot({"xattn": runs["xattn"]}, "test")
    y = pd.read_csv(runs["xattn"] / "predictions_test.csv")
    c = calibration(y["y"].to_numpy(), y["p"].to_numpy())
    assert f"ECE {c['ece']:.3f}" in fig.axes[0].get_legend().get_texts()[0].get_text()


def test_xai_summary_draws_random_baseline_hollow_and_every_method(runs):
    fig = plots.xai_summary_plot(runs["xattn"], "test")
    assert len(fig.axes) == 5
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["Grad-CAM", "Grad$\\times$act", "Rollout", "Cross-attn.", "Random map"]
    hollow = [ln for ln in fig.axes[0].lines if ln.get_marker() == "o" and ln.get_markerfacecolor() == "white"]
    assert len(hollow) == 2                                       # one per stream


def test_overlay_panel_scales_each_map_and_handles_missing_masks():
    yy, xx = np.mgrid[0:64, 0:64]
    item = dict(image=np.full((64, 64), 0.5), cam=(xx + yy).astype(float) * 7 + 3, mask=None, title="t")
    fig = plots.overlay_panel([item, dict(item, mask=(xx - 30) ** 2 + (yy - 30) ** 2 < 100)])
    assert fig.axes[0].images and len(fig.axes) == 4
    ov = fig.axes[2].images[1].get_array()
    assert ov.min() == pytest.approx(0) and ov.max() == pytest.approx(1)


def test_composition_counts_match_the_table():
    df = pd.DataFrame(dict(split=["train"] * 5 + ["val"] * 2 + ["test"] * 3, label=[0, 0, 1, 1, 1, 0, 1, 0, 0, 1]))
    fig = plots.composition_plot(df)
    heights = [p.get_height() for p in fig.axes[0].patches]               # one bar segment per (split, class)
    assert sorted(heights) == sorted([2, 3, 1, 1, 2, 1])
    totals = [t.get_text() for t in fig.axes[0].texts]
    assert totals == ["5", "2", "3"]
    plt.close(fig)


@needs_tex
def test_make_figures_writes_vector_and_600_dpi_outputs_with_clean_fonts(runs, tmp_path):
    from PIL import Image

    made = make.make_figures(runs, tmp_path / "out", reference="concat", xai_run="xattn")
    names = {m.stem for m in made}
    assert {"fig_roc", "fig_ladder", "fig_calibration", "fig_controls_xattn", "fig_xai_xattn"} <= names
    for pdf in made:
        fonts = tex.pdf_fonts(pdf)
        assert fonts and all(f["embedded"] and "Type 3" not in f["type"] for f in fonts), (pdf.name, fonts)
        assert all(abs(d - 600) < 1 for d in Image.open(pdf.with_suffix(".png")).info["dpi"])
        assert pdf.with_suffix(".svg").exists()


def test_make_figures_skips_runs_without_artifacts_and_says_so(runs, tmp_path, caplog):
    broken = tmp_path / "runs" / "broken"
    broken.mkdir()
    with caplog.at_level(logging.WARNING):
        made = make.make_figures({"concat": runs["concat"], "broken": broken}, tmp_path / "out")
    assert "run 'broken' has no metrics.json" in caplog.text
    assert not any("broken" in m.name for m in made) and "fig_ladder" not in {m.stem for m in made}      # one usable run: no ladder

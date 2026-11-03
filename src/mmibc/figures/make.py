"""Render every figure whose inputs exist. Missing inputs are skipped with a log line, never faked."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mmibc.figures import plots
from mmibc.figures.style import save_figure  # noqa: F401  (re-exported for notebooks)

log = logging.getLogger(__name__)


def make_figures(runs: dict[str, Path], out_dir: Path, template: str = "lncs", split: str = "test", reference: str | None = None,
                 splits_dir: Path | None = None, xai_run: str | None = None) -> list[Path]:
    """runs: {label: run_dir}. Writes into ``out_dir`` and returns the PDF paths."""
    out_dir = Path(out_dir)
    made: list[Path] = []

    def done(name: str) -> None:
        made.append(out_dir / f"{name}.pdf")

    ready = {k: Path(v) for k, v in runs.items() if (Path(v) / "metrics.json").exists() and (Path(v) / f"predictions_{split}.csv").exists()}
    for k in set(runs) - set(ready):
        log.warning("run '%s' has no metrics.json / predictions_%s.csv: skipped", k, split)
    if ready:
        plots.roc_plot(ready, split, template, out_dir / "fig_roc"), done("fig_roc")
        plots.calibration_plot(ready, split, template, out_dir / "fig_calibration"), done("fig_calibration")
    if len(ready) >= 2:
        plots.ladder_plot(ready, split, reference, template, out_dir / "fig_ladder"), done("fig_ladder")
    for k, r in ready.items():
        if (r / "controls.csv").exists():
            plots.controls_plot(r, template, out_dir / f"fig_controls_{k}"), done(f"fig_controls_{k}")
    xr = ready.get(xai_run) if xai_run else None
    for k, r in ([(xai_run, xr)] if xr else ready.items()):
        if (r / f"xai_summary_{split}.csv").exists():
            plots.xai_summary_plot(r, split, template, out_dir / f"fig_xai_{k}"), done(f"fig_xai_{k}")
    if splits_dir:
        for name, col in (("vindr_split.csv", "label"), ("ultrasound_split.csv", "label")):
            f = Path(splits_dir) / name
            if f.exists():
                df = pd.read_csv(f)
                col = col if col in df else ("class_name" if "class_name" in df else df.columns[0])
                stem = f"fig_composition_{name.split('_')[0]}"
                plots.composition_plot(df, template, out_dir / stem, group_col=col), done(stem)
    import matplotlib.pyplot as plt

    plt.close("all")
    return made

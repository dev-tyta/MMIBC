"""Data figures for the paper, drawn from the artifacts a run leaves on disk (predictions_*.csv, metrics.json,
controls.csv, xai_summary_*.csv, cams/*.npy).  Every function returns a matplotlib figure; pass ``out`` (a path stem) to
also write vector PDF + SVG + 600-dpi PNG through ``save_figure``.  Colours are colour-blind safe, text is the STIX serif
face (matches Times / LNCS body text) and widths follow the venue template (``mmibc.figures.style.WIDTHS``).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve

from mmibc.evaluation.metrics import calibration, positive_probability
from mmibc.evaluation.stats import delong_ci
from mmibc.figures.style import CYCLE, PALETTE, apply_style, fig_width, save_figure

CONTROL_LABELS = {"none": "paired (as trained)", "drop_a": "drop stream a", "drop_b": "drop stream b", "shuffle_b": "shuffled partner"}
METHOD_LABELS = {"gradcam": "Grad-CAM", "gradxact": "Grad$\\times$act", "rollout": "Rollout", "xattn_cls": "Cross-attn.", "random": "Random map"}
XAI_PANELS = [("pointing", "Pointing\ngame", None), ("energy", "Energy in\nlesion", None), ("iou50", "IoU@0.5\n", None), ("insertion_auc", "Insertion AUC\n(higher is better)", None), ("deletion_auc", "Deletion AUC\n(lower is better)", None)]


def _finish(fig, out, formats=("pdf", "svg", "png")):
    if out is not None:
        save_figure(fig, out, formats)
    return fig


def _metrics(run: Path) -> dict:
    return json.loads((Path(run) / "metrics.json").read_text())


def _pred(run: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(Path(run) / f"predictions_{split}.csv")
    return df["y"].to_numpy(int), positive_probability(df)


def _name(s: str) -> str:
    return s.replace("_", " ")


# --------------------------------------------------------------------------------------------------- ROC
def roc_plot(runs: dict[str, Path], split: str = "test", template: str = "lncs", out=None, width: str = "single", ci: bool = True):
    """ROC curves with the DeLong AUROC interval; the filled marker is the operating point of the validation-chosen threshold."""
    apply_style(template)
    w = fig_width(template, width) * (0.72 if template == "lncs" else 1.0)
    fig, ax = plt.subplots(figsize=(w, w * 0.92))
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls=(0, (3, 2)), zorder=1)
    for i, (name, run) in enumerate(runs.items()):
        y, p = _pred(run, split)
        fpr, tpr, _ = roc_curve(y, p)
        auc, lo, hi = delong_ci(y, p)
        lab = f"{_name(name)}  {auc:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if ci else "")
        ax.plot(fpr, tpr, color=CYCLE[i % len(CYCLE)], label=lab, zorder=3)
        m = _metrics(run)["splits"][split]
        ax.plot(1 - m["specificity"], m["sensitivity"], "o", ms=3.6, color=CYCLE[i % len(CYCLE)], mec="white", mew=0.5, zorder=4)
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), xlabel="1 $-$ specificity", ylabel="Sensitivity", aspect="equal")
    ax.legend(loc="lower right", title="AUROC [95% CI]", title_fontsize=6.3, handlelength=1.4, borderaxespad=0.3)
    ax.grid(alpha=0.18, lw=0.4)
    return _finish(fig, out)


# --------------------------------------------------------------------------------------------------- ladder / forest
def ladder_plot(runs: dict[str, Path], split: str = "test", reference: str | None = None, template: str = "lncs", out=None, sort: bool = False):
    """One dot per run (AUROC with its DeLong interval): the fusion ladder, the backbone bake-off, the pairing ablation."""
    apply_style(template)
    rows = []
    for name, run in runs.items():
        m = _metrics(run)["splits"][split]
        lo, hi = m.get("auroc_delong_ci", m["ci"]["auroc"])
        rows.append((name, m["auroc"], lo, hi))
    if sort:
        rows.sort(key=lambda r: r[1])
    w = fig_width(template, "single")
    fig, ax = plt.subplots(figsize=(w, 0.24 * len(rows) + 0.75))
    ys = np.arange(len(rows))
    ref = next((r[1] for r in rows if r[0] == reference), None)
    for y, (name, a, lo, hi) in zip(ys, rows):
        col = PALETTE["red"] if name == reference else PALETTE["blue"]
        ax.plot([lo, hi], [y, y], color=col, lw=1.2, solid_capstyle="butt", zorder=2)
        ax.plot(a, y, "o", color=col, ms=4, mec="white", mew=0.5, zorder=3)
        ax.text(1.002, y, f"{a:.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=6.3)
    if ref is not None:
        ax.axvline(ref, color=PALETTE["red"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(ys, [_name(r[0]) for r in rows])
    ax.invert_yaxis()
    ax.set(xlabel=f"AUROC ({split}, 95% CI)")
    ax.grid(axis="x", alpha=0.18, lw=0.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return _finish(fig, out)


# --------------------------------------------------------------------------------------------------- controls
def controls_plot(run: Path, template: str = "lncs", out=None):
    """AUROC under the modality-reliance controls. A flat profile means the model ignores that stream / the pairing is uninformative."""
    apply_style(template)
    rows = _metrics(run).get("controls") or pd.read_csv(Path(run) / "controls.csv").to_dict("records")
    w = fig_width(template, "single")
    fig, ax = plt.subplots(figsize=(w, 0.3 * len(rows) + 0.8))
    for y, r in enumerate(rows):
        col = PALETTE["blue"] if r["control"] == "none" else PALETTE["orange"]
        ax.plot([r["auroc_lo"], r["auroc_hi"]], [y, y], color=col, lw=1.2, solid_capstyle="butt")
        ax.plot(r["auroc"], y, "o", color=col, ms=4, mec="white", mew=0.5, zorder=3)
        txt = f"{r['auroc']:.3f}" if r["control"] == "none" else f"{r['auroc']:.3f}  ($\\Delta$ {r['delta_auroc']:+.3f})"
        ax.text(1.002, y, txt, transform=ax.get_yaxis_transform(), va="center", fontsize=6.3)
    ax.axvline(0.5, color="#999999", lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(range(len(rows)), [CONTROL_LABELS.get(r["control"], _name(r["control"])) for r in rows])
    ax.invert_yaxis()
    ax.set(xlabel="Test AUROC (95% bootstrap CI)")
    ax.grid(axis="x", alpha=0.18, lw=0.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return _finish(fig, out)


# --------------------------------------------------------------------------------------------------- calibration
def calibration_plot(runs: dict[str, Path], split: str = "test", template: str = "lncs", out=None, bins: int = 10):
    apply_style(template)
    w = fig_width(template, "single") * (0.72 if template == "lncs" else 1.0)
    fig, (ax, hx) = plt.subplots(2, 1, figsize=(w, w * 1.15), gridspec_kw=dict(height_ratios=[3.2, 1], hspace=0.12), sharex=True)
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls=(0, (3, 2)))
    edges = np.linspace(0, 1, bins + 1)
    for i, (name, run) in enumerate(runs.items()):
        y, p = _pred(run, split)
        idx = np.minimum((p * bins).astype(int), bins - 1)
        xs = [p[idx == b].mean() for b in range(bins) if (idx == b).sum() >= 3]
        ys = [y[idx == b].mean() for b in range(bins) if (idx == b).sum() >= 3]
        c = calibration(y, p, bins)
        ax.plot(xs, ys, "o-", ms=3, color=CYCLE[i % len(CYCLE)], label=f"{_name(name)}  ECE {c['ece']:.3f}, Brier {c['brier']:.3f}")
        hx.hist(p, bins=edges, histtype="step", color=CYCLE[i % len(CYCLE)], lw=1.0)
    ax.set(ylabel="Observed malignant fraction", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", handlelength=1.4, borderaxespad=0.3)
    ax.grid(alpha=0.18, lw=0.4)
    hx.set(xlabel="Predicted malignant probability", ylabel="Count")
    return _finish(fig, out)


# --------------------------------------------------------------------------------------------------- explanation summary
def xai_summary_plot(run: Path, split: str = "test", template: str = "lncs", out=None, panels=XAI_PANELS):
    """Localisation and faithfulness per method with bootstrap CIs, against the random-map baseline (hollow grey)."""
    apply_style(template)
    df = pd.read_csv(Path(run) / f"xai_summary_{split}.csv")
    methods = [m for m in METHOD_LABELS if m in set(df["method"])]
    streams = sorted(df["stream"].unique())
    panels = [(k, t, _) for k, t, _ in panels if k in df and df[k].notna().any()]
    w = fig_width(template, "double")
    fig, axes = plt.subplots(1, len(panels), figsize=(w, 0.2 * len(methods) + 1.0), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (key, title, _) in zip(axes, panels):
        for j, m in enumerate(methods):
            for si, s in enumerate(streams):
                r = df[(df["method"] == m) & (df["stream"] == s)]
                if r.empty or np.isnan(r[key].iloc[0]):
                    continue
                r = r.iloc[0]
                y = j + (si - (len(streams) - 1) / 2) * 0.28
                rand = m == "random"
                col = "#8a8a8a" if rand else CYCLE[si]
                ax.plot([r[f"{key}_lo"], r[f"{key}_hi"]], [y, y], color=col, lw=1.0, solid_capstyle="butt")
                ax.plot(r[key], y, "o", ms=3.6, mfc="white" if rand else col, mec=col, mew=0.9, zorder=3)
        if key == "energy" and "chance_energy" in df:
            ax.axvline(float(df["chance_energy"].mean()), color="#999999", lw=0.6, ls=(0, (3, 2)))
        ax.set_title(title, fontsize=6.6)
        ax.grid(axis="x", alpha=0.18, lw=0.4)
        ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
    axes[0].set_yticks(range(len(methods)), [METHOD_LABELS[m] for m in methods])
    axes[0].invert_yaxis()
    handles = [Line2D([], [], marker="o", ls="", color=CYCLE[i], ms=3.6, label=f"stream {s}") for i, s in enumerate(streams)]
    fig.subplots_adjust(wspace=0.22, bottom=0.27)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.0), fontsize=6.3)
    return _finish(fig, out)


def overlay_panel(items: list[dict], template: str = "lncs", out=None, cmap: str = "magma", alpha: float = 0.5):
    """Per item two rows: the image with the lesion outline, then the explanation map over it.

    item = dict(image=HxW or HxWx3 in [0,1], cam=HxW, mask=HxW bool or None, title=str).  Maps are min-max scaled per image.
    Images are drawn at their stored resolution with Lanczos resampling and no JPEG step, so the panel stays sharp at 600 dpi.
    """
    from matplotlib import colors

    apply_style(template)
    n = len(items)
    w = fig_width(template, "double")
    fig, axes = plt.subplots(2, n, figsize=(w, w / n * 2.08), squeeze=False, gridspec_kw=dict(wspace=0.03, hspace=0.04))
    for j, it in enumerate(items):
        img = np.asarray(it["image"], float)
        img = img if img.ndim == 3 else np.stack([img] * 3, -1)
        cam = np.asarray(it["cam"], float)
        cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
        for r in range(2):
            ax = axes[r, j]
            ax.imshow(img, interpolation="lanczos")
            if r == 1:
                ax.imshow(cam, cmap=cmap, alpha=alpha, interpolation="bilinear", norm=colors.Normalize(0, 1))
            if it.get("mask") is not None:
                ax.contour(np.asarray(it["mask"], float), levels=[0.5], colors=[PALETTE["sky"]], linewidths=0.9)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0, j].set_title(it.get("title", ""), fontsize=6.6)
    axes[0, 0].set_ylabel("image", fontsize=6.6), axes[1, 0].set_ylabel("explanation", fontsize=6.6)
    return _finish(fig, out)


# --------------------------------------------------------------------------------------------------- data composition
def composition_plot(df: pd.DataFrame, template: str = "lncs", out=None, split_col: str = "split", group_col: str = "label", source_col: str | None = None):
    """Stacked counts per split and class (and, optionally, per source): the first thing a reviewer asks about a merged dataset."""
    apply_style(template)
    order = [s for s in ("train", "val", "test", "ext_test") if s in set(df[split_col])] or sorted(df[split_col].unique())
    groups = sorted(df[group_col].unique())
    w = fig_width(template, "single")
    fig, ax = plt.subplots(figsize=(w, 1.9))
    bottom = np.zeros(len(order))
    for i, g in enumerate(groups):
        cnt = np.array([int(((df[split_col] == s) & (df[group_col] == g)).sum()) for s in order])
        ax.bar(order, cnt, bottom=bottom, color=CYCLE[i % len(CYCLE)], label=str(g), width=0.62, edgecolor="white", lw=0.4)
        bottom += cnt
    for x, tot in enumerate(bottom):
        ax.text(x, tot, f"{int(tot)}", ha="center", va="bottom", fontsize=6.3)
    ax.set(ylabel="Images", ylim=(0, bottom.max() * 1.12))
    ax.legend(title=group_col, ncol=min(len(groups), 4), loc="upper right", fontsize=6.3, title_fontsize=6.3)
    return _finish(fig, out)

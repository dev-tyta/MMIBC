"""Publication figure style: vector output, embedded fonts, journal column widths, colour-blind-safe palette."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe & Ito (2008): distinguishable under the common forms of colour-vision deficiency, and in greyscale print.
PALETTE = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", sky="#56B4E9", purple="#CC79A7", yellow="#F0E442", black="#000000", grey="#7F7F7F")
CYCLE = [PALETTE[k] for k in ("blue", "orange", "green", "red", "sky", "purple", "grey")]

# text widths in inches: (single column, double column / full page)
WIDTHS = {"lncs": (4.80, 4.80), "ieee": (3.50, 7.16), "elsevier": (3.54, 7.48), "generic": (3.40, 7.00)}


def fig_width(template: str = "lncs", width: str = "single") -> float:
    s, d = WIDTHS[template]
    return s if width == "single" else d


def apply_style(template: str = "lncs", base_pt: float = 7.0) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": base_pt, "axes.titlesize": base_pt + 0.5, "axes.labelsize": base_pt, "xtick.labelsize": base_pt - 0.5, "ytick.labelsize": base_pt - 0.5,
        "legend.fontsize": base_pt - 0.5, "legend.frameon": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.1, "patch.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.prop_cycle": plt.cycler(color=CYCLE),
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",   # TrueType (Type 42) in PDF/PS: editable, no Type-3 bitmap fonts
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "image.interpolation": "lanczos",
    })


def save_figure(fig, stem: str | Path, formats=("pdf", "png"), dpi: int = 600) -> list[Path]:
    """Writes <stem>.pdf (vector), <stem>.svg (vector) and/or <stem>.png (raster, ``dpi``). PDF metadata carries no timestamp, so rebuilds are byte-stable."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        p = stem.with_suffix(f".{fmt}")
        kw = {"dpi": dpi} if fmt == "png" else {}
        if fmt == "pdf":
            kw["metadata"] = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}
        if fmt == "svg":
            kw["metadata"] = {"Date": None}
        fig.savefig(p, format=fmt, **kw)
        out.append(p)
    return out

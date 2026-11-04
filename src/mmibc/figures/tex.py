"""Build the TikZ paper figures (paper/figures/*.tex) into vector PDF, vector SVG and 600-dpi PNG.

The diagrams are LaTeX/TikZ so that their text is typeset by the paper's own fonts and maths. They need a TeX
installation (Colab: ``apt-get install texlive-latex-extra texlive-fonts-recommended poppler-utils``); the built
PDF / SVG / PNG are also committed under paper/figures/pdf so a machine without LaTeX can still use them.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from mmibc.figures.corrections import IDS, latex_table

DIAGRAMS = ("fig_architecture", "fig_algorithm")
FONTS = {"times": r"\usepackage{mathptmx}", "cm": ""}
TEXT_WIDTH_CM = 12.2  # Springer LNCS / MICCAI text width the fragments are drawn for


def fig_dir() -> Path:
    d = Path(__file__).resolve().parents[3] / "paper" / "figures"
    if not d.is_dir():
        raise FileNotFoundError(f"{d} not found: run from a source checkout (pip install -e .)")
    return d


def have_tex() -> bool:
    return all(shutil.which(t) for t in ("pdflatex", "pdftoppm", "pdftocairo", "pdffonts"))


def _run(cmd: list[str], cwd: Path) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode}):\n{(r.stdout + r.stderr)[-2500:]}")
    return r.stdout


def compile_tex(body: str, out_pdf: Path, font: str = "times", documentclass: str = r"\documentclass[border=3pt]{standalone}") -> Path:
    """Compile ``body`` (inside \\begin{document}) with the shared figure preamble to ``out_pdf``."""
    here = fig_dir()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "w.tex").write_text(
            f"{documentclass}\n\\usepackage[T1]{{fontenc}}\n{FONTS[font]}\n\\input{{{(here / 'mmibc_tikz').as_posix()}}}\n"
            f"\\begin{{document}}\n{body}\n\\end{{document}}\n")
        _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "w.tex"], tmp)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp / "w.pdf", out_pdf)
        (out_pdf.with_suffix(".log")).write_text((tmp / "w.log").read_text(errors="replace"))
    return out_pdf


def pdf_fonts(pdf: Path) -> list[dict]:
    rows = _run(["pdffonts", str(pdf)], pdf.parent).splitlines()[2:]
    out = []
    for ln in rows:
        m = re.match(r"^(\S+)\s+(.+?)\s{2,}\S+\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", ln)
        if m:
            out.append(dict(name=m.group(1), type=m.group(2).strip(), embedded=m.group(3) == "yes"))
    return out


def check_pdf_fonts(pdf: Path) -> None:
    fonts = pdf_fonts(pdf)
    bad = [f for f in fonts if not f["embedded"] or "Type 3" in f["type"]]
    if not fonts or bad:
        raise AssertionError(f"{pdf.name}: fonts must be embedded and not Type 3; found {fonts if not fonts else bad}")


def pdf_width_cm(pdf: Path) -> float:
    m = re.search(r"Page size:\s+([\d.]+) x", _run(["pdfinfo", str(pdf)], pdf.parent))
    return float(m.group(1)) / 72 * 2.54


def build_diagram(name: str, out_dir: Path | None = None, font: str = "times", formats=("pdf", "svg", "png"), dpi: int = 600) -> list[Path]:
    out_dir = Path(out_dir) if out_dir else fig_dir() / "pdf"
    pdf = compile_tex(f"\\input{{{(fig_dir() / name).as_posix()}}}%", out_dir / f"{name}.pdf", font)
    check_pdf_fonts(pdf)
    made = [pdf]
    if "svg" in formats:
        _run(["pdftocairo", "-svg", str(pdf), str(pdf.with_suffix(".svg"))], out_dir)
        made.append(pdf.with_suffix(".svg"))
    if "png" in formats:
        _run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(pdf), str(out_dir / name)], out_dir)
        made.append(out_dir / f"{name}.png")
    if "pdf" not in formats:
        pdf.unlink()
        made.remove(pdf)
    pdf.with_suffix(".log").unlink(missing_ok=True)
    return made


def write_corrections_table(directory: Path | None = None) -> list[Path]:
    d = Path(directory) if directory else fig_dir()
    (d / "tab_corrections.tex").write_text(latex_table(False))
    (d / "tab_corrections_evidence.tex").write_text(latex_table(True))
    return [d / "tab_corrections.tex", d / "tab_corrections_evidence.tex"]


def badge_ids(tex: Path) -> set[str]:
    """Correction badges used in a figure fragment (the legend's generic 'Cn' is not an id)."""
    return set(re.findall(r"\bC\d+\b", tex.read_text()))


def build_all(out_dir: Path | None = None, font: str = "times") -> list[Path]:
    write_corrections_table()
    made: list[Path] = []
    for n in DIAGRAMS:
        made += build_diagram(n, out_dir, font)
    return made


__all__ = ["DIAGRAMS", "IDS", "build_all", "build_diagram", "write_corrections_table", "check_pdf_fonts", "pdf_fonts", "pdf_width_cm", "badge_ids", "have_tex", "fig_dir", "compile_tex"]

"""The corrections key: what the rejected submission (v1) did and what this work (v2) does.

Single source of truth for (i) the Cn badges in paper/figures/fig_architecture.tex and fig_algorithm.tex and
(ii) the LaTeX table paper/figures/tab_corrections.tex (regenerate with ``mmibc figures --tex``).
Statements about v1 are limited to what was verified in the submitted code or in the MIWAI 2026 submission text.  "-" = no counterpart in v1.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class Correction(NamedTuple):
    cid: str
    component: str
    v1: str
    v2: str
    evidence: str


KEY: list[Correction] = [
    Correction("C1", "Splits", "sklearn train_test_split on files / images (not group-aware).",
               "Group-stratified by study, patient and near-duplicate cluster; official VinDr test untouched; assert_no_leakage.", "test_data, test_merge_audit"),
    Correction("C2", "Ultrasound data", "'normal' mapped to benign (label 0); duplicate and conflicting-label copies kept.",
               "Default binary_no_normal; dHash de-duplication, conflicts dropped; Doppler / caliper overlays inpainted; masks merged.", "test_data (planted BUSI defects)"),
    Correction("C3", "Pairing", "Random pairing of mammogram and ultrasound images that share a label (as described in the submission).",
               "Tiers: native CC-MLO; real mammogram-US (evaluation only); synthetic label-matched / size-conditioned; random and shuffled controls. Within-split; frozen 1-to-1 val / test pairs; train pairs re-drawn each epoch.", "test_data, test_merge_audit"),
    Correction("C4", "Merge audit", "No integrity checks.",
               "Schema, label mapping, cross-split duplicates, external-source isolation, class counts, file integrity, source-label confounding; errors stop the run.", "test_merge_audit"),
    Correction("C5", "Cross-attention", "Attention over a length-1 sequence: softmax = 1, output = v_proj(context), query ignored.",
               "Bidirectional multi-head cross-attention over patch-token sequences (pre-norm, FFN); query-dependence assertion.", "test_models (legacy no-op, query dependence)"),
    Correction("C6", "Backbone", "DINOv2 only; CLS vector only.",
               "Registry: DINOv2, DINOv2 with registers, DINOv3 (patch 16, 4 registers excluded from the grid), CNN baselines; dead mask_token frozen.", "test_models (token layout x 4 families)"),
    Correction("C7", "Fusion", "Single fusion variant.",
               "Ladder: unimodal, early, concat, GMU, x-attn, x-attn + GMU; late fusion computed at evaluation.", "test_models, test_benchmarks_suite"),
    Correction("C8", "Optimiser", "Multimodal trainer optimised fusion_classifier only.",
               "AdamW groups (head / backbone lr, no decay on norms and tokens); every trainable parameter must be in the optimiser.", "test_training (coverage assertion)"),
    Correction("C9", "Checkpoints", "Not resumable.",
               "Atomic checkpoints; exact resume (RNG restored); config and frozen-weight fingerprints.", "test_training (resumed run = uninterrupted)"),
    Correction("C10", "Loss / sampling", "Focal alpha was one scalar multiplier (no class weighting).",
               "Class-indexed alpha; balanced sampler; warning when both correct imbalance.", "test_training (focal)"),
    Correction("C11", "Evaluation", "Point estimates on a 117-image test set; 95% CIs for the unimodal rows only; checkpoint chosen by lowest validation loss.",
               "Threshold chosen on validation only; stratified bootstrap CIs; DeLong, exact McNemar, paired bootstrap; drop-a / drop-b / shuffled-partner controls.", "test_eval (DeLong calibrated under H0)"),
    Correction("C12", "Explanation", "Qualitative Grad-CAM overlays on each stream; no quantitative localisation or faithfulness evaluation.",
               "Grad-CAM on patch tokens through the fused logit; rollout; cross-attention maps; pointing, energy, IoU, deletion / insertion vs random.", "test_xai (oracle map, sanity check)"),
    Correction("C13", "Benchmarks", "Four-row comparison with works on other datasets and modalities (some private); no protocol-matched baselines.",
               "Literature table with protocol flags (no unread numbers); protocol-matched config; baselines on identical splits.", "test_benchmarks_suite"),
    Correction("C14", "Figures", "Raster PNG figures in the default plotting style (e.g. the Grad-CAM panel).",
               "Vector PDF / 600-dpi PNG, embedded fonts, journal column widths, colour-blind-safe palette.", "test_figures"),
]
IDS = [k.cid for k in KEY]

_SPECIALS = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "{": r"\{", "}": r"\}", "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
_SNAKE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+")


def tex_escape(s: str) -> str:
    """Plain text -> LaTeX. snake_case identifiers become \\texttt{}, ' x ' becomes a times sign, 'quoted' becomes ``quoted''."""
    s = re.sub(r"'([^']+)'", "\x00\\1\x01", s)                       # protect quotes from the escaper
    s = s.replace(" x ", " \x02 ").replace("CC-MLO", "CC--MLO").replace("mammogram-US", "mammogram--US")
    snake = {m.group(0) for m in _SNAKE.finditer(s)}
    for i, tok in enumerate(sorted(snake, key=len, reverse=True)):    # placeholders survive the escaper
        s = s.replace(tok, f"\x03{i}\x04")
    s = "".join(_SPECIALS.get(c, c) for c in s)
    for i, tok in enumerate(sorted(snake, key=len, reverse=True)):
        s = s.replace(f"\x03{i}\x04", r"\texttt{" + tok.replace("_", r"\_\allowbreak{}") + "}")
    return s.replace("\x00", "``").replace("\x01", "''").replace("\x02", r"$\times$")


def latex_table(with_evidence: bool = False) -> str:
    """A booktabs ``tabular`` (no float) sized for a 12.2 cm text width; wrap it in ``table`` and add the caption in the paper."""
    widths = ("0.55cm", "1.55cm", "3.15cm", "4.55cm") if not with_evidence else ("0.55cm", "1.45cm", "2.9cm", "3.35cm", "2.35cm")
    cols = "".join([r">{\raggedright\arraybackslash}p{%s}" % w if i else r"@{}l" for i, w in enumerate(widths)])
    head = ["ID", "Component", "Rejected submission (v1)", "This work (v2)"] + (["Evidence"] if with_evidence else [])
    lines = [r"% Generated by mmibc.figures.corrections.latex_table(); do not edit by hand.",
             r"% Needs \usepackage{booktabs,array} (loaded by mmibc_tikz.tex).",
             r"\begingroup\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.08}",
             r"\begin{tabular}{%s@{}}" % cols, r"\toprule", " & ".join(r"\textbf{%s}" % h for h in head) + r" \\ \midrule"]
    for k in KEY:
        v1 = r"\textit{no counterpart}" if k.v1 == "-" else tex_escape(k.v1)
        cells = [k.cid, tex_escape(k.component), v1, tex_escape(k.v2)] + ([tex_escape(k.evidence)] if with_evidence else [])
        lines.append(" & ".join(cells) + r" \\" + ("" if k is KEY[-1] else r"[1.5pt]"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup", ""]
    return "\n".join(lines)

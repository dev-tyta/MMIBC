"""Compare our runs with published results - honestly.

Every literature row states its dataset, label definition, split protocol and unit, and lists the settings our run must
satisfy to be called comparable (``requires``). A row is only ever shown as comparable when the run actually used those
settings; everything else is shown with its caveat. No number is invented: rows whose values were not retrieved stay empty.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LIT_PATH = Path(__file__).with_name("literature.csv")


def load_literature() -> pd.DataFrame:
    return pd.read_csv(LIT_PATH, dtype={"value": "float64"})


def _get(cfg, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def satisfies(cfg, requires: str) -> bool:
    """requires: ';'-joined tokens. 'a.b=c' must equal the config value; 'official_test' holds for VinDr runs (the official test split is never modified)."""
    for tok in filter(None, str(requires).split(";")):
        if tok == "official_test":
            continue  # guaranteed by assign_vindr_splits + the audit (official_test_changed)
        key, val = tok.split("=", 1)
        if str(_get(cfg, key)) != val:
            return False
    return True


def comparison_table(our_rows: list[dict], cfg, dataset_prefix: str = "VinDr") -> pd.DataFrame:
    """our_rows: [{'model':..., 'auroc':..., 'lo':..., 'hi':...}]. Returns one table with ours and the published rows for that dataset."""
    lit = load_literature()
    lit = lit[lit["dataset"].str.startswith(dataset_prefix) | lit["dataset"].str.contains(dataset_prefix, case=False)]
    rows = [dict(source="this work", model=r["model"], protocol=f"{cfg.task.kind}, labels={cfg.data.vindr.label_scheme}, official test split, patient-level splits",
                 auroc=r["auroc"], interval=f"{r['lo']:.3f}-{r['hi']:.3f} (95% CI, DeLong)", comparable_with_published="n/a", caveat="") for r in our_rows]
    for r in lit.itertuples():
        ok = (r.comparability in ("partial", "high")) and satisfies(cfg, r.requires)
        rows.append(dict(source=r.citation, model=r.model, protocol=f"{r.labels}; {r.split_protocol}", auroc=r.value, interval=str(r.spread) if isinstance(r.spread, str) else "",
                         comparable_with_published="yes, protocol matched" if ok else ("only if run with " + str(r.requires) if r.comparability in ("partial", "high") else "no"), caveat=r.caveat))
    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    def esc(s: str) -> str:
        return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")

    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}", r"\small", r"\begin{tabular}{p{0.34\linewidth}p{0.2\linewidth}rp{0.26\linewidth}}", r"\toprule",
             r"Method & Protocol & AUROC & Comparable \\", r"\midrule"]
    for r in df.itertuples():
        v = "--" if pd.isna(r.auroc) else f"{r.auroc:.3f}"
        lines.append(f"{esc(r.model)} & {esc(r.protocol)[:70]} & {v} & {esc(r.comparable_with_published)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

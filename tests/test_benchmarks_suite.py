from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmibc.benchmarks.compare import comparison_table, load_literature, satisfies, to_latex
from mmibc.evaluation.compare import compare_runs, late_fusion
from mmibc.suite import load_suite, run_suite
from mmibc.utils.config import load_config


def test_literature_rows_are_sourced_and_never_invented():
    lit = load_literature()
    assert lit["id"].is_unique and lit["url"].str.startswith("https://").all()
    # every 'requires' token must name a real config key, so a typo can never silently disable (or fake) comparability
    base = load_config("base.yaml")
    for req in lit["requires"].dropna():
        for tok in filter(None, req.split(";")):
            if tok != "official_test":
                assert satisfies(base, tok) in (True, False) and tok.split("=")[0].startswith("data.")
    got = lit[lit["value"].notna()]
    assert (got["status"] == "extracted").all() and got["spread"].notna().all()
    # rows whose numbers were not retrieved or not read must carry no value
    assert lit[lit["status"].isin(["not_read", "numbers_not_retrieved"])]["value"].isna().all()
    assert (lit["caveat"].fillna("").str.len() > 20).all()
    assert lit.loc[lit["id"] == "petrini25_twoview", "value"].item() == pytest.approx(0.8511)


def test_comparability_requires_matching_protocol():
    default = load_config("vindr_multiview.yaml")
    matched = load_config("protocol_petrini.yaml")
    assert not satisfies(default, "data.vindr.label_scheme=birads_3plus")
    assert satisfies(matched, "data.vindr.label_scheme=birads_3plus;official_test")
    ours = [dict(model="DINOv3-B xattn", auroc=0.9, lo=0.88, hi=0.92)]
    t_default, t_matched = comparison_table(ours, default), comparison_table(ours, matched)
    row = lambda t: t[t["model"].str.contains("two-view")].iloc[0]  # noqa: E731
    assert row(t_default)["comparable_with_published"].startswith("only if run with")
    assert row(t_matched)["comparable_with_published"] == "yes, protocol matched"
    assert "this work" in set(t_matched["source"]) and t_matched[t_matched["source"] == "this work"]["comparable_with_published"].iloc[0] == "n/a"
    tex = to_latex(t_matched, "Comparison with published results.", "tab:lit")
    assert tex.count(r"\begin{tabular}") == 1 and r"\toprule" in tex and "0.851" in tex
    never_matched = comparison_table(ours, matched, "BUSI")
    assert not (never_matched["comparable_with_published"] == "yes, protocol matched").any()


def test_compare_runs_and_late_fusion_validate_alignment():
    n = 3000                                                                       # large n: a single unlucky draw cannot flip the expected ordering
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.4).astype(int)
    mk = lambda s, seed: pd.DataFrame(dict(pair_id=[f"p{i}" for i in range(n)], y=y, p=1 / (1 + np.exp(-(np.random.default_rng(seed).normal(0, 1, n) + s * y)))))  # noqa: E731
    a, b = mk(1.2, 1), mk(1.2, 2)
    lf = late_fusion(a, b)
    assert lf["p"].between(0, 1).all()
    table = compare_runs({"a": a, "b": b, "late": lf}, {"a": 0.5, "b": 0.5, "late": 0.5}, "a")
    assert table.loc[table.model == "a", "delta_auroc"].isna().all() and (table["auroc_lo"] < table["auroc"]).all()
    assert table.loc[table.model == "late", "delta_auroc"].item() > 0                       # averaging two informative scorers helps
    with pytest.raises(ValueError, match="same pairs"):
        late_fusion(a, b.sample(frac=1.0, random_state=0).reset_index(drop=True))
    with pytest.raises(ValueError, match="same pairs"):
        compare_runs({"a": a, "b": b.iloc[::-1].reset_index(drop=True)}, {"a": 0.5, "b": 0.5}, "a")


def test_suite_files_parse_and_every_override_is_a_valid_config():
    for name in ("ladder", "backbones", "pairing"):
        spec = load_suite(name)
        assert spec["runs"] and spec["reference"] in {r["name"] for r in spec["runs"]}
        for r in spec["runs"]:
            cfg = load_config(spec["config"], r["set"])
            assert cfg.model.fusion and cfg.model.backbone


def test_run_suite_end_to_end_with_late_fusion(project, tmp_path):
    paths, over = project
    suite = tmp_path / "mini.yaml"
    suite.write_text("config: vindr_multiview.yaml\nreference: concat\ncontinue_on_error: true\nruns:\n"
                     "  - {name: unimodal_a, set: [model.fusion=unimodal_a]}\n  - {name: unimodal_b, set: [model.fusion=unimodal_b]}\n  - {name: concat, set: [model.fusion=concat]}\n"
                     "  - {name: broken, set: [model.backbone=dinov3_vitb16, model.pretrained=true]}\n")
    base = [*over, "model.shared_backbone=true", "model.backbone=dinov2_tiny_test", "model.pretrained=false", "model.freeze=none", "model.xattn_heads=2",
            "model.head_hidden=32", "training.epochs=1", "input.image_size=56", "eval.controls=[none]", "eval.bootstrap_samples=20", "data.store_long_side=112"]
    res = run_suite(suite, paths, base + ["training.batch_size=8"])
    ok = res[res["auroc"].notna()]
    assert {"unimodal_a", "unimodal_b", "concat", "late_fusion"} <= set(ok["model"])
    assert ok.loc[ok.model == "concat", "delta_auroc"].isna().all()                 # the reference row
    failed = res[res["error"].notna()] if "error" in res else pd.DataFrame()
    assert len(failed) == 1 and failed.iloc[0]["model"] == "broken"                  # no network / gated: recorded, suite continued
    assert (paths.runs / "suite_mini" / "results.csv").exists()

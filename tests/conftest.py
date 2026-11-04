from __future__ import annotations

import pytest

from mmibc.data.synthetic import make_fake_busi, make_fake_extra_us, make_fake_vindr


@pytest.fixture(scope="session")
def raw_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    make_fake_vindr(root, n_studies=40)
    make_fake_busi(root)
    make_fake_extra_us(root)
    return root


@pytest.fixture(scope="session")
def vindr_manifest(raw_root, tmp_path_factory):
    from mmibc.data.vindr import prepare_vindr

    out = tmp_path_factory.mktemp("proc_vindr")
    m = prepare_vindr(raw_root / "vindr_images", raw_root / "vindr_annotations", out, long_side=112, workers=2)
    return out, m


@pytest.fixture(scope="session")
def us_manifest(raw_root, tmp_path_factory):
    import pandas as pd

    from mmibc.data.ultrasound import prepare_ultrasound, scan_busi, scan_class_folders

    out = tmp_path_factory.mktemp("proc_us")
    scan = pd.concat([scan_busi(raw_root / "busi"), scan_class_folders(raw_root / "extra_us", "extra_us")], ignore_index=True)
    m = prepare_ultrasound(scan, out, "binary_no_normal", long_side=112, workers=2)
    return out, m


@pytest.fixture(scope="session")
def project(raw_root, tmp_path_factory):
    """A prepared project (fake raw data -> processed) under a temporary Drive/local root. Returns (paths, extra overrides)."""
    import shutil

    from mmibc.pipeline import prepare
    from mmibc.utils.config import load_config
    from mmibc.utils.paths import ProjectPaths

    root = tmp_path_factory.mktemp("proj_shared")
    paths = ProjectPaths(root / "drive", root / "local", raw_on_drive=False).ensure()
    for name in ("vindr_images", "vindr_annotations", "busi", "extra_us"):
        shutil.copytree(raw_root / name, paths.raw / name)
    over = ["audit.min_per_class_split=1", "training.num_workers=0", "data.us_sources=[busi,extra_us]", "data.us_external_source=extra_us"]
    prepare(load_config("smoke.yaml", over), paths)
    return paths, over


# ------------------------------------------------------------------------------------ synthetic run directories (figure tests)
def make_fake_run(root, name: str, sep: float = 1.4, seed: int = 0, n: int = 260, with_xai: bool = True):
    """A run directory with the artifacts ``evaluate_model`` / ``explain_split`` write, computed with the real metric code."""
    import json

    import numpy as np
    import pandas as pd

    from mmibc.evaluation.evaluate import _split_metrics
    from mmibc.evaluation.metrics import choose_threshold

    rng = np.random.default_rng(seed)
    run = root / name
    run.mkdir(parents=True)
    preds = {}
    for s in ("val", "test"):
        y = (rng.random(n) < 0.35).astype(int)
        p = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + sep * (y - 0.35))))
        preds[s] = pd.DataFrame(dict(pair_id=[f"a{i}|b{i}" for i in range(n)], uid_a=[f"a{i}" for i in range(n)], uid_b=[f"b{i}" for i in range(n)], y=y, p0=1 - p, p1=p, p=p, loss=0.5))
        preds[s].to_csv(run / f"predictions_{s}.csv", index=False)
    thr = choose_threshold(preds["val"]["y"].to_numpy(), preds["val"]["p"].to_numpy(), "youden")
    out = dict(threshold=thr, threshold_method="youden", threshold_chosen_on="val", splits={s: _split_metrics(d, thr, 100) for s, d in preds.items()})
    rows = []
    for c, shift in (("none", 0.0), ("drop_a", -0.08), ("drop_b", -0.02), ("shuffle_b", -0.03)):
        a = out["splits"]["test"]["auroc"] + shift
        rows.append(dict(control=c, auroc=a, auroc_lo=a - 0.03, auroc_hi=a + 0.03, delta_auroc=shift, sensitivity=0.8, specificity=0.8))
    out["controls"] = rows
    pd.DataFrame(rows).to_csv(run / "controls.csv", index=False)
    (run / "metrics.json").write_text(json.dumps(out, default=float))
    if with_xai:
        recs = []
        for m, base in (("gradcam", 0.62), ("gradxact", 0.55), ("rollout", 0.40), ("xattn_cls", 0.45), ("random", 0.12)):
            for s in ("a", "b"):
                rec = dict(stream=s, method=m, n=40, n_annotated=30, chance_energy=0.1)
                for k, off in (("pointing", 0.0), ("energy", -0.3), ("iou50", -0.35), ("insertion_auc", 0.1), ("deletion_auc", -0.2)):
                    v = float(np.clip(base + off + (0.04 if s == "b" else 0), 0.02, 0.95))
                    rec.update({k: v, f"{k}_lo": v - 0.07, f"{k}_hi": v + 0.07})
                recs.append(rec)
        pd.DataFrame(recs).to_csv(run / "xai_summary_test.csv", index=False)
    return run

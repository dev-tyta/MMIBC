"""Run the explanation methods over a split and score them against lesion annotations."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from mmibc.xai.explain import cross_attention_cam, deletion_insertion, rollout_cam, to_input_size, token_cam
from mmibc.xai.ground_truth import GroundTruth
from mmibc.xai.metrics import bootstrap_mean, energy_in_mask, iou_at, pointing_game

log = logging.getLogger(__name__)
METRICS = ("pointing", "energy", "iou50", "deletion_auc", "insertion_auc")


def _streams(model) -> list[str]:
    return {"unimodal_a": ["a"], "unimodal_b": ["b"], "early": ["a"]}.get(model.fusion, ["a", "b"])


def explain_split(model, ds, gt: GroundTruth, cfg, run_dir: Path, device: torch.device, split: str = "test", n_save: int = 8) -> pd.DataFrame:
    run_dir = Path(run_dir)
    x = cfg.xai
    model.to(device).eval()
    streams = _streams(model)
    # prefer pairs that have lesion annotations in at least one stream
    has_gt = [gt.mask(r.uid_a) is not None or gt.mask(r.uid_b) is not None for r in ds.pairs.itertuples()]
    idx = [i for i, g in enumerate(has_gt) if g][: int(x.n_samples)]
    if not idx:
        log.warning("no annotated pairs in '%s': reporting faithfulness only", split)
        idx = list(range(min(int(x.n_samples), len(ds))))
    loader = DataLoader(Subset(ds, idx), batch_size=4, shuffle=False, num_workers=0)
    rng = torch.Generator().manual_seed(int(cfg.seed))
    rows, saved, rollout_ok = [], 0, True
    (run_dir / "cams").mkdir(exist_ok=True)
    for batch in loader:
        a, b, y = batch["a"].to(device), batch["b"].to(device), batch["y"]
        with torch.no_grad():
            logits = model(a, b)
        pred = logits.argmax(1)
        for s in streams:
            maps = {m: token_cam(model, a, b, s, pred, m)["cam"].detach() for m in ("gradcam", "gradxact")}
            if model.fusion.startswith("xattn"):
                maps["xattn_cls"] = cross_attention_cam(model, a, b, query_stream="b" if s == "a" else "a")
            bb, xin = getattr(model, f"backbone_{s}"), (a if s == "a" else b)
            if rollout_ok and getattr(bb, "spec", None) is not None and bb.spec.family != "tv":
                try:
                    maps["rollout"] = rollout_cam(bb, xin)
                except ValueError:
                    rollout_ok = False
                    log.warning("rollout skipped: model was not built with eager attention")
            gh, gw = maps["gradcam"].shape[-2:]
            maps["random"] = torch.rand(maps["gradcam"].shape, generator=rng).to(device)
            uids = batch["uid_a"] if s == "a" else batch["uid_b"]
            for name, grid in maps.items():
                full = to_input_size(grid, tuple(xin.shape[-2:]))
                dele, ins = deletion_insertion(model, a, b, grid, s, pred, int(x.steps))
                for j in range(a.shape[0]):
                    m = gt.mask(uids[j]) if model.fusion != "early" else None
                    rec = dict(split=split, pair_id=batch["pair_id"][j], uid=uids[j], stream=s, method=name, y=int(y[j]), pred=int(pred[j]), deletion_auc=float(dele[j]), insertion_auc=float(ins[j]),
                               pointing=np.nan, energy=np.nan, iou50=np.nan, lesion_frac=np.nan)
                    if m is not None:
                        rec.update(pointing=pointing_game(full[j], m), energy=energy_in_mask(full[j], m), iou50=iou_at(full[j], m, float(x.cam_threshold)), lesion_frac=float(m.mean()))
                    rows.append(rec)
                    if saved < n_save and name != "random":
                        np.save(run_dir / "cams" / f"{batch['pair_id'][j].replace('|', '_')}_{s}_{name}.npy", full[j].astype(np.float16))
            saved += a.shape[0] if s == streams[-1] else 0
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / f"xai_samples_{split}.csv", index=False)
    summ = []
    for (s, m), g in df.groupby(["stream", "method"]):
        rec = dict(stream=s, method=m, n=int(len(g)), n_annotated=int(g["pointing"].notna().sum()), chance_energy=float(g["lesion_frac"].mean()))
        for k in METRICS:
            mean, lo, hi = bootstrap_mean(g[k].to_numpy(), n=500)
            rec.update({k: mean, f"{k}_lo": lo, f"{k}_hi": hi})
        summ.append(rec)
    out = pd.DataFrame(summ)
    out.to_csv(run_dir / f"xai_summary_{split}.csv", index=False)
    (run_dir / f"xai_summary_{split}.json").write_text(json.dumps(out.to_dict("records"), indent=2, default=float))
    return out

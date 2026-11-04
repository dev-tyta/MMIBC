from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from mmibc.evaluation.metrics import auprc, auroc, multiclass_summary
from mmibc.training.checkpoint import EarlyStopping, atomic_save, load_model_state, model_state
from mmibc.training.ema import EMA
from mmibc.training.losses import build_loss
from mmibc.training.optim import assert_optimizer_covers, build_optimizer, build_scheduler

log = logging.getLogger(__name__)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_settings(mode: str, device: torch.device) -> tuple[torch.dtype | None, bool]:
    """-> (autocast dtype or None, use GradScaler). 'auto': bf16 where supported, else fp16 on CUDA, else off."""
    if device.type != "cuda" or mode == "off":
        return None, False
    if mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16, False
    return torch.float16, True


def _loader(ds, tcfg, device, shuffle=False, sampler=None, drop_last=False, seed=0) -> DataLoader:
    nw = int(tcfg.num_workers)
    return DataLoader(
        ds, batch_size=int(tcfg.batch_size), shuffle=shuffle and sampler is None, sampler=sampler, drop_last=drop_last, num_workers=nw,
        pin_memory=device.type == "cuda", persistent_workers=False, generator=torch.Generator().manual_seed(seed),
    )


def apply_drop(a: torch.Tensor, b: torch.Tensor, drop: str | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a whole stream by the neutral input (zeros = the dataset-mean image after normalisation)."""
    if drop == "a":
        a = torch.zeros_like(a)
    elif drop == "b":
        b = torch.zeros_like(b)
    elif drop is not None:
        raise ValueError(f"drop must be None, 'a' or 'b', got {drop!r}")
    return a, b


@torch.no_grad()
def predict(model: nn.Module, ds, tcfg, device: torch.device, drop: str | None = None) -> pd.DataFrame:
    """Probabilities for every pair, in dataset order. One row per pair; deterministic (eval mode, no shuffling)."""
    was_training = model.training
    model.eval()
    dtype, _ = amp_settings(tcfg.amp, device)
    rows: list[pd.DataFrame] = []
    for batch in _loader(ds, tcfg, device):
        a, b = apply_drop(batch["a"].to(device), batch["b"].to(device), drop)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(a, b)
        logits = logits.float().cpu()
        probs = torch.softmax(logits, -1).numpy()
        df = pd.DataFrame(dict(pair_id=batch["pair_id"], uid_a=batch["uid_a"], uid_b=batch["uid_b"], y=batch["y"].numpy()))
        for k in range(probs.shape[1]):
            df[f"p{k}"] = probs[:, k]
        df["p"] = probs[:, 1] if probs.shape[1] == 2 else np.nan
        if probs.shape[1] == 2:
            df["logit"] = (logits[:, 1] - logits[:, 0]).numpy()  # exact log-odds (p saturates in float32); used for recalibration
        df["loss"] = nn.functional.cross_entropy(logits, batch["y"], reduction="none").numpy()
        rows.append(df)
    model.train(was_training)
    return pd.concat(rows, ignore_index=True)


def summarize_predictions(df: pd.DataFrame) -> dict[str, float]:
    y = df["y"].to_numpy()
    out = dict(loss=float(df["loss"].mean()))
    pcols = [c for c in df.columns if c.startswith("p") and c[1:].isdigit()]
    if len(pcols) == 2:
        p = df["p"].to_numpy()
        out.update(auroc=auroc(y, p), auprc=auprc(y, p), acc=float(((p >= 0.5).astype(int) == y).mean()))
    else:
        m = multiclass_summary(y, df[pcols].to_numpy())
        out.update(auroc=m["macro_auroc_ovr"], acc=m["accuracy"], macro_f1=m["macro_f1"])
    return out


# Options added after the first v2 runs. While they sit at their default they are left out of the fingerprint, so runs trained
# (or half trained) before they existed keep their fingerprint and can still be resumed, skipped when finished, and re-evaluated.
_ADDED_DEFAULTS: dict[str, dict] = {
    "input": {"image_size_a": None, "image_size_b": None},
    "data": {"split_seed": None, "mammo_source": "vindr", "cmmd": {"subsets": ["D1", "D2"], "val_fraction": 0.15, "test_fraction": 0.15, "crop_breast": True}},
    "model": {"drop_path": 0.0},
    "training": {"label_smoothing": 0.0, "ema_decay": 0.0, "layer_decay": 1.0},
}


def config_fingerprint(cfg) -> str:
    keys = {k: cfg[k] for k in ("model", "task", "input", "data")}
    t = {k: v for k, v in dict(cfg["training"]).items() if k not in ("epochs", "resume", "num_workers", "patience")}
    groups = {k: dict(v) if hasattr(v, "keys") else v for k, v in keys.items()}
    groups["training"] = t
    for grp, defaults in _ADDED_DEFAULTS.items():
        for k, dv in defaults.items():
            if k in groups[grp]:
                cur = groups[grp][k]
                from omegaconf import OmegaConf

                if (OmegaConf.to_container(cur, resolve=True) if OmegaConf.is_config(cur) else cur) == dv:
                    del groups[grp][k]
    blob = json.dumps({**groups, "seed": cfg["seed"]}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _rng_state() -> dict:
    s = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), py=random.getstate())
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_state(s: dict) -> None:
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    random.setstate(s["py"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


class Trainer:
    def __init__(self, model: nn.Module, cfg, run_dir: Path, train_ds, val_ds, device: torch.device | None = None, n_classes: int = 2):
        self.model, self.cfg, self.t = model, cfg, cfg.training
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.train_ds, self.val_ds, self.n_classes = train_ds, val_ds, n_classes
        self.device = device or get_device()
        self.model.to(self.device)
        counts = np.bincount(np.asarray(train_ds.targets), minlength=n_classes)
        if (counts == 0).any():
            raise ValueError(f"training pairs lack a class entirely: counts={counts.tolist()}")
        self.class_counts = counts
        self.loss_fn = build_loss(self.t, counts, self.device)
        self.opt = build_optimizer(model, self.t)
        bs, accum = int(self.t.batch_size), int(self.t.grad_accum_steps)
        self.drop_last = len(train_ds) > bs
        n_batches = len(train_ds) // bs if self.drop_last else 1
        self.steps_per_epoch = max(1, math.ceil(n_batches / accum))
        self.sched = build_scheduler(self.opt, self.t, self.steps_per_epoch)
        self.amp_dtype, use_scaler = amp_settings(self.t.amp, self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        self.stopper = EarlyStopping(self.t.monitor_mode, int(self.t.patience), float(self.t.min_delta))
        ema_decay = float(self.t.get("ema_decay", 0.0))
        self.ema = EMA(self.model, ema_decay) if ema_decay > 0 else None
        self.history: list[dict] = []
        self.start_epoch = 0
        self.finished = False
        self.fp = config_fingerprint(cfg)
        self.stop_after_epoch: int | None = None  # tests / debugging: simulate a Colab disconnect

    # ------------------------------------------------------------------ io
    @property
    def last_path(self) -> Path:
        return self.run_dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.run_dir / "best.pt"

    def _save_last(self, epoch: int) -> None:
        atomic_save(dict(
            model=model_state(self.model, bool(self.t.save_trainable_only) and bool(self.cfg.model.pretrained)), opt=self.opt.state_dict(),
            sched=self.sched.state_dict(), scaler=self.scaler.state_dict(), stopper=self.stopper.state_dict(), history=self.history,
            epoch=epoch, finished=self.finished, fp=self.fp, rng=_rng_state(), ema=self.ema.state_dict() if self.ema else None,
        ), self.last_path)

    def _try_resume(self) -> None:
        if not (self.t.resume and self.last_path.exists()):
            return
        blob = torch.load(self.last_path, map_location="cpu", weights_only=False)
        if blob["fp"] != self.fp:
            raise RuntimeError(f"{self.last_path} was written by a different configuration ({blob['fp']} != {self.fp}). Use a new run_name, or delete the run directory.")
        load_model_state(self.model, blob["model"])
        self.opt.load_state_dict(blob["opt"])
        self.sched.load_state_dict(blob["sched"])
        self.scaler.load_state_dict(blob["scaler"])
        self.stopper.load_state_dict(blob["stopper"])
        if self.ema is not None and blob.get("ema"):
            self.ema.load_state_dict(blob["ema"])
        self.history, self.start_epoch, self.finished = blob["history"], blob["epoch"] + 1, blob["finished"]
        _set_rng_state(blob["rng"])
        log.info("Resumed from epoch %d (%s)", blob["epoch"], self.last_path)

    def _eval_weights(self):
        """Context in which the model holds the weights that get validated and saved: the EMA copy when enabled, else the live weights."""
        return self.ema.applied(self.model) if self.ema is not None else nullcontext()

    # ------------------------------------------------------------------ loop
    def _train_loader(self, epoch: int) -> DataLoader:
        self.train_ds.set_epoch(epoch)  # synthetic pairs: fresh partner draw
        seed = int(self.cfg.seed) * 1000 + epoch
        sampler = None
        if self.t.sampler == "balanced":
            tg = np.asarray(self.train_ds.targets)
            w = 1.0 / np.clip(np.bincount(tg, minlength=self.n_classes)[tg], 1, None)
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(self.train_ds), replacement=True, generator=torch.Generator().manual_seed(seed))
        return _loader(self.train_ds, self.t, self.device, shuffle=True, sampler=sampler, drop_last=self.drop_last, seed=seed)

    def _modality_dropout(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = float(self.t.get("modality_dropout", 0.0))
        if p <= 0 or self.model.fusion.startswith("unimodal") or self.model.fusion == "early":
            return a, b
        u = torch.rand(a.shape[0], device=a.device)
        drop_a, drop_b = (u < p / 2).float()[:, None, None, None], ((u >= p / 2) & (u < p)).float()[:, None, None, None]
        return a * (1 - drop_a), b * (1 - drop_b)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        accum = int(self.t.grad_accum_steps)
        total, n, t0 = 0.0, 0, time.time()
        self.opt.zero_grad(set_to_none=True)
        loader = self._train_loader(epoch)
        for i, batch in enumerate(tqdm(loader, desc=f"epoch {epoch + 1}/{self.t.epochs}", leave=False, disable=None)):
            a, b, y = batch["a"].to(self.device, non_blocking=True), batch["b"].to(self.device, non_blocking=True), batch["y"].to(self.device)
            a, b = self._modality_dropout(a, b)
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                logits = self.model(a, b)
            loss = self.loss_fn(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1} step {i}. Lower training.lr_head / lr_backbone or set training.amp=off.")
            self.scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(loader):
                if self.t.grad_clip:
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), float(self.t.grad_clip))
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)
                if not isinstance(self.sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.sched.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        self._epoch_seconds = time.time() - t0
        return total / max(1, n)

    def fit(self) -> dict:
        assert_optimizer_covers(self.model, self.opt)  # catches layers unfrozen after the optimizer was built
        self._try_resume()
        if self.finished:
            log.info("Run already finished; loading the best checkpoint.")
        for epoch in range(self.start_epoch, int(self.t.epochs)):
            if self.finished:
                break
            tr_loss = self._train_epoch(epoch)
            with self._eval_weights():
                preds = predict(self.model, self.val_ds, self.t, self.device)
            val = summarize_predictions(preds)
            rec = dict(epoch=epoch + 1, train_loss=tr_loss, **{f"val_{k}": v for k, v in val.items()}, lr_head=self.opt.param_groups[-1]["lr"], seconds=self._epoch_seconds)
            if self.t.monitor not in rec:
                raise KeyError(f"training.monitor='{self.t.monitor}' is not a metric; available: {sorted(rec)}")
            if isinstance(self.sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.sched.step(rec[self.t.monitor])
            self.history.append(rec)
            pd.DataFrame(self.history).to_csv(self.run_dir / "metrics.csv", index=False)
            improved = self.stopper.step(rec[self.t.monitor])
            if improved:
                with self._eval_weights():
                    atomic_save(dict(model=model_state(self.model, bool(self.t.save_trainable_only) and bool(self.cfg.model.pretrained)), epoch=epoch + 1, metric=rec[self.t.monitor], fp=self.fp), self.best_path)
                preds.to_csv(self.run_dir / "val_predictions_best.csv", index=False)
            log.info("epoch %d  train_loss %.4f  val_loss %.4f  val_auroc %.4f%s", epoch + 1, tr_loss, rec["val_loss"], rec.get("val_auroc", float("nan")), "  *" if improved else "")
            self.finished = self.stopper.should_stop or (epoch + 1) == int(self.t.epochs)
            self._save_last(epoch)
            if self.stop_after_epoch is not None and epoch + 1 >= self.stop_after_epoch and not self.finished:
                return dict(interrupted=True, epoch=epoch + 1)
        if not self.best_path.exists():
            raise RuntimeError(f"no usable checkpoint: the monitored metric '{self.t.monitor}' was NaN in every epoch (validation set lacks a class?)")
        best = torch.load(self.best_path, map_location="cpu", weights_only=False)
        load_model_state(self.model, best["model"])
        summary = dict(best_epoch=best["epoch"], best_metric=best["metric"], monitor=self.t.monitor, epochs_run=len(self.history), stopped_early=len(self.history) < int(self.t.epochs), config_fingerprint=self.fp, ema=self.ema is not None)
        (self.run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

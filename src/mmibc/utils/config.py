"""YAML configuration with dotted command-line overrides (OmegaConf)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf


def config_dir() -> Path:
    env = os.environ.get("MMIBC_CONFIG_DIR")
    if env:
        return Path(env)
    # <repo>/src/mmibc/utils/config.py -> <repo>/configs  (editable install / source checkout)
    d = Path(__file__).resolve().parents[3] / "configs"
    if not (d / "base.yaml").exists():
        raise FileNotFoundError(
            f"Could not find the configs directory (looked for {d / 'base.yaml'}). The configs live in the repository, not in the "
            "installed package: install from a clone with `pip install -e .` (or `pip install -e \".[download]\"`), "
            "or point the MMIBC_CONFIG_DIR environment variable at the repository's configs/ folder."
        )
    return d


def _resolve(path: str | os.PathLike) -> Path:
    p = Path(path)
    if p.exists():
        return p
    for cand in (config_dir() / p, config_dir() / f"{p}.yaml"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Config '{path}' not found (looked in cwd and {config_dir()})")


def load_config(path: str | os.PathLike | None = None, overrides: Iterable[str] = ()) -> DictConfig:
    cfg = OmegaConf.load(config_dir() / "base.yaml")
    if path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(_resolve(path)))
    overrides = [o for o in overrides if o]
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]


def save_config(cfg: DictConfig, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]

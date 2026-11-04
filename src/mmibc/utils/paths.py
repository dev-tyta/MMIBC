"""Where things live, on Colab and elsewhere.

Layout (``root`` is on Google Drive so it survives runtime resets; ``local`` is Colab's fast disk):

    <root>/raw/<dataset>/         raw downloads (default; ``paths.raw_on_drive``)  (persistent)
    <root>/processed/<dataset>/   cropped + downscaled images and manifest.csv   (persistent)
    <root>/splits/                frozen split / pair tables                      (persistent)
    <root>/runs/<run_name>/       checkpoints, predictions, metrics, config       (persistent)
    <root>/cache/hf/              Hugging Face weights cache                      (persistent)
    <local>/raw/<dataset>/        raw downloads when ``paths.raw_on_drive=false``  (ephemeral)
    <local>/processed/<dataset>/  staged copy of processed data for fast reading  (ephemeral)
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DRIVE_MOUNT = "/content/drive"
DRIVE_DEFAULT_ROOT = f"{DRIVE_MOUNT}/MyDrive/MMIBC"
COLAB_LOCAL = "/content/mmibc_local"


def in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules


def mount_drive(mount_point: str = DRIVE_MOUNT) -> bool:
    """Mount Google Drive when running in Colab. Returns True if Drive is available. No-op elsewhere."""
    if not in_colab():
        return False
    if Path(mount_point, "MyDrive").exists():
        return True
    try:
        from google.colab import drive  # type: ignore

        drive.mount(mount_point)  # only works inside a notebook cell, not from a `!mmibc ...` subprocess
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not mount Google Drive from this process: %s", exc)
    return Path(mount_point, "MyDrive").exists()


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    local: Path
    raw_on_drive: bool = True

    @property
    def raw(self) -> Path:
        return (self.root if self.raw_on_drive else self.local) / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def local_processed(self) -> Path:
        return self.local / "processed"

    @property
    def splits(self) -> Path:
        return self.root / "splits"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def hf_cache(self) -> Path:
        return self.root / "cache" / "hf"

    def ensure(self) -> "ProjectPaths":
        for p in (self.root, self.local, self.raw, self.processed, self.local_processed, self.splits, self.runs, self.hf_cache):
            p.mkdir(parents=True, exist_ok=True)
        return self


def resolve_paths(root: str | os.PathLike | None = None, local: str | os.PathLike | None = None, raw_on_drive: bool = True) -> ProjectPaths:
    """Resolve persistent root and local scratch.

    Root precedence: explicit argument > $MMIBC_ROOT > mounted Drive (Colab) > ./workspace.
    """
    root = root or os.environ.get("MMIBC_ROOT")
    if not root:
        if in_colab():
            if not mount_drive():
                raise RuntimeError(
                    "Google Drive is not mounted, so nothing could be saved to it (falling back to the temporary Colab disk would "
                    "lose your data when the runtime resets). Run this in a notebook cell, then try again:\n"
                    "    from google.colab import drive; drive.mount('/content/drive')\n"
                    "or pass an explicit location with --root / the MMIBC_ROOT environment variable."
                )
            root = DRIVE_DEFAULT_ROOT
        else:
            root = Path.cwd() / "workspace"
    root = Path(root).expanduser().resolve()

    local = local or os.environ.get("MMIBC_LOCAL")
    if not local:
        local = COLAB_LOCAL if in_colab() else root.parent / (root.name + "_local")
    local = Path(local).expanduser().resolve()
    return ProjectPaths(root=root, local=local, raw_on_drive=raw_on_drive).ensure()


def paths_from_cfg(cfg) -> ProjectPaths:
    p = cfg.paths
    paths = resolve_paths(p.get("root"), p.get("local"), bool(p.get("raw_on_drive", True)))
    # Keep HF weights on Drive so a fresh Colab runtime does not re-download them.
    os.environ.setdefault("HF_HOME", str(paths.hf_cache))
    os.environ.setdefault("TORCH_HOME", str(paths.root / "cache" / "torch"))  # torchvision CNN weights
    return paths


def _signature(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def stage_to_local(src: Path, dst: Path, signature_file: str = "manifest.csv") -> Path:
    """Copy a processed dataset from Drive to local disk once per runtime (Drive reads are slow).

    Skipped when ``dst`` already holds a copy made from the same manifest.
    """
    src, dst = Path(src), Path(dst)
    sig_path = src / signature_file
    sig = _signature(sig_path) if sig_path.exists() else "nosig"
    marker = dst / ".staged"
    if marker.exists() and marker.read_text().strip() == sig:
        log.info("Local copy of %s is current (%s)", src.name, sig)
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    log.info("Staging %s -> %s ...", src, dst)
    if shutil.which("rsync"):
        subprocess.run(["rsync", "-a", f"{src}/", f"{dst}/"], check=True)
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    marker.write_text(sig)
    return dst

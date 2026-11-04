"""Transforms and torch datasets. Everything here is picklable (works with spawn-based DataLoader workers)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from mmibc.data.preprocess import Affine, as_hw, letterbox, letterbox_params

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageTransform:
    """Grayscale PIL -> normalised 3xSxS tensor. Letterboxed (aspect preserved), optional CLAHE, optional light augmentation."""

    def __init__(self, size, train: bool = False, clahe: bool = False, aug: dict | None = None):
        self.size, self.train, self.clahe = as_hw(size), train, clahe  # (H, W)
        aug = dict(aug or {})
        self.aug = None
        if train:
            ops: list = []
            if aug.get("affine_degrees", 0) or aug.get("translate", 0) or aug.get("scale"):
                ops.append(T.RandomAffine(degrees=aug.get("affine_degrees", 0), translate=(aug.get("translate", 0),) * 2, scale=tuple(aug.get("scale", (1, 1))), fill=0))
            if aug.get("brightness", 0) or aug.get("contrast", 0):
                ops.append(T.ColorJitter(brightness=aug.get("brightness", 0), contrast=aug.get("contrast", 0)))
            if aug.get("hflip", False):
                ops.append(T.RandomHorizontalFlip())
            self.aug = T.Compose(ops) if ops else None
        self.to_tensor = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("L")
        if self.clahe:
            import cv2

            img = Image.fromarray(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(np.asarray(img)))
        if self.aug is not None:
            img = self.aug(img)
        img, _ = letterbox(img, self.size)
        return self.to_tensor(img.convert("RGB"))


def stream_size(cfg, stream: str) -> tuple[int, int]:
    """Model-input size (H, W) of stream 'a' or 'b'.

    ``input.image_size`` is the default; ``input.image_size_a`` / ``image_size_b`` override it per stream (an int or [H, W]), so a
    mammogram can be read at high resolution while the ultrasound stays small. In the native multi-view task both streams are
    mammograms and share the stream-a size.
    """
    if stream not in ("a", "b"):
        raise ValueError(f"stream must be 'a' or 'b', got {stream!r}")
    inp = cfg.input
    if cfg.task.kind == "multiview":
        stream = "a"
    override = inp.get(f"image_size_{stream}")
    return as_hw(override if override is not None else inp.image_size)


def build_transforms(cfg_input, train: bool, size=None) -> ImageTransform:
    return ImageTransform(cfg_input.image_size if size is None else size, train=train, clahe=bool(cfg_input.clahe), aug=dict(cfg_input.augment) if train else None)


def affine_for(width: int, height: int, size: int) -> Affine:
    """Deterministic stored-image -> model-input mapping (eval transforms), for mapping boxes / masks into model space."""
    return letterbox_params(width, height, size)


class PairDataset(Dataset):
    """Yields dicts with tensors ``a`` and ``b``, long ``y`` and string ids. Fails loudly on a missing file."""

    def __init__(self, pairs: pd.DataFrame, root_a: Path, root_b: Path, tf_a: Callable, tf_b: Callable, resampler: Callable[[int], pd.DataFrame] | None = None):
        self.pairs = pairs.reset_index(drop=True)
        self.root_a, self.root_b = Path(root_a), Path(root_b)
        self.tf_a, self.tf_b = tf_a, tf_b
        self.resampler = resampler

    def set_epoch(self, epoch: int) -> None:
        if self.resampler is not None:
            self.pairs = self.resampler(epoch).reset_index(drop=True)

    @property
    def targets(self) -> list[int]:
        return self.pairs["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _load(path: Path) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}. Did you run `mmibc prepare` and stage the data locally?")
        return Image.open(path)

    def __getitem__(self, i: int) -> dict:
        r = self.pairs.iloc[i]
        return {
            "a": self.tf_a(self._load(self.root_a / r["path_a"])),
            "b": self.tf_b(self._load(self.root_b / r["path_b"])),
            "y": torch.tensor(int(r["label"]), dtype=torch.long),
            "pair_id": str(r["pair_id"]),
            "uid_a": str(r["uid_a"]),
            "uid_b": str(r["uid_b"]),
        }

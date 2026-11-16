"""PyTorch datasets reading the memory-mapped uint8 caches written by ``pneumo prepare``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from ..paths import DATA_PROCESSED, images_path, meta_path


class RandomGamma(torch.nn.Module):
    """Random gamma curve on a float image in [0, 1] (log-uniform gamma): mimics different
    detector response and post-processing curves between hospitals."""

    def __init__(self, gamma: tuple[float, float] = (0.7, 1.5), p: float = 0.5):
        super().__init__()
        self.log_lo, self.log_hi, self.p = float(np.log(gamma[0])), float(np.log(gamma[1])), p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x
        g = torch.empty(1).uniform_(self.log_lo, self.log_hi).exp().item()
        return x.clamp(0, 1) ** g


class Clamp01(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0, 1)


AUGMENTATIONS = ("standard", "strong")


def build_transforms(train: bool, img_size: int = 224, augment: str = "standard") -> v2.Compose:
    """Input: uint8 tensor (1, H, W). Output: float32 tensor (1, img_size, img_size) in [0, 1].

    Augmentations are kept anatomically plausible: small rotations / shifts / zoom and
    exposure changes. Horizontal flips are deliberately not used (the heart is on the left).

    ``strong`` targets what differs between hospitals rather than between patients: exposure,
    contrast and gamma curves, histogram equalisation, sharpness, detector noise and small
    burnt-in markers (random erasing of 0.5-3 % patches).
    """
    if augment not in AUGMENTATIONS:
        raise ValueError(f"augment must be one of {AUGMENTATIONS}, got {augment!r}")
    if train and augment == "strong":
        return v2.Compose(
            [
                v2.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25), antialias=True),
                v2.RandomAffine(degrees=12, translate=(0.06, 0.06), shear=(-6.0, 6.0, -6.0, 6.0)),
                v2.RandomEqualize(p=0.15),
                v2.RandomAutocontrast(p=0.15),
                v2.ToDtype(torch.float32, scale=True),
                v2.ColorJitter(brightness=0.3, contrast=0.3),
                RandomGamma((0.7, 1.5), p=0.5),
                v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.2))], p=0.2),
                v2.RandomApply([v2.GaussianNoise(sigma=0.02)], p=0.2),
                v2.RandomErasing(p=0.25, scale=(0.005, 0.03), ratio=(0.5, 2.0), value=0.0),
                Clamp01(),
            ]
        )
    if train:
        return v2.Compose(
            [
                v2.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15), antialias=True),
                v2.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                v2.ToDtype(torch.float32, scale=True),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    return v2.Compose(
        [
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


class CXRDataset(Dataset):
    """Images from one source. The memmap is opened lazily so the dataset pickles cheaply
    into DataLoader worker processes (important on Windows, which spawns workers)."""

    def __init__(self, image_file: Path, meta: pd.DataFrame, transform: v2.Compose):
        self.image_file = Path(image_file)
        self.meta = meta.reset_index(drop=True)
        self.indices = self.meta["idx"].to_numpy()
        self.labels = self.meta["label"].to_numpy().astype(np.float32)
        self.transform = transform
        self._images: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._images is None:
            self._images = np.load(self.image_file, mmap_mode="r")
        img = torch.from_numpy(np.array(self._images[self.indices[i]])).unsqueeze(0)
        return self.transform(img), torch.tensor(self.labels[i])

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_images"] = None
        return state


def load_meta(source: str, processed_dir: Path = DATA_PROCESSED) -> pd.DataFrame:
    path = meta_path(source, processed_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `pneumo prepare {source}` first.")
    return pd.read_csv(path)


def make_dataset(
    source: str,
    split: str,
    train: bool,
    img_size: int = 224,
    processed_dir: Path = DATA_PROCESSED,
    max_samples: int | None = None,
    seed: int = 42,
    augment: str = "standard",
) -> CXRDataset:
    meta = load_meta(source, processed_dir)
    meta = meta[meta["split"] == split]
    if max_samples is not None and len(meta) > max_samples:
        meta = meta.sample(max_samples, random_state=seed)
    return CXRDataset(images_path(source, processed_dir), meta, build_transforms(train, img_size, augment))

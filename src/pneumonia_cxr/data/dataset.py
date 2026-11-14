"""PyTorch datasets reading the memory-mapped uint8 caches written by ``pneumo prepare``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from ..paths import DATA_PROCESSED, images_path, meta_path


def build_transforms(train: bool, img_size: int = 224) -> v2.Compose:
    """Input: uint8 tensor (1, H, W). Output: float32 tensor (1, img_size, img_size) in [0, 1].

    Augmentations are kept anatomically plausible: small rotations / shifts / zoom and
    exposure changes. Horizontal flips are deliberately not used (the heart is on the left).
    """
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
) -> CXRDataset:
    meta = load_meta(source, processed_dir)
    meta = meta[meta["split"] == split]
    if max_samples is not None and len(meta) > max_samples:
        meta = meta.sample(max_samples, random_state=seed)
    return CXRDataset(images_path(source, processed_dir), meta, build_transforms(train, img_size))

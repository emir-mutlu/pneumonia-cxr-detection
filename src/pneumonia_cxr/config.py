"""Training configuration loaded from YAML files in ``configs/``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    name: str
    arch: str = "convnext_tiny.fb_in22k_ft_in1k"
    pretrained: bool = True
    sources: list[str] = field(default_factory=lambda: ["kermany"])
    img_size: int = 224
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    patience: int = 4
    drop_path_rate: float = 0.1
    num_workers: int = 4
    seed: int = 42
    amp: bool = True
    # Optional cap on training images per source (for smoke tests and CI).
    max_train_samples: int | None = None
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> TrainConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

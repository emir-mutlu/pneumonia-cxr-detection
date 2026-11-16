"""A hospital's local operating point, written by ``pneumo calibrate``.

numpy and the standard library only, so the torch-free serving image (ONNX Runtime + FastAPI,
no scipy / scikit-learn) can load and apply it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


@dataclass
class SiteProfile:
    site: str
    platt_a: float
    platt_b: float
    threshold: float
    target_sensitivity: float | None
    n: int
    n_positive: int
    created: str = field(default_factory=lambda: date.today().isoformat())
    report: dict = field(default_factory=dict)

    def apply(self, prob: np.ndarray | float) -> np.ndarray:
        """Model probability (already temperature-scaled) -> locally recalibrated probability."""
        return _sigmoid(self.platt_a * logit(np.atleast_1d(prob)) + self.platt_b)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> SiteProfile:
        return cls(**json.loads(Path(path).read_text()))

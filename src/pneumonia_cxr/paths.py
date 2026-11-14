"""Project directory layout. Override the root with the PNEUMO_ROOT environment variable."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("PNEUMO_ROOT", Path.cwd())).resolve()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

SOURCES = ("kermany", "rsna")


def images_path(source: str, processed_dir: Path = DATA_PROCESSED) -> Path:
    return processed_dir / f"{source}_images.npy"


def meta_path(source: str, processed_dir: Path = DATA_PROCESSED) -> Path:
    return processed_dir / f"{source}_meta.csv"

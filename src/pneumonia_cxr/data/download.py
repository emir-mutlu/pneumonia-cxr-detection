"""Download the raw datasets from their public sources (resumable, no account needed).

* Kermany et al. pediatric chest X-rays (Guangzhou Women and Children's Medical Center),
  CC BY 4.0, via the Hugging Face mirror ``hf-vision/chest-xray-pneumonia``
  (original: https://data.mendeley.com/datasets/rscbjbr9sj/2).
* RSNA-STR Pneumonia Detection Challenge 2018 (adult, NIH ChestX-ray14 images with
  radiologist bounding boxes), from RSNA's official download site.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from tqdm import tqdm

from ..paths import DATA_RAW

HF_BASE = "https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia/resolve/main/data"
KERMANY_FILES = [
    "validation-00000-of-00001.parquet",
    "test-00000-of-00001.parquet",
    *[f"train-0000{i}-of-00007.parquet" for i in range(7)],
]

RSNA_BASE = "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018"
RSNA_FILES = [
    "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json",
    "pneumonia-challenge-dataset-mappings_2018.json",
    "pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip",
]


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        length = r.headers.get("Content-Length")
    return int(length) if length else None


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Download ``url`` to ``dest``, resuming a partial file if one exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = _remote_size(url)
    have = dest.stat().st_size if dest.exists() else 0
    if total is not None and have == total:
        return dest

    headers = {"Range": f"bytes={have}-"} if have else {}
    req = urllib.request.Request(url, headers=headers)
    with (
        urllib.request.urlopen(req, timeout=120) as r,
        open(dest, "ab" if have else "wb") as f,
        tqdm(total=total, initial=have, unit="B", unit_scale=True, desc=dest.name) as bar,
    ):
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            bar.update(len(buf))
    if total is not None and dest.stat().st_size != total:
        raise OSError(f"Incomplete download of {dest.name}; run the command again to resume.")
    return dest


def download_kermany(raw_dir: Path = DATA_RAW) -> Path:
    out = raw_dir / "kermany"
    for name in KERMANY_FILES:
        download_file(f"{HF_BASE}/{name}", out / name)
    return out


def download_rsna(raw_dir: Path = DATA_RAW) -> Path:
    out = raw_dir / "rsna"
    free_gb = shutil.disk_usage(raw_dir.parent if raw_dir.exists() else Path.cwd()).free / 1e9
    if free_gb < 6:
        raise OSError(f"RSNA needs ~4.1 GB of free disk space, only {free_gb:.1f} GB left.")
    for name in RSNA_FILES:
        download_file(f"{RSNA_BASE}/{name}", out / name)
    return out

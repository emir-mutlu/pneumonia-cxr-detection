"""Kermany et al. (2018) pediatric chest X-ray dataset -> cached uint8 array + metadata.

Known issues of the official split that this module fixes or documents:

* The official validation set has only 16 images, far too few to select a model or a
  decision threshold. We pool the official train + val images and re-split them
  **by patient** into train (85 %) and validation (15 %).
* File names encode the patient (``person123_bacteria_456.jpeg``, ``IM-0115-0001.jpeg``),
  so several images of the same child exist. Splitting by image would leak patients.
* The official test set (624 images) is kept untouched as the in-distribution test set.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from ..paths import DATA_PROCESSED, DATA_RAW, images_path, meta_path
from .imaging import CACHE_SIZE, resize_square, to_uint8_gray
from .splits import check_no_patient_leak, patient_split

_PNEUMONIA_RE = re.compile(r"^(person\d+)_(bacteria|virus)_\d+", re.IGNORECASE)
_NORMAL_RE = re.compile(r"^((?:NORMAL2-)?IM-\d+)-\d+", re.IGNORECASE)


def parse_kermany_filename(filename: str) -> tuple[str, str]:
    """Return ``(patient_id, subtype)`` from a Kermany file name.

    >>> parse_kermany_filename("person1086_bacteria_3026.jpeg")
    ('person1086', 'bacteria')
    >>> parse_kermany_filename("NORMAL2-IM-0577-0001.jpeg")
    ('NORMAL2-IM-0577', 'normal')
    """
    stem = Path(filename).name
    m = _PNEUMONIA_RE.match(stem)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    m = _NORMAL_RE.match(stem)
    if m:
        return m.group(1).upper(), "normal"
    return Path(stem).stem, "unknown"


def _official_split(parquet_name: str) -> str:
    return parquet_name.split("-")[0]  # train / validation / test


def prepare_kermany(
    raw_dir: Path = DATA_RAW,
    out_dir: Path = DATA_PROCESSED,
    size: int = CACHE_SIZE,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    files = sorted((raw_dir / "kermany").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No Kermany parquet files found. Run `pneumo download kermany` first.")

    n_total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(
        images_path("kermany", out_dir), mode="w+", dtype=np.uint8, shape=(n_total, size, size)
    )

    rows = []
    idx = 0
    with tqdm(total=n_total, desc="kermany") as bar:
        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=64, columns=["image", "label"]):
                for image, label in zip(
                    batch.column("image").to_pylist(), batch.column("label").to_pylist(), strict=True
                ):
                    with Image.open(io.BytesIO(image["bytes"])) as im:
                        w, h = im.size
                        images[idx] = resize_square(to_uint8_gray(im), size)
                    filename = Path(image["path"]).name
                    patient_id, subtype = parse_kermany_filename(filename)
                    rows.append(
                        {
                            "idx": idx,
                            "source": "kermany",
                            "filename": filename,
                            "label": int(label),
                            "subtype": subtype,
                            "patient_id": patient_id,
                            "official_split": _official_split(f.name),
                            "orig_w": w,
                            "orig_h": h,
                        }
                    )
                    idx += 1
                    bar.update(1)
    images.flush()

    meta = pd.DataFrame(rows)
    meta["split"] = "test"
    pool = meta[meta["official_split"] != "test"]
    meta.loc[pool.index, "split"] = patient_split(
        pool, {"train": 1 - val_fraction, "val": val_fraction}, seed=seed
    )
    check_no_patient_leak(meta[meta["split"] != "test"])
    meta.to_csv(meta_path("kermany", out_dir), index=False)

    summary = summarise(meta)
    (out_dir / "kermany_summary.json").write_text(json.dumps(summary, indent=2))
    return meta


def summarise(meta: pd.DataFrame) -> dict:
    """Counts per split plus the patient overlap between the official train and test sets."""
    train_patients = set(meta.loc[meta["official_split"] != "test", "patient_id"])
    test_patients = set(meta.loc[meta["official_split"] == "test", "patient_id"])
    per_split = {
        split: {
            "images": int(len(g)),
            "patients": int(g["patient_id"].nunique()),
            "pneumonia": int(g["label"].sum()),
            "normal": int((g["label"] == 0).sum()),
        }
        for split, g in meta.groupby("split")
    }
    return {
        "splits": per_split,
        "images_per_patient_max": int(meta.groupby("patient_id").size().max()),
        "patients_with_multiple_images": int((meta.groupby("patient_id").size() > 1).sum()),
        "official_train_test_patient_id_overlap": len(train_patients & test_patients),
    }

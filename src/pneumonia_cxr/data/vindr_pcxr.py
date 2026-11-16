"""VinDr-PCXR (Vietnam, pediatric, under 10 years) -> cached uint8 array + metadata.

PhysioNet restricted-access data (PhysioNet Restricted Health Data License 1.5.0). Sign the
data use agreement on https://physionet.org/content/vindr-pcxr/, download the ZIP from the
project page and run ``pneumo physionet-import <zip> --project vindr-pcxr``. The images are
never redistributed by this project; only aggregate metrics are published.

Labels come from ``image_labels_{train,test}.csv`` (one row per image, read by one of six
pediatric radiologists):

* any of ``Pneumonia``, ``Brocho-pneumonia`` (sic), ``Pleuro-pneumonia`` -> label 1
* ``No finding`` only                                                    -> label 0
* other diagnoses only (bronchitis, bronchiolitis, ...)                  -> label 0 (hard negatives)

The official test split is kept as the test set and 15 % of the official training split is
held out for validation. The dataset has no patient identifiers, so every image is treated
as its own patient (splits and bootstrap CIs are image-level).
"""

from __future__ import annotations

import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..paths import DATA_PROCESSED, DATA_RAW, images_path, meta_path
from .imaging import CACHE_SIZE, dicom_to_gray, resize_square
from .splits import check_no_patient_leak, patient_split

SOURCE = "vindr_pcxr"
RAW_NAME = "vindr-pcxr"
PNEUMONIA_COLUMNS = ("Pneumonia", "Brocho-pneumonia", "Pleuro-pneumonia")
ID_COLUMNS = ("image_id", "rad_ID")


def load_labels(raw: Path) -> pd.DataFrame:
    """One row per image with the binary label, a coarse class and its DICOM path."""
    frames = []
    for split in ("train", "test"):
        df = pd.read_csv(raw / f"image_labels_{split}.csv")
        diseases = [c for c in df.columns if c not in ID_COLUMNS]
        positive = df[list(PNEUMONIA_COLUMNS)].max(axis=1).astype(int)
        no_finding_only = (df["No finding"] == 1) & (df[diseases].sum(axis=1) == 1)
        frames.append(
            pd.DataFrame(
                {
                    "image_id": df["image_id"].astype(str),
                    "official_split": split,
                    "label": positive,
                    "vindr_class": np.select([positive == 1, no_finding_only], ["pneumonia", "no_finding"], "other"),
                    "file": [f"{split}/{i}.dicom" for i in df["image_id"]],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def parse_age_years(value: str) -> float:
    """DICOM age string (``005Y``, ``018M``, ``010W``, ``020D``) -> years; -1 if unknown."""
    m = re.fullmatch(r"(\d{1,3})\s*([DWMY])", str(value).strip().upper())
    if not m:
        return -1.0
    return round(int(m[1]) / {"D": 365.25, "W": 52.1775, "M": 12.0, "Y": 1.0}[m[2]], 3)


def acquisition_info(ds) -> dict:
    """Technical fingerprint of the imaging system. Manufacturer tags are removed in VinDr-PCXR,
    but pixel polarity, encoding and bit depth still separate the X-ray systems."""
    return {
        "photometric": str(ds.get("PhotometricInterpretation", "") or ""),
        "transfer_syntax": ds.file_meta.TransferSyntaxUID.name if "TransferSyntaxUID" in ds.file_meta else "",
        "bits_stored": int(ds.get("BitsStored", 0) or 0),
    }


def _read_dicom(args: tuple[str, int]) -> tuple[np.ndarray, dict]:
    import pydicom

    path, size = args
    ds = pydicom.dcmread(path)
    img = dicom_to_gray(ds)  # stored window applied, MONOCHROME1 inverted
    info = {
        "orig_h": int(img.height),
        "orig_w": int(img.width),
        "view": str(getattr(ds, "ViewPosition", "") or ""),
        "sex": str(getattr(ds, "PatientSex", "") or ""),
        "age": parse_age_years(getattr(ds, "PatientAge", "") or ""),
        **acquisition_info(ds),
    }
    return resize_square(img, size), info


def prepare_vindr_pcxr(
    raw_dir: Path = DATA_RAW,
    out_dir: Path = DATA_PROCESSED,
    size: int = CACHE_SIZE,
    val_fraction: float = 0.15,
    seed: int = 42,
    workers: int = 8,
) -> pd.DataFrame:
    raw = raw_dir / RAW_NAME
    for name in ("image_labels_train.csv", "image_labels_test.csv"):
        if not (raw / name).exists():
            raise FileNotFoundError(
                f"Missing {raw / name}. Download the ZIP from https://physionet.org/content/vindr-pcxr/ "
                "and run `pneumo physionet-import <zip> --project vindr-pcxr`."
            )

    meta = load_labels(raw).sort_values(["official_split", "image_id"], ignore_index=True)
    missing = [f for f in meta["file"] if not (raw / f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} DICOM files are missing, e.g. {missing[:3]}")
    meta["patient_id"] = meta["image_id"]  # no patient ids in this dataset
    meta["idx"] = np.arange(len(meta))
    meta["source"] = SOURCE

    out_dir.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(
        images_path(SOURCE, out_dir), mode="w+", dtype=np.uint8, shape=(len(meta), size, size)
    )
    tasks = [(str(raw / f), size) for f in meta["file"]]
    infos = []
    with Pool(workers) as pool:
        for i, (arr, info) in enumerate(tqdm(pool.imap(_read_dicom, tasks, chunksize=16), total=len(tasks), desc=SOURCE)):
            images[i] = arr
            infos.append(info)
    images.flush()
    meta = pd.concat([meta, pd.DataFrame(infos)], axis=1)

    split = pd.Series("test", index=meta.index, name="split")
    train = meta[meta["official_split"] == "train"]
    split[train.index] = patient_split(train, {"train": 1 - val_fraction, "val": val_fraction}, seed=seed)
    meta["split"] = split
    check_no_patient_leak(meta)

    cols = [
        "idx", "source", "image_id", "patient_id", "label", "vindr_class", "official_split",
        "view", "sex", "age", "photometric", "transfer_syntax", "bits_stored", "orig_w", "orig_h", "split",
    ]  # fmt: skip
    meta = meta[cols]
    meta.to_csv(meta_path(SOURCE, out_dir), index=False)
    (out_dir / f"{SOURCE}_summary.json").write_text(json.dumps(summarise(meta), indent=2))
    return meta


def summarise(meta: pd.DataFrame) -> dict:
    """Aggregate counts only (safe to print and publish)."""
    return {
        "splits": {
            split: {
                "images": int(len(g)),
                "pneumonia": int(g["label"].sum()),
                "no_finding": int((g["vindr_class"] == "no_finding").sum()),
                "other_disease": int((g["vindr_class"] == "other").sum()),
            }
            for split, g in meta.groupby("split")
        },
        "sex": {str(k): int(v) for k, v in meta["sex"].replace("", "unknown").value_counts().items()},
        "view": {str(k): int(v) for k, v in meta["view"].replace("", "unknown").value_counts().items()},
        "photometric": {str(k): int(v) for k, v in meta["photometric"].value_counts().items()},
        "transfer_syntax": {str(k): int(v) for k, v in meta["transfer_syntax"].value_counts().items()},
        "age_years": {
            "known": int((meta["age"] >= 0).sum()),
            "median": float(meta.loc[meta["age"] >= 0, "age"].median()) if (meta["age"] >= 0).any() else None,
        },
    }

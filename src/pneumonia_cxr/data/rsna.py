"""RSNA-STR Pneumonia Detection Challenge 2018 -> cached uint8 array + metadata.

Labels come from the final ("Calculated" label group) adjudicated Kaggle annotations:

* ``Lung Opacity``                      -> label 1, with radiologist bounding boxes
* ``No Lung Opacity / Not Normal``      -> label 0 (other abnormalities, hard negatives)
* ``Normal``                            -> label 0

The Kaggle ``patientId`` is really an *image* id. Using the official RSNA -> NIH mapping we
recover the true NIH patient id (``00012773_000.png`` -> ``00012773``), because many NIH
patients contribute several follow-up radiographs. Splits are made on the NIH patient id.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..paths import DATA_PROCESSED, DATA_RAW, images_path, meta_path
from .imaging import CACHE_SIZE, resize_square, to_uint8_gray
from .splits import check_no_patient_leak, patient_split

ANNOTATIONS = "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"
MAPPINGS = "pneumonia-challenge-dataset-mappings_2018.json"
IMAGES_ZIP = "pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip"

FINAL_GROUP = "Calculated"
CLASS_NAMES = {"Lung Opacity": 1, "No Lung Opacity / Not Normal": 0, "Normal": 0}


def load_labels(annotations_file: Path) -> pd.DataFrame:
    """One row per image (SOPInstanceUID) with its final class and boxes in pixel coords."""
    with open(annotations_file, encoding="utf-8") as f:
        project = json.load(f)
    label_ids = {
        lab["id"]: lab["name"]
        for group in project["labelGroups"]
        if group["name"] == FINAL_GROUP
        for lab in group["labels"]
    }
    classes: dict[str, str] = {}
    boxes: dict[str, list[list[float]]] = defaultdict(list)
    for ann in project["datasets"][0]["annotations"]:
        name = label_ids.get(ann["labelId"])
        if name is None:
            continue
        uid = ann["SOPInstanceUID"]
        classes[uid] = name
        if name == "Lung Opacity" and ann["data"]:
            d = ann["data"]
            boxes[uid].append([d["x"], d["y"], d["width"], d["height"]])

    return pd.DataFrame(
        {
            "sop_uid": list(classes),
            "rsna_class": list(classes.values()),
            "label": [CLASS_NAMES[c] for c in classes.values()],
            "boxes": [json.dumps(boxes.get(uid, [])) for uid in classes],
        }
    )


def load_mapping(mappings_file: Path) -> pd.DataFrame:
    with open(mappings_file, encoding="utf-8") as f:
        mapping = pd.DataFrame(json.load(f))
    mapping = mapping.rename(
        columns={"SOPInstanceUID": "sop_uid", "subset_img_id": "image_id", "img_id": "nih_image"}
    )
    mapping["patient_id"] = mapping["nih_image"].str.split("_").str[0]
    return mapping[["sop_uid", "image_id", "nih_image", "patient_id"]]


_ZIP: zipfile.ZipFile | None = None


def _init_worker(zip_path: str) -> None:
    global _ZIP
    _ZIP = zipfile.ZipFile(zip_path)


def _read_dicom(args: tuple[str, int]) -> tuple[np.ndarray, dict]:
    import pydicom

    member, size = args
    assert _ZIP is not None
    ds = pydicom.dcmread(io.BytesIO(_ZIP.read(member)))
    pixels = ds.pixel_array
    invert = getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1"
    img = to_uint8_gray(pixels, invert=invert)
    age = str(getattr(ds, "PatientAge", "") or "")
    info = {
        "orig_h": int(pixels.shape[0]),
        "orig_w": int(pixels.shape[1]),
        "view": str(getattr(ds, "ViewPosition", "") or ""),
        "sex": str(getattr(ds, "PatientSex", "") or ""),
        "age": int(age) if age.isdigit() else -1,
    }
    return resize_square(img, size), info


def prepare_rsna(
    raw_dir: Path = DATA_RAW,
    out_dir: Path = DATA_PROCESSED,
    size: int = CACHE_SIZE,
    fractions: dict[str, float] | None = None,
    seed: int = 42,
    workers: int = 8,
) -> pd.DataFrame:
    fractions = fractions or {"train": 0.7, "val": 0.1, "test": 0.2}
    rsna_dir = raw_dir / "rsna"
    for name in (ANNOTATIONS, MAPPINGS, IMAGES_ZIP):
        if not (rsna_dir / name).exists():
            raise FileNotFoundError(f"Missing {name}. Run `pneumo download rsna` first.")

    meta = load_labels(rsna_dir / ANNOTATIONS).merge(
        load_mapping(rsna_dir / MAPPINGS), on="sop_uid", how="inner", validate="one_to_one"
    )

    with zipfile.ZipFile(rsna_dir / IMAGES_ZIP) as zf:
        members = {
            Path(n).stem: n for n in zf.namelist() if n.lower().endswith(".dcm")  # stem = SOPInstanceUID
        }
    meta = meta[meta["sop_uid"].isin(members)].sort_values("image_id").reset_index(drop=True)
    meta["idx"] = np.arange(len(meta))
    meta["source"] = "rsna"

    out_dir.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(
        images_path("rsna", out_dir), mode="w+", dtype=np.uint8, shape=(len(meta), size, size)
    )
    tasks = [(members[uid], size) for uid in meta["sop_uid"]]
    infos = []
    with Pool(workers, initializer=_init_worker, initargs=(str(rsna_dir / IMAGES_ZIP),)) as pool:
        for i, (arr, info) in enumerate(
            tqdm(pool.imap(_read_dicom, tasks, chunksize=32), total=len(tasks), desc="rsna")
        ):
            images[i] = arr
            infos.append(info)
    images.flush()
    meta = pd.concat([meta, pd.DataFrame(infos)], axis=1)

    meta["split"] = patient_split(meta, fractions, seed=seed)
    check_no_patient_leak(meta)
    cols = [
        "idx", "source", "image_id", "nih_image", "patient_id", "label", "rsna_class",
        "boxes", "view", "sex", "age", "orig_w", "orig_h", "split",
    ]  # fmt: skip
    meta = meta[cols]
    meta.to_csv(meta_path("rsna", out_dir), index=False)
    (out_dir / "rsna_summary.json").write_text(json.dumps(summarise(meta), indent=2))
    return meta


def summarise(meta: pd.DataFrame) -> dict:
    per_patient = meta.groupby("patient_id").size()
    return {
        "splits": {
            split: {
                "images": int(len(g)),
                "patients": int(g["patient_id"].nunique()),
                "lung_opacity": int(g["label"].sum()),
                "not_normal": int((g["rsna_class"] == "No Lung Opacity / Not Normal").sum()),
                "normal": int((g["rsna_class"] == "Normal").sum()),
            }
            for split, g in meta.groupby("split")
        },
        "patients": int(len(per_patient)),
        "patients_with_multiple_images": int((per_patient > 1).sum()),
        "images_from_multi_image_patients": int(per_patient[per_patient > 1].sum()),
        "images_per_patient_max": int(per_patient.max()),
    }

"""VinDr-PCXR preparation on a tiny synthetic copy of its layout (no real patient data)."""

import json

import numpy as np
import pandas as pd
import pytest

pydicom = pytest.importorskip("pydicom")

from pneumonia_cxr.data.imaging import dicom_to_gray  # noqa: E402
from pneumonia_cxr.data.vindr_pcxr import load_labels, parse_age_years, prepare_vindr_pcxr  # noqa: E402
from pneumonia_cxr.evaluate import vindr_subgroup_metrics  # noqa: E402

DISEASES = ["No finding", "Bronchitis", "Brocho-pneumonia", "Other disease", "Bronchiolitis", "Pneumonia",
            "Pleuro-pneumonia"]  # fmt: skip


def _dicom(path, pixels: np.ndarray, age: str = "003Y", window: tuple[int, int] | None = None, mono1: bool = True):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.BitsAllocated, ds.BitsStored, ds.HighBit = 1, 16, 12, 11
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME1" if mono1 else "MONOCHROME2"
    ds.PatientAge, ds.PatientSex = age, "F"
    if window:
        ds.WindowCenter, ds.WindowWidth = window
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return path


def test_parse_age_years():
    assert parse_age_years("005Y") == 5.0
    assert parse_age_years("018M") == 1.5
    assert parse_age_years("026W") == pytest.approx(0.498, abs=1e-3)
    assert parse_age_years("") == -1.0 and parse_age_years("abc") == -1.0


def test_dicom_window_and_monochrome1(tmp_path):
    ramp = np.tile(np.linspace(0, 4095, 64).astype(np.uint16), (8, 1))
    windowed = np.asarray(dicom_to_gray(pydicom.dcmread(_dicom(tmp_path / "w.dicom", ramp, window=(1000, 1000)))))
    # Below the window -> black, then inverted (MONOCHROME1) -> white; above the window -> black.
    assert windowed[0, 0] == 255 and windowed[0, -1] == 0
    assert (windowed[0] == 0).sum() > 30  # everything above 1500 is saturated
    plain = np.asarray(dicom_to_gray(pydicom.dcmread(_dicom(tmp_path / "p.dicom", ramp, mono1=False))))
    assert plain[0, 0] == 0 and plain[0, -1] == 255 and len(np.unique(plain[0])) > 50


def _write_layout(raw, n_train: int = 20, n_test: int = 6) -> None:
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("test", n_test)):
        rows = []
        for i in range(n):
            labels = dict.fromkeys(DISEASES, 0)
            labels[["No finding", "Pneumonia", "Bronchitis", "Brocho-pneumonia"][i % 4]] = 1
            rows.append({"image_id": f"{split}{i:03d}", "rad_ID": "R1", **labels})
            _dicom(raw / split / f"{split}{i:03d}.dicom", rng.integers(0, 4096, (24, 20)), age=f"{i % 9:03d}Y")
        pd.DataFrame(rows).to_csv(raw / f"image_labels_{split}.csv", index=False)


def test_load_labels_maps_classes(tmp_path):
    raw = tmp_path / "vindr-pcxr"
    _write_layout(raw, 8, 4)
    meta = load_labels(raw)
    by_id = meta.set_index("image_id")
    assert by_id.loc["train000", ["label", "vindr_class"]].tolist() == [0, "no_finding"]
    assert by_id.loc["train001", ["label", "vindr_class"]].tolist() == [1, "pneumonia"]
    assert by_id.loc["train002", ["label", "vindr_class"]].tolist() == [0, "other"]
    assert by_id.loc["train003", ["label", "vindr_class"]].tolist() == [1, "pneumonia"]  # broncho-pneumonia
    assert by_id.loc["test000", "file"] == "test/test000.dicom"


def test_prepare_end_to_end(tmp_path):
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "processed"
    _write_layout(raw_dir / "vindr-pcxr")
    meta = prepare_vindr_pcxr(raw_dir, out_dir, size=16, workers=1)
    assert len(meta) == 26
    assert (meta["split"] == "test").sum() == 6 and set(meta.loc[meta["split"] == "test", "official_split"]) == {"test"}
    assert {"train", "val"} <= set(meta["split"])
    assert meta["age"].between(0, 9).all() and (meta["sex"] == "F").all()
    assert (meta["photometric"] == "MONOCHROME1").all() and (meta["bits_stored"] == 12).all()
    assert meta["transfer_syntax"].str.contains("Little Endian").all()
    images = np.load(out_dir / "vindr_pcxr_images.npy")
    assert images.shape == (26, 16, 16) and images.dtype == np.uint8
    summary = json.loads((out_dir / "vindr_pcxr_summary.json").read_text())
    assert sum(s["images"] for s in summary["splits"].values()) == 26


def test_prepare_explains_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="physionet-import"):
        prepare_vindr_pcxr(tmp_path, tmp_path / "out")


def test_vindr_subgroups():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "label": rng.integers(0, 2, n),
            "sex": rng.choice(["F", "M"], n),
            "age": rng.choice([0.5, 3.0, 7.0, -1.0], n),
            "vindr_class": rng.choice(["pneumonia", "no_finding", "other"], n),
            "photometric": rng.choice(["MONOCHROME1", "MONOCHROME2"], n),
        }
    )
    df.loc[df["vindr_class"] == "pneumonia", "label"] = 1
    df.loc[df["vindr_class"] != "pneumonia", "label"] = 0
    df["prob"] = np.clip(df["label"] * 0.6 + rng.random(n) * 0.5, 0, 1)
    sub = vindr_subgroup_metrics(df, threshold=0.5)
    assert {"sex=F", "sex=M", "age=<1y", "age=1-4y", "age=5-9y", "pneumonia_vs_no_finding_only"} <= set(sub)
    assert {"system=MONOCHROME1", "system=MONOCHROME2"} <= set(sub)
    assert all(v["n_positive"] >= 10 for k, v in sub.items() if k != "pneumonia_vs_no_finding_only")
    assert sub["pneumonia_vs_no_finding_only"]["n"] == int(df["vindr_class"].isin(["pneumonia", "no_finding"]).sum())

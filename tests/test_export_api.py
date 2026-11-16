"""End-to-end: tiny checkpoint -> ONNX export (parity-checked) -> ONNX predictor -> REST API."""

import importlib
import io

import numpy as np
import pytest
from PIL import Image

from pneumonia_cxr.inference import decode_image


def test_onnx_predictor(model_dir, png_bytes):
    from pneumonia_cxr.inference import OnnxPredictor

    pred = OnnxPredictor(model_dir).predict(Image.open(io.BytesIO(png_bytes)))
    assert 0.0 <= pred.probability <= 1.0
    assert pred.label in {"PNEUMONIA", "NORMAL"}


def test_decode_image_rejects_garbage_and_returns_grayscale(png_bytes):
    img = decode_image(png_bytes, "x.png")
    assert img.mode == "L" and img.size == (260, 300)
    with pytest.raises(ValueError):
        decode_image(b"not an image", "x.png")


def test_decode_image_reads_dicom(tmp_path):
    pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(tmp_path / "x.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = 32, 48
    ds.SamplesPerPixel, ds.BitsAllocated, ds.BitsStored, ds.HighBit = 1, 16, 12, 11
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.PixelData = (np.arange(32 * 48, dtype=np.uint16) % 4096).reshape(32, 48).tobytes()
    ds.save_as(tmp_path / "x.dcm", enforce_file_format=True)

    img = decode_image((tmp_path / "x.dcm").read_bytes(), "x.dcm")
    assert img.mode == "L" and img.size == (48, 32)


def test_rest_api(model_dir, png_bytes, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    import app.api as api

    api = importlib.reload(api)
    with TestClient(api.app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        assert client.get("/model").json()["img_size"] == 64
        r = client.post("/predict", files={"file": ("x.png", png_bytes, "image/png")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert 0.0 <= body["probability_pneumonia"] <= 1.0
        assert "not a medical device" in body["disclaimer"]
        bad = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
        assert bad.status_code == 400


def test_serving_path_needs_no_torch_sklearn_scipy(model_dir, tmp_path):
    """The Docker image ships only inference.py, site_profile.py, data/imaging.py and app/api.py on
    ONNX Runtime + FastAPI. None of them may import torch, scikit-learn or scipy (a site-profile
    import once broke the image at start-up)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = f"""
import sys
for name in ("torch", "torchvision", "timm", "sklearn", "scipy"):
    sys.modules[name] = None  # importing any of them now raises ImportError
import numpy as np
import app.api  # noqa: F401
from pneumonia_cxr.inference import OnnxPredictor
from pneumonia_cxr.site_profile import SiteProfile
profile = SiteProfile("s", 1.0, 0.5, 0.3, 0.9, 10, 5).save(r"{tmp_path / "s.json"}")
for site in (None, profile):
    print(OnnxPredictor(r"{model_dir}", site_profile=site).predict(np.zeros((64, 64), np.uint8)).label)
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert len(result.stdout.split()) == 2

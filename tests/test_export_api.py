"""End-to-end: tiny checkpoint -> ONNX export (parity-checked) -> ONNX predictor -> REST API."""

import importlib
import io

import numpy as np
import pytest
import torch
from PIL import Image

from pneumonia_cxr.config import TrainConfig
from pneumonia_cxr.models import build_model

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    run = tmp_path_factory.mktemp("run")
    cfg = TrainConfig(name="tiny", arch="simple_cnn", pretrained=False, img_size=64)
    torch.manual_seed(0)
    model = build_model("simple_cnn", pretrained=False)
    torch.save(
        {"config": cfg.to_dict(), "state_dict": model.state_dict(), "temperature": 1.5, "threshold": 0.5},
        run / "model.pt",
    )
    from pneumonia_cxr.export import export_onnx

    out = tmp_path_factory.mktemp("models")
    export_onnx(run, out, check=True)
    return out


def _png_bytes() -> bytes:
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (300, 260), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_onnx_predictor(model_dir):
    from pneumonia_cxr.inference import OnnxPredictor

    pred = OnnxPredictor(model_dir).predict(Image.open(io.BytesIO(_png_bytes())))
    assert 0.0 <= pred.probability <= 1.0
    assert pred.label in {"PNEUMONIA", "NORMAL"}


def test_rest_api(model_dir, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    import app.api as api

    api = importlib.reload(api)
    with TestClient(api.app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        assert client.get("/model").json()["img_size"] == 64
        r = client.post("/predict", files={"file": ("x.png", _png_bytes(), "image/png")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert 0.0 <= body["probability_pneumonia"] <= 1.0
        assert "not a medical device" in body["disclaimer"]
        bad = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
        assert bad.status_code == 400

"""Clinical web demo: static UI, model info, examples and the Grad-CAM analysis endpoint."""

import base64
import importlib
import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def client(model_dir, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    import app.web as web

    web = importlib.reload(web)
    with TestClient(web.app) as c:
        yield c


def test_ui_and_static_assets(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "app.js" in page.text and "Research use only" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_model_info_and_examples(client):
    info = client.get("/api/model").json()
    assert info["img_size"] == 64 and 0 < info["threshold"] < 1
    examples = client.get("/api/examples").json()
    assert {e["reference"] for e in examples} == {"pneumonia", "normal"}
    assert client.get(f"/examples/{examples[0]['file']}").status_code == 200


def test_analyse_returns_probability_heatmap_and_region(client, png_bytes):
    r = client.post("/api/analyse", files={"file": ("x.png", png_bytes, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["positive"] == (body["probability"] >= body["threshold"])
    assert body["region"]["side"] in {"left", "right"}
    assert body["region"]["zone"] in {"upper", "middle", "lower"}
    image = Image.open(io.BytesIO(base64.b64decode(body["image"]["png"])))
    heat = Image.open(io.BytesIO(base64.b64decode(body["heatmap"]["png"])))
    assert image.size == heat.size == (260, 300)
    assert heat.mode == "RGBA"


def test_analyse_rejects_bad_input(client):
    assert client.post("/api/analyse", files={"file": ("x.png", b"", "image/png")}).status_code == 400
    assert client.post("/api/analyse", files={"file": ("x.png", b"garbage", "image/png")}).status_code == 400


def test_strongest_region_uses_radiological_convention():
    from app.web import strongest_region

    heat = np.zeros((90, 90))
    heat[80, 10] = 1.0  # image left, bottom -> patient's right lower zone
    region = strongest_region(heat)
    assert (region["side"], region["zone"]) == ("right", "lower")

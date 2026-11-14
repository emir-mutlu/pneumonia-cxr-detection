"""Clinical-style web demo: X-ray viewer with Grad-CAM and a structured AI findings panel.

Needs PyTorch (Grad-CAM uses gradients). For a slim, torch-free service use ``app/api.py``.

    pneumo demo                          # http://localhost:7860
    uvicorn app.web:app --port 7860
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import matplotlib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from pneumonia_cxr import __version__
from pneumonia_cxr.data.imaging import preprocess_for_model
from pneumonia_cxr.gradcam import GradCAM
from pneumonia_cxr.inference import DISCLAIMER, decode_image
from pneumonia_cxr.train import get_device, load_checkpoint

HERE = Path(__file__).parent
STATIC = HERE / "static"
EXAMPLES = HERE / "examples"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
MAX_BYTES = 25 * 1024 * 1024
DISPLAY_MAX = 1024  # longest side of the image sent back to the browser
_CMAP = matplotlib.colormaps["inferno"]
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    device = get_device()
    model, ckpt = load_checkpoint(MODEL_DIR, device)
    meta_file = MODEL_DIR / "model_meta.json"
    state.update(
        model=model,
        ckpt=ckpt,
        device=device,
        meta=json.loads(meta_file.read_text()) if meta_file.exists() else {},
        # One dedicated worker: Grad-CAM hooks on a shared model are not thread-safe, and
        # CUDA/cuBLAS set-up cost is paid once instead of once per request thread.
        executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="model"),
    )
    state["executor"].submit(analyse, Image.new("L", (256, 256), 128)).result()  # warm-up
    yield
    state["executor"].shutdown(wait=False)
    state.clear()


app = FastAPI(title="Pneumonia CXR demo", version=__version__, lifespan=lifespan, docs_url="/api/docs")


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def strongest_region(heat: np.ndarray) -> dict:
    """Location of the Grad-CAM peak in radiological convention.

    On a frontal (PA/AP) radiograph the patient's right is on the left of the image.
    Zones are thirds of the image height: upper / middle / lower.
    """
    h, w = heat.shape
    y, x = np.unravel_index(int(np.argmax(heat)), heat.shape)
    return {
        "side": "right" if x < w / 2 else "left",
        "zone": "upper" if y < h / 3 else ("middle" if y < 2 * h / 3 else "lower"),
        "x": round(float(x / w), 3),
        "y": round(float(y / h), 3),
    }


def heatmap_rgba(heat: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Inferno-coloured heatmap whose transparency follows the activation, so weak
    regions stay clear and the anatomy remains readable underneath."""
    small = Image.fromarray(np.clip(heat * 255, 0, 255).astype(np.uint8))
    h = np.asarray(small.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    rgba = _CMAP(h)
    rgba[..., 3] = np.clip((h - 0.15) / 0.85, 0, 1) ** 0.8
    return Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")


def analyse(gray: Image.Image) -> dict:
    model, ckpt, device = state["model"], state["ckpt"], state["device"]
    size = ckpt["config"]["img_size"]
    w, h = gray.size
    scale = min(1.0, DISPLAY_MAX / max(w, h))
    display = gray.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)

    t0 = time.perf_counter()
    x = torch.from_numpy(preprocess_for_model(gray, size)).to(device)
    with GradCAM(model) as cam:
        heat, logit = cam(x)
    prob = float(torch.sigmoid(logit.float() / ckpt["temperature"])[0])
    elapsed_ms = (time.perf_counter() - t0) * 1000
    heat_np = heat[0].cpu().numpy()

    threshold = float(ckpt["threshold"])
    meta = state["meta"]
    return {
        "probability": prob,
        "threshold": threshold,
        "positive": prob >= threshold,
        "label": "PNEUMONIA" if prob >= threshold else "NORMAL",
        "region": strongest_region(heat_np),
        "image": {"width": w, "height": h, "png": _png_b64(display)},
        "heatmap": {"png": _png_b64(heatmap_rgba(heat_np, display.size))},
        "inference_ms": round(elapsed_ms, 1),
        "device": "GPU" if device.type == "cuda" else "CPU",
        "model": {
            "arch": ckpt["config"]["arch"],
            "train_sources": ckpt["config"]["sources"],
            "version": meta.get("package_version", __version__),
        },
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/analyse")
async def analyse_upload(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File larger than 25 MB")
    try:
        gray = decode_image(data, file.filename or "")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    result = await asyncio.get_running_loop().run_in_executor(state["executor"], analyse, gray)
    return {"filename": file.filename, **result}


@app.get("/api/model")
def model_info() -> dict:
    ckpt, meta = state["ckpt"], state["meta"]
    return {
        "arch": ckpt["config"]["arch"],
        "train_sources": ckpt["config"]["sources"],
        "img_size": ckpt["config"]["img_size"],
        "threshold": ckpt["threshold"],
        "temperature": ckpt["temperature"],
        "n_params": ckpt.get("n_params"),
        "version": meta.get("package_version", __version__),
        "test_metrics": meta.get("test_metrics", {}),
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/examples")
def examples() -> list[dict]:
    index = EXAMPLES / "examples.json"
    return json.loads(index.read_text(encoding="utf-8")) if index.exists() else []


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state, "device": str(state.get("device", ""))}


app.mount("/examples", StaticFiles(directory=EXAMPLES), name="examples")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")

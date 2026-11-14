"""REST API serving the exported ONNX model (no PyTorch required).

Run locally:   uvicorn app.api:app --port 8000
Docker:        docker build -t pneumonia-cxr . && docker run -p 8000:8000 pneumonia-cxr
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from pneumonia_cxr import __version__
from pneumonia_cxr.inference import DISCLAIMER, OnnxPredictor, decode_image

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
MAX_BYTES = 25 * 1024 * 1024
state: dict[str, OnnxPredictor] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["predictor"] = OnnxPredictor(MODEL_DIR)
    yield
    state.clear()


app = FastAPI(
    title="Pneumonia CXR API",
    version=__version__,
    description=f"Chest X-ray pneumonia classifier. {DISCLAIMER}",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "predictor" in state}


@app.get("/model")
def model_info() -> dict:
    return state["predictor"].meta


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File larger than 25 MB")
    try:
        image = decode_image(data, file.filename or "")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"filename": file.filename, **state["predictor"].predict(image).to_dict()}

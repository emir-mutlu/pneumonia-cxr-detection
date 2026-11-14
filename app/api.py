"""REST API serving the exported ONNX model (no PyTorch required).

Run locally:   uvicorn app.api:app --port 8000
Docker:        docker build -t pneumonia-cxr . && docker run -p 8000:8000 pneumonia-cxr
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from pneumonia_cxr import __version__
from pneumonia_cxr.inference import DISCLAIMER, OnnxPredictor

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


def _decode(data: bytes, filename: str) -> Image.Image:
    if filename.lower().endswith(".dcm") or data[128:132] == b"DICM":
        try:
            import pydicom
        except ImportError as e:  # pragma: no cover
            raise HTTPException(415, "DICOM support needs pydicom") from e
        ds = pydicom.dcmread(io.BytesIO(data))
        from pneumonia_cxr.data.imaging import to_uint8_gray

        invert = getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1"
        return to_uint8_gray(ds.pixel_array, invert=invert)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, "Could not decode the image (PNG, JPEG or DICOM expected)") from e


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
    image = _decode(data, file.filename or "")
    return {"filename": file.filename, **state["predictor"].predict(image).to_dict()}

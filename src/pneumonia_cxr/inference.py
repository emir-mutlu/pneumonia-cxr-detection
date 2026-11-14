"""Torch-free inference with ONNX Runtime (used by the REST API and the Docker image)."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .data.imaging import preprocess_for_model, to_uint8_gray

DISCLAIMER = "Research prototype - not a medical device and not for clinical decisions."


def decode_image(data: bytes, filename: str = "") -> Image.Image:
    """Decode PNG / JPEG / DICOM bytes into an 8-bit grayscale PIL image.

    Raises ``ValueError`` if the bytes are not a readable image.
    """
    if filename.lower().endswith(".dcm") or data[128:132] == b"DICM":
        try:
            import pydicom
        except ImportError as e:  # pragma: no cover
            raise ValueError("DICOM support needs pydicom (pip install pydicom)") from e
        try:
            ds = pydicom.dcmread(io.BytesIO(data))
            invert = getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1"
            return to_uint8_gray(ds.pixel_array, invert=invert)
        except Exception as e:
            raise ValueError(f"Could not read the DICOM file: {e}") from e
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return to_uint8_gray(img)
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Could not decode the image (PNG, JPEG or DICOM expected)") from e


@dataclass
class Prediction:
    probability: float
    label: str
    threshold: float

    def to_dict(self) -> dict:
        return {
            "probability_pneumonia": round(self.probability, 4),
            "prediction": self.label,
            "threshold": round(self.threshold, 4),
            "disclaimer": DISCLAIMER,
        }


class OnnxPredictor:
    def __init__(self, model_dir: str | Path, providers: list[str] | None = None):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "model_meta.json").read_text())
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), providers=providers or ["CPUExecutionProvider"]
        )
        self.img_size = int(self.meta["img_size"])
        self.cache_size = int(self.meta.get("cache_size", 256))
        self.threshold = float(self.meta["threshold"])

    def predict_proba(self, images: list[Image.Image | np.ndarray]) -> np.ndarray:
        batch = np.concatenate([preprocess_for_model(im, self.img_size, self.cache_size) for im in images])
        return self.session.run(None, {"image": batch})[0].astype(float)

    def predict(self, image: Image.Image | np.ndarray) -> Prediction:
        p = float(self.predict_proba([image])[0])
        return Prediction(p, "PNEUMONIA" if p >= self.threshold else "NORMAL", self.threshold)

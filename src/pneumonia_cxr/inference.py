"""Torch-free inference with ONNX Runtime (used by the REST API and the Docker image)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .data.imaging import preprocess_for_model

DISCLAIMER = "Research prototype - not a medical device and not for clinical decisions."


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

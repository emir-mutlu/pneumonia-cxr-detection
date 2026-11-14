import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # makes `app` importable


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """A tiny untrained checkpoint exported like the released model (model.pt, model.onnx, meta)."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    import torch

    from pneumonia_cxr.config import TrainConfig
    from pneumonia_cxr.export import export_onnx
    from pneumonia_cxr.models import build_model

    run = tmp_path_factory.mktemp("run")
    cfg = TrainConfig(name="tiny", arch="simple_cnn", pretrained=False, img_size=64)
    torch.manual_seed(0)
    model = build_model("simple_cnn", pretrained=False)
    torch.save(
        {"config": cfg.to_dict(), "state_dict": model.state_dict(), "temperature": 1.5, "threshold": 0.5},
        run / "model.pt",
    )
    out = tmp_path_factory.mktemp("models")
    export_onnx(run, out, check=True)
    return out


@pytest.fixture
def png_bytes() -> bytes:
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (300, 260), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

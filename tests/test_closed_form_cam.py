"""Closed-form Grad-CAM (used for in-browser inference) must equal autograd Grad-CAM."""

import pytest
import torch

from pneumonia_cxr.config import TrainConfig
from pneumonia_cxr.gradcam import ClosedFormGradCAM, GradCAM
from pneumonia_cxr.models import build_model


@pytest.mark.parametrize("arch", ["convnext_atto", "resnet18"])
def test_closed_form_matches_autograd(arch):
    torch.manual_seed(0)
    model = build_model(arch, pretrained=False).eval()
    x = torch.rand(3, 1, 64, 64)
    with torch.no_grad():
        prob, cam = ClosedFormGradCAM(model, temperature=1.3)(x)
    with GradCAM(model) as gc:
        ref_cam, logit = gc(x)
    assert torch.allclose(prob, torch.sigmoid(logit / 1.3), atol=1e-5)
    assert torch.allclose(cam, ref_cam, atol=1e-4)
    assert cam.shape == (3, 64, 64)


def test_unsupported_backbone_is_rejected():
    with pytest.raises(ValueError):
        ClosedFormGradCAM(build_model("simple_cnn", pretrained=False))


def test_export_browser_onnx(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    import onnxruntime as ort

    from pneumonia_cxr.export import export_browser_onnx

    torch.manual_seed(0)
    cfg = TrainConfig(name="tiny", arch="convnext_atto", pretrained=False, img_size=64)
    model = build_model("convnext_atto", pretrained=False)
    torch.save(
        {"config": cfg.to_dict(), "state_dict": model.state_dict(), "temperature": 1.2, "threshold": 0.5},
        tmp_path / "model.pt",
    )
    path = export_browser_onnx(tmp_path, tmp_path / "out" / "model_browser.onnx")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [o.name for o in sess.get_outputs()] == ["probability", "cam"]
    prob, cam = sess.run(None, {"image": torch.rand(2, 1, 64, 64).numpy()})
    assert prob.shape == (2,) and cam.shape == (2, 64, 64)
    assert ((prob >= 0) & (prob <= 1)).all() and cam.min() >= 0 and cam.max() <= 1 + 1e-6

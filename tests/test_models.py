import json

import numpy as np
import pytest
import torch

from pneumonia_cxr.gradcam import GradCAM, boxes_to_mask, localisation_scores
from pneumonia_cxr.models import build_model, cam_target_layer


@pytest.mark.parametrize("arch", ["simple_cnn", "resnet18", "convnext_atto", "efficientnet_b0"])
def test_forward_shape_and_gradcam(arch):
    torch.manual_seed(0)
    model = build_model(arch, pretrained=False).eval()
    x = torch.rand(2, 1, 64, 64)
    assert model(x).shape == (2,)
    assert cam_target_layer(model) is not None
    with GradCAM(model) as cam:
        heat, logits = cam(x)
    assert heat.shape == (2, 64, 64)
    assert logits.shape == (2,)
    assert float(heat.min()) >= 0.0 and float(heat.max()) <= 1.0 + 1e-6


def test_imagenet_normalisation_inside_model():
    model = build_model("resnet18", pretrained=False)
    x = torch.full((1, 1, 8, 8), 0.485)
    assert torch.allclose(model.normalise(x)[:, 0], torch.zeros(1, 8, 8), atol=1e-6)
    assert model.normalise(x).shape[1] == 3


def test_boxes_to_mask_scales_coordinates():
    mask = boxes_to_mask([[512, 0, 512, 512]], orig_w=1024, orig_h=1024, size=8)
    expected = np.zeros((8, 8), bool)
    expected[:4, 4:] = True
    assert np.array_equal(mask, expected)


def test_localisation_scores():
    mask = np.zeros((10, 10), bool)
    mask[:5, :5] = True
    cam = np.zeros((10, 10))
    cam[2, 2] = 1.0
    cam[8, 8] = 1.0
    s = localisation_scores(cam, mask)
    assert s["hit"] == 1.0
    assert s["energy_in_box"] == pytest.approx(0.5)
    assert s["box_fraction"] == pytest.approx(0.25)
    assert json.dumps(s)  # serialisable

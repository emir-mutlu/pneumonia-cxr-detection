"""Grad-CAM (Selvaraju et al., 2017) and localisation scores against radiologist boxes."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import PneumoniaNet, cam_target_layer


class GradCAM:
    """Class activation maps for the single pneumonia logit.

    Usage::

        with GradCAM(model) as cam:
            heatmaps, logits = cam(x)   # heatmaps: (B, H, W) in [0, 1]
    """

    def __init__(self, model: PneumoniaNet, layer: nn.Module | None = None):
        self.model = model
        self.layer = layer if layer is not None else cam_target_layer(model)
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._handle = None

    def _hook(self, _module, _inputs, output: torch.Tensor) -> None:
        self._acts = output
        output.register_hook(self._save_grad)

    def _save_grad(self, grad: torch.Tensor) -> None:
        self._grads = grad

    def __enter__(self) -> GradCAM:
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        x = x.detach().requires_grad_(True)  # ensures a graph even if the backbone is frozen
        with torch.enable_grad():
            logits = self.model(x)
            self.model.zero_grad(set_to_none=True)
            logits.sum().backward()
        acts, grads = self._acts.detach().float(), self._grads.detach().float()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        flat = cam.flatten(1)
        lo, hi = flat.min(1, keepdim=True).values, flat.max(1, keepdim=True).values
        cam = ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(cam)
        return cam, logits.detach()


def boxes_to_mask(boxes: list[list[float]], orig_w: int, orig_h: int, size: int) -> np.ndarray:
    """Rasterise [x, y, w, h] boxes (original pixel coordinates) on a ``size`` x ``size`` grid."""
    mask = np.zeros((size, size), dtype=bool)
    sx, sy = size / orig_w, size / orig_h
    for x, y, w, h in boxes:
        x0, y0 = int(np.floor(x * sx)), int(np.floor(y * sy))
        x1, y1 = int(np.ceil((x + w) * sx)), int(np.ceil((y + h) * sy))
        mask[max(0, y0) : min(size, y1), max(0, x0) : min(size, x1)] = True
    return mask


def localisation_scores(cam: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Pointing game hit, share of CAM energy inside the boxes and the uniform-map baseline."""
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    total = float(cam.sum())
    return {
        "hit": float(mask[peak]),
        "energy_in_box": float(cam[mask].sum() / total) if total > 0 else 0.0,
        "box_fraction": float(mask.mean()),
    }


class ClosedFormGradCAM(nn.Module):
    """Probability and Grad-CAM in a single forward pass, without autograd.

    For heads of the form ``global average pool -> [LayerNorm] -> Linear`` the gradient of the
    logit with respect to the last feature map is the same at every spatial position, so the
    Grad-CAM channel weights have a closed form:

    * linear head:            alpha_k = w_k / (H * W)
    * LayerNorm + linear head: alpha = (g - mean(g) - x_hat * mean(g * x_hat)) / (sigma * H * W),
      with g = w * gamma and x_hat the normalised pooled features.

    Because it needs no backward pass, the module exports to ONNX and runs anywhere ONNX
    Runtime does, including in a web browser. It matches :class:`GradCAM` to float precision.
    Supported backbones: timm ConvNeXt and timm ResNet-style models.
    """

    def __init__(self, model: PneumoniaNet, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        head = getattr(model.backbone, "head", None)
        if head is not None and hasattr(head, "norm") and hasattr(head, "fc"):
            self.kind = "layernorm_linear"  # ConvNeXt: NormMlpClassifierHead
            if not isinstance(getattr(head, "pre_logits", nn.Identity()), nn.Identity):
                raise ValueError("ConvNeXt heads with a hidden MLP are not supported")
        elif isinstance(getattr(model.backbone, "fc", None), nn.Linear):
            self.kind = "linear"  # ResNet-style: global_pool + fc
        else:
            raise ValueError(f"No closed-form Grad-CAM for {type(model.backbone).__name__}")

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        backbone = self.model.backbone
        feats = backbone.forward_features(self.model.normalise(image))  # (B, C, h, w)
        pooled = feats.mean((2, 3))
        if self.kind == "layernorm_linear":
            ln, fc = backbone.head.norm, backbone.head.fc
            mu = pooled.mean(1, keepdim=True)
            sigma = torch.sqrt(pooled.var(1, unbiased=False, keepdim=True) + ln.eps)
            x_hat = (pooled - mu) / sigma
            logit = ((x_hat * ln.weight + ln.bias) @ fc.weight.t() + fc.bias).squeeze(1)
            g = fc.weight[0] * ln.weight
            d_pooled = (g - g.mean() - x_hat * (g * x_hat).mean(1, keepdim=True)) / sigma
        else:
            fc = backbone.fc
            logit = (pooled @ fc.weight.t() + fc.bias).squeeze(1)
            d_pooled = fc.weight[0].expand_as(pooled)
        alpha = d_pooled / (feats.shape[2] * feats.shape[3])
        cam = F.relu((alpha[:, :, None, None] * feats).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        flat = cam.flatten(1)
        lo, hi = flat.min(1, keepdim=True).values, flat.max(1, keepdim=True).values
        cam = ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(cam)
        return torch.sigmoid(logit / self.temperature), cam

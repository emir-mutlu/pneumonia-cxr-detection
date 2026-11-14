"""Model zoo. Every model takes a (B, 1, H, W) float tensor in [0, 1] and returns (B,) logits.

Input normalisation lives inside the model, so training, ONNX export and serving all feed
the same raw [0, 1] grayscale tensor.

* ``simple_cnn``            - small CNN trained from scratch (the original 2020 project).
* ``xrv_densenet121``       - DenseNet-121 pre-trained on chest X-rays (TorchXRayVision,
                              CheXpert weights), i.e. in-domain transfer learning.
* any ``timm`` model name   - ImageNet-pre-trained backbone, e.g. ``resnet50.a1_in1k``,
                              ``efficientnet_b0.ra_in1k``, ``convnext_tiny.fb_in22k_ft_in1k``.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SimpleCNN(nn.Module):
    """Four conv blocks + global pooling, roughly the Keras CNN of the original project."""

    def __init__(self, widths: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for w in widths:
            layers += [
                nn.Conv2d(in_ch, w, 3, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
                nn.Conv2d(w, w, 3, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class XRVDenseNet(nn.Module):
    """TorchXRayVision DenseNet-121 backbone with a new single-logit head."""

    def __init__(self, weights: str | None = "densenet121-res224-chex"):
        super().__init__()
        import torchxrayvision as xrv

        base = xrv.models.DenseNet(weights=weights) if weights else xrv.models.DenseNet()
        self.features = base.features
        self.head = nn.Linear(base.classifier.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = torch.relu(self.features(x))
        return self.head(torch.flatten(nn.functional.adaptive_avg_pool2d(f, 1), 1))


class PneumoniaNet(nn.Module):
    def __init__(self, backbone: nn.Module, input_mode: str):
        super().__init__()
        if input_mode not in {"raw", "imagenet", "xrv"}:
            raise ValueError(input_mode)
        self.backbone = backbone
        self.input_mode = input_mode
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "imagenet":
            return (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        if self.input_mode == "xrv":
            return (x * 2.0 - 1.0) * 1024.0  # TorchXRayVision expects [-1024, 1024]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.normalise(x)).squeeze(-1)


def build_model(arch: str, pretrained: bool = True, drop_path_rate: float = 0.0) -> PneumoniaNet:
    if arch == "simple_cnn":
        return PneumoniaNet(SimpleCNN(), "raw")
    if arch == "xrv_densenet121":
        return PneumoniaNet(XRVDenseNet("densenet121-res224-chex" if pretrained else None), "xrv")
    kwargs = {"drop_path_rate": drop_path_rate} if drop_path_rate else {}
    backbone = timm.create_model(arch, pretrained=pretrained, num_classes=1, **kwargs)
    return PneumoniaNet(backbone, "imagenet")


def cam_target_layer(model: PneumoniaNet) -> nn.Module:
    """Last spatial feature map, used by Grad-CAM."""
    b = model.backbone
    if isinstance(b, SimpleCNN | XRVDenseNet):
        return b.features
    for attr in ("stages", "layer4", "features", "blocks"):  # convnext / resnet / densenet / effnet
        if hasattr(b, attr):
            layer = getattr(b, attr)
            if attr == "blocks" and hasattr(b, "conv_head"):
                return b.conv_head
            return layer[-1] if isinstance(layer, nn.Sequential) and attr != "features" else layer
    raise ValueError(f"Don't know the Grad-CAM layer for {type(b).__name__}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

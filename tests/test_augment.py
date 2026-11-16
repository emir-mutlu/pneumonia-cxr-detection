"""Strong (hospital-agnostic) augmentation and hospital-balanced sampling."""

import collections

import pytest
import torch

from pneumonia_cxr.config import TrainConfig
from pneumonia_cxr.data.dataset import RandomGamma, build_transforms
from pneumonia_cxr.train import balanced_sampler


@pytest.mark.parametrize("augment", ["standard", "strong"])
def test_train_transforms_shape_and_range(augment):
    torch.manual_seed(0)
    tf = build_transforms(True, 64, augment)
    img = torch.randint(0, 256, (1, 96, 96), dtype=torch.uint8)
    for _ in range(25):
        out = tf(img)
        assert out.shape == (1, 64, 64) and out.dtype == torch.float32
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_eval_transform_ignores_augment():
    img = torch.randint(0, 256, (1, 96, 96), dtype=torch.uint8)
    assert torch.equal(build_transforms(False, 64, "strong")(img), build_transforms(False, 64)(img))


def test_unknown_augment_is_rejected():
    with pytest.raises(ValueError):
        build_transforms(True, 64, "extreme")


def test_random_gamma_is_monotonic_and_bounded():
    x = torch.linspace(0, 1, 101).view(1, 1, -1)
    y = RandomGamma((0.5, 2.0), p=1.0)(x)
    assert y.min() >= 0 and y.max() <= 1 and (y.diff() >= 0).all()
    assert torch.equal(RandomGamma(p=0.0)(x), x)


def test_balanced_sampler_draws_sources_equally():
    sizes = [100, 1000, 300]
    draws = list(balanced_sampler(sizes, seed=0))
    assert len(draws) == sum(sizes)
    bounds = torch.tensor(sizes).cumsum(0).tolist()
    per_source = collections.Counter(next(i for i, b in enumerate(bounds) if d < b) for d in draws)
    share = [per_source[i] / len(draws) for i in range(3)]
    assert all(abs(s - 1 / 3) < 0.04 for s in share)


def test_config_accepts_new_keys():
    cfg = TrainConfig.from_dict({"name": "x", "augment": "strong", "balance_sources": True})
    assert cfg.augment == "strong" and cfg.balance_sources

"""Local validation kit: Platt recalibration, local threshold, honest held-out estimate, CLI."""

import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pneumonia_cxr.metrics import sigmoid
from pneumonia_cxr.site_calibration import SiteProfile, calibrate_site, fit_platt, format_report


def _shifted_site(n: int = 1200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A hospital where the model still ranks well but is far too unconfident (like VinDr-PCXR)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    z = rng.normal(0, 1, n) + 1.6 * y  # separable scores
    return sigmoid(0.5 * z - 3.5), y  # squashed and shifted towards 0


def test_fit_platt_recovers_parameters():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, 20000)
    y = (rng.random(z.size) < sigmoid(1.7 * z - 0.8)).astype(int)
    a, b = fit_platt(z, y, l2=0.0)
    assert a == pytest.approx(1.7, abs=0.08) and b == pytest.approx(-0.8, abs=0.08)


def test_calibrate_site_fixes_a_threshold_that_does_not_transfer():
    probs, y = _shifted_site()
    profile = calibrate_site(probs, y, "test-site", model_threshold=0.576, target_sensitivity=0.9, n_boot=200)
    r = profile.report
    assert r["before"]["sensitivity"] < 0.1  # the released threshold misses almost everything
    assert r["after_held_out_estimate"]["sensitivity"] > 0.8  # close to the 0.9 target on unseen halves
    assert r["after_held_out_estimate"]["ece"] < r["before"]["ece"]
    assert r["auroc_ci95"][0] < r["auroc"] < r["auroc_ci95"][1]
    assert not r["warnings"] and "Site 'test-site'" in format_report(profile)


def test_small_samples_warn_and_youden_works():
    probs, y = _shifted_site(n=120, seed=3)
    profile = calibrate_site(probs, y, "tiny", model_threshold=0.5, target_sensitivity=None, n_boot=50)
    assert profile.target_sensitivity is None and profile.report["warnings"]


def test_profile_roundtrip_and_monotonic(tmp_path):
    profile = SiteProfile("x", platt_a=1.3, platt_b=0.7, threshold=0.4, target_sensitivity=0.9, n=10, n_positive=3)
    loaded = SiteProfile.load(profile.save(tmp_path / "x.json"))
    assert loaded == profile
    p = np.linspace(0.01, 0.99, 50)
    assert np.all(np.diff(loaded.apply(p)) > 0)


def test_needs_both_classes():
    with pytest.raises(ValueError):
        calibrate_site(np.full(10, 0.3), np.zeros(10), "x", 0.5)


def test_predictor_uses_site_profile(model_dir, png_bytes, tmp_path):
    from pneumonia_cxr.inference import OnnxPredictor, decode_image

    img = decode_image(png_bytes, "x.png")
    base = OnnxPredictor(model_dir)
    path = SiteProfile("s", 1.0, 2.0, threshold=0.123, target_sensitivity=0.9, n=10, n_positive=5).save(
        tmp_path / "s.json"
    )
    local = OnnxPredictor(model_dir, site_profile=path)
    raw = base.predict_raw([img])[0]
    assert local.threshold == 0.123 and base.threshold == base.model_threshold
    assert local.predict_proba([img])[0] == pytest.approx(float(sigmoid(np.log(raw / (1 - raw)) + 2.0)))


def test_calibrate_cli(model_dir, tmp_path):
    from typer.testing import CliRunner

    from pneumonia_cxr.cli import app

    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        buf = io.BytesIO()
        Image.fromarray(rng.integers(0, 256, (80, 70), dtype=np.uint8)).save(buf, format="PNG")
        (tmp_path / f"{i}.png").write_bytes(buf.getvalue())
        rows.append({"file": f"{i}.png", "label": i % 2})
    pd.DataFrame(rows).to_csv(tmp_path / "labels.csv", index=False)
    out = tmp_path / "profile.json"
    result = CliRunner().invoke(
        app,
        ["calibrate", str(tmp_path / "labels.csv"), "--site", "demo", "--images-dir", str(tmp_path),
         "--model-dir", str(model_dir), "--out", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output  # 20 + 20 cases is below the recommended minimum
    assert SiteProfile.load(out).n == 40

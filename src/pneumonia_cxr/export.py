"""Export a trained run to ONNX for torch-free serving.

The exported graph includes input normalisation and temperature scaling:
``image`` float32 (N, 1, H, W) in [0, 1]  ->  ``probability`` float32 (N,) calibrated P(pneumonia).
A ``model_meta.json`` sidecar stores the decision threshold and provenance.

With ``fp16_weights=True`` (default) large weight tensors are *stored* in float16 and cast back
to float32 when the session is created (ONNX Runtime constant-folds the casts), so files are
half the size while every computation still runs in float32.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import __version__
from .train import load_checkpoint

FP16_MIN_ELEMENTS = 1024  # small tensors (norms, biases, layer scales) stay float32


class CalibratedModel(nn.Module):
    def __init__(self, model: nn.Module, temperature: float):
        super().__init__()
        self.model = model
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(image) / self.temperature)


def store_onnx_weights_fp16(onnx_path: Path, min_elements: int = FP16_MIN_ELEMENTS) -> int:
    """Replace large float32 initializers by float16 ones followed by a Cast to float32."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(onnx_path))
    graph = model.graph
    graph_inputs = {i.name for i in graph.input}
    casts, converted = [], 0
    for init in list(graph.initializer):
        if init.data_type != TensorProto.FLOAT or init.name in graph_inputs:
            continue
        if int(np.prod(init.dims)) < min_elements:
            continue
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), f"{init.name}__fp16")
        casts.append(
            helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT, name=f"{init.name}__cast")
        )
        graph.initializer.remove(init)
        graph.initializer.append(half)
        converted += 1
    for i, node in enumerate(casts):
        graph.node.insert(i, node)
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
    return converted


def save_checkpoint_fp16(src: Path, dst: Path, min_elements: int = FP16_MIN_ELEMENTS) -> None:
    """Copy of a run checkpoint with large float tensors stored in float16.
    ``load_state_dict`` casts them back to the float32 parameters on load."""
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    ckpt["state_dict"] = {
        k: v.half() if v.is_floating_point() and v.numel() >= min_elements else v for k, v in ckpt["state_dict"].items()
    }
    ckpt["weights_storage"] = "float16 (computed in float32)"
    torch.save(ckpt, dst)


def export_onnx(run_dir: Path, out_dir: Path, opset: int = 18, check: bool = True, fp16_weights: bool = True) -> Path:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_checkpoint(run_dir, torch.device("cpu"))
    size = ckpt["config"]["img_size"]
    wrapped = CalibratedModel(model, ckpt["temperature"]).eval()
    dummy = torch.rand(2, 1, size, size)
    onnx_path = out_dir / "model.onnx"

    torch.onnx.export(
        wrapped,
        (dummy,),
        str(onnx_path),
        input_names=["image"],
        output_names=["probability"],
        dynamic_shapes={"image": {0: torch.export.Dim("batch", min=1, max=256)}},
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )
    if fp16_weights:
        n = store_onnx_weights_fp16(onnx_path)
        print(f"Stored {n} weight tensors as float16")

    eval_summary = {}
    for source in ("kermany", "rsna"):
        f = run_dir / f"eval_{source}_test.json"
        if f.exists():
            m = json.loads(f.read_text())
            eval_summary[source] = {k: m[k] for k in ("auroc", "sensitivity", "specificity", "accuracy", "ece")}
    meta = {
        "name": ckpt["config"]["name"],
        "arch": ckpt["config"]["arch"],
        "train_sources": ckpt["config"]["sources"],
        "img_size": size,
        "cache_size": 256,
        "threshold": ckpt["threshold"],
        "temperature": ckpt["temperature"],
        "input": "float32 (N, 1, H, W) grayscale in [0, 1]",
        "output": "calibrated P(pneumonia) per image",
        "weights_storage": "float16 (computed in float32)" if fp16_weights else "float32",
        "test_metrics": eval_summary,
        "package_version": __version__,
        "intended_use": "Research and education only. Not a medical device.",
    }
    # PyTorch checkpoint next to the ONNX file: needed by the Gradio demo for Grad-CAM.
    if fp16_weights:
        save_checkpoint_fp16(run_dir / "model.pt", out_dir / "model.pt")
    else:
        torch.save(ckpt, out_dir / "model.pt")

    if check:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        x = np.random.default_rng(0).random((4, 1, size, size), dtype=np.float32)
        ref = wrapped(torch.from_numpy(x)).detach().numpy()
        got = sess.run(None, {"image": x})[0]
        diff = float(np.abs(ref - got).max())
        print(f"ONNX parity check: max |torch fp32 - onnxruntime| = {diff:.2e}")
        if diff > (1e-2 if fp16_weights else 1e-3):
            raise AssertionError(f"ONNX output differs from PyTorch by {diff}")
        meta["onnx_parity_max_abs_diff"] = diff

        one = x[:1]
        for _ in range(3):
            sess.run(None, {"image": one})
        times = []
        for _ in range(30):
            t = time.perf_counter()
            sess.run(None, {"image": one})
            times.append((time.perf_counter() - t) * 1000)
        meta["onnx_cpu_latency_ms_batch1"] = round(float(np.median(times)), 1)
        print(f"ONNX Runtime CPU latency (batch 1, median of 30): {meta['onnx_cpu_latency_ms_batch1']} ms")
    meta["onnx_size_mb"] = round(onnx_path.stat().st_size / 1e6, 1)
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    return onnx_path


def export_browser_onnx(run_dir: Path, out_path: Path, opset: int = 18, fp16_weights: bool = True) -> Path:
    """ONNX graph with two outputs, ``probability`` (N,) and ``cam`` (N, H, W) in [0, 1], for
    in-browser inference with ONNX Runtime Web (Grad-CAM in closed form, no backward pass)."""
    from .gradcam import ClosedFormGradCAM, GradCAM

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_checkpoint(run_dir, torch.device("cpu"))
    size = ckpt["config"]["img_size"]
    wrapped = ClosedFormGradCAM(model, ckpt["temperature"]).eval()
    x = torch.rand(2, 1, size, size, generator=torch.Generator().manual_seed(0))

    with torch.no_grad():
        prob, cam = wrapped(x)
    with GradCAM(model) as gc:  # the closed form must equal autograd Grad-CAM
        ref_cam, ref_logit = gc(x)
    cam_diff = float((cam - ref_cam).abs().max())
    prob_diff = float((prob - torch.sigmoid(ref_logit / ckpt["temperature"])).abs().max())
    if cam_diff > 1e-4 or prob_diff > 1e-5:
        raise AssertionError(f"closed-form Grad-CAM mismatch: cam {cam_diff:.2e}, prob {prob_diff:.2e}")

    torch.onnx.export(
        wrapped,
        (x,),
        str(out_path),
        input_names=["image"],
        output_names=["probability", "cam"],
        dynamic_shapes={"image": {0: torch.export.Dim("batch", min=1, max=64)}},
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )
    if fp16_weights:
        store_onnx_weights_fp16(out_path)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    got_prob, got_cam = sess.run(None, {"image": x.numpy()})
    diff = max(float(np.abs(got_prob - prob.numpy()).max()), float(np.abs(got_cam - cam.numpy()).max()))
    print(f"Browser ONNX: closed-form vs autograd Grad-CAM {cam_diff:.1e}; torch vs onnxruntime {diff:.1e}")
    if diff > 2e-2:
        raise AssertionError(f"ONNX output differs from PyTorch by {diff}")
    return out_path

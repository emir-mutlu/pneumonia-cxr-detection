"""Interactive demo: upload a chest X-ray, get a calibrated probability and a Grad-CAM map.

python app/gradio_app.py --model-dir models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import torch
from PIL import Image

from pneumonia_cxr.data.imaging import preprocess_for_model
from pneumonia_cxr.gradcam import GradCAM
from pneumonia_cxr.inference import DISCLAIMER
from pneumonia_cxr.train import load_checkpoint

_CMAP = matplotlib.colormaps["inferno"]
EXAMPLES = Path(__file__).parent / "examples"


def build_demo(model_dir: Path) -> gr.Blocks:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(model_dir, device)
    size, temperature, threshold = ckpt["config"]["img_size"], ckpt["temperature"], ckpt["threshold"]

    def analyse(image: Image.Image | None, overlay_alpha: float):
        if image is None:
            return None, None, ""
        x = torch.from_numpy(preprocess_for_model(image, size)).to(device)
        with GradCAM(model) as cam:
            heat, logit = cam(x)
        p = float(torch.sigmoid(logit.float() / temperature)[0])
        gray = x[0, 0].cpu().numpy()
        heat_rgb = _CMAP(heat[0].cpu().numpy())[..., :3]
        blend = (1 - overlay_alpha) * np.repeat(gray[..., None], 3, -1) + overlay_alpha * heat_rgb
        verdict = "PNEUMONIA" if p >= threshold else "NORMAL"
        text = (
            f"### {verdict}\nCalibrated P(pneumonia) = **{p:.3f}** (decision threshold {threshold:.3f}).\n\n"
            f"_{DISCLAIMER}_"
        )
        return {"Pneumonia": p, "Normal": 1 - p}, Image.fromarray((blend * 255).astype(np.uint8)), text

    with gr.Blocks(title="Pneumonia CXR") as demo:
        gr.Markdown(
            "# Pneumonia detection from chest X-rays\n"
            f"Model: `{ckpt['config']['arch']}` trained on {', '.join(ckpt['config']['sources'])}. "
            "Grad-CAM shows the regions that drove the prediction."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Chest X-ray (PA/AP, PNG or JPEG)", image_mode="L")
                alpha = gr.Slider(0.0, 0.8, value=0.45, label="Heatmap opacity")
                btn = gr.Button("Analyse", variant="primary")
                examples = sorted(str(p) for p in EXAMPLES.glob("*.*") if p.suffix in {".png", ".jpeg", ".jpg"})
                if examples:
                    gr.Examples(examples, inputs=inp, label="Held-out test images (see app/examples/README.md)")
            with gr.Column():
                label = gr.Label(num_top_classes=2, label="Probability")
                cam_img = gr.Image(label="Grad-CAM", type="pil")
                verdict = gr.Markdown()
        btn.click(analyse, [inp, alpha], [label, cam_img, verdict])
        inp.change(analyse, [inp, alpha], [label, cam_img, verdict])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Folder containing model.pt")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_demo(args.model_dir).launch(server_port=args.port, share=args.share)

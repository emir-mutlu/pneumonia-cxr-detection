"""Quantitative and visual evaluation of Grad-CAM explanations.

RSNA provides radiologist bounding boxes for every Lung Opacity image, so instead of only
showing hand-picked heatmaps we measure, over all positive test images:

* **pointing game**: does the CAM peak fall inside a radiologist box?
* **energy in box**: share of the CAM mass inside the boxes,
  compared with the share expected from a uniform (uninformative) map = box area fraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

from .data.dataset import make_dataset
from .gradcam import GradCAM, boxes_to_mask, localisation_scores
from .metrics import sigmoid
from .paths import DATA_PROCESSED
from .train import get_device, load_checkpoint


def localisation_eval(run_dir: Path, processed_dir: Path = DATA_PROCESSED, batch_size: int = 32) -> dict:
    run_dir = Path(run_dir)
    device = get_device()
    model, ckpt = load_checkpoint(run_dir, device)
    size = ckpt["config"]["img_size"]
    ds = make_dataset("rsna", "test", False, size, processed_dir)
    pos = np.where(ds.labels == 1)[0]

    rows = []
    with GradCAM(model) as cam:
        for start in range(0, len(pos), batch_size):
            ids = pos[start : start + batch_size]
            x = torch.stack([ds[i][0] for i in ids]).to(device)
            heat, logits = cam(x)
            heat = heat.cpu().numpy()
            probs = sigmoid(logits.float().cpu().numpy() / ckpt["temperature"])
            for k, i in enumerate(ids):
                r = ds.meta.iloc[i]
                mask = boxes_to_mask(json.loads(r["boxes"]), r["orig_w"], r["orig_h"], size)
                s = localisation_scores(heat[k], mask)
                s.update(idx=int(r["idx"]), prob=float(probs[k]), detected=bool(probs[k] >= ckpt["threshold"]))
                rows.append(s)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "localisation_rsna_test.csv", index=False)

    def summary(d: pd.DataFrame) -> dict:
        return {
            "n": int(len(d)),
            "pointing_game_hit_rate": float(d["hit"].mean()),
            "random_point_hit_rate": float(d["box_fraction"].mean()),
            "energy_in_box": float(d["energy_in_box"].mean()),
            "uniform_map_energy_in_box": float(d["box_fraction"].mean()),
        }

    result = {"all_positives": summary(df), "detected_positives": summary(df[df["detected"]])}
    (run_dir / "localisation_rsna_test.json").write_text(json.dumps(result, indent=2))
    a = result["all_positives"]
    print(
        f"[{run_dir.name}] pointing game {a['pointing_game_hit_rate']:.3f} (random {a['random_point_hit_rate']:.3f}) | "
        f"energy in box {a['energy_in_box']:.3f} (uniform {a['uniform_map_energy_in_box']:.3f})"
    )
    return result


def plot_cam_comparison(
    run_dirs: list[Path],
    labels: list[str],
    out_file: Path,
    processed_dir: Path = DATA_PROCESSED,
    n: int = 6,
    seed: int = 11,
) -> Path:
    """Same RSNA test positives, one row of Grad-CAM overlays per model, radiologist boxes in blue."""
    device = get_device()
    models = [load_checkpoint(Path(r), device) for r in run_dirs]
    size = models[0][1]["config"]["img_size"]
    ds = make_dataset("rsna", "test", False, size, processed_dir)
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.where(ds.labels == 1)[0], n, replace=False)
    scores = [json.loads((Path(r) / "localisation_rsna_test.json").read_text()) for r in run_dirs]

    fig, axes = plt.subplots(len(models) + 1, n, figsize=(2.2 * n, 2.3 * (len(models) + 1)))
    for col, i in enumerate(ids):
        x = ds[i][0].unsqueeze(0).to(device)
        r = ds.meta.iloc[i]
        sx, sy = size / r["orig_w"], size / r["orig_h"]
        img = x[0, 0].cpu().numpy()
        axes[0, col].imshow(img, cmap="gray")
        for row, (model, _ckpt) in enumerate(models, start=1):
            with GradCAM(model) as cam:
                heat, _ = cam(x)
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].imshow(heat[0].cpu().numpy(), cmap="inferno", alpha=0.45)
        for row in range(len(models) + 1):
            for bx, by, bw, bh in json.loads(r["boxes"]):
                axes[row, col].add_patch(
                    Rectangle((bx * sx, by * sy), bw * sx, bh * sy, fill=False, ec="#4fc3f7", lw=1.3)
                )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("input + boxes", fontsize=9)
    for row, (label, s) in enumerate(zip(labels, scores, strict=True), start=1):
        hit = s["all_positives"]["pointing_game_hit_rate"]
        axes[row, 0].set_ylabel(f"{label}\npointing game {hit:.2f}", fontsize=9)
    fig.suptitle(
        "Grad-CAM on RSNA test pneumonia (blue: radiologist boxes); pointing game over all positives",
        fontsize=10,
    )
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_examples(
    run_dir: Path, out_file: Path, processed_dir: Path = DATA_PROCESSED, n_per_source: int = 4, seed: int = 3
) -> Path:
    """Grid: chest X-ray, Grad-CAM overlay (+ radiologist boxes on RSNA)."""
    run_dir = Path(run_dir)
    device = get_device()
    model, ckpt = load_checkpoint(run_dir, device)
    size = ckpt["config"]["img_size"]
    rng = np.random.default_rng(seed)

    panels = []
    for source in ("kermany", "rsna"):
        ds = make_dataset(source, "test", False, size, processed_dir)
        preds_file = run_dir / f"preds_{source}_test.csv"
        candidates = np.where(ds.labels == 1)[0]
        if preds_file.exists():  # show true positives: the explanation of a correct call
            p = pd.read_csv(preds_file)["prob"].to_numpy()
            candidates = np.where((ds.labels == 1) & (p >= ckpt["threshold"]))[0]
        for i in rng.choice(candidates, n_per_source, replace=False):
            panels.append((source, ds, int(i)))

    fig, axes = plt.subplots(2, len(panels), figsize=(2.6 * len(panels), 5.6))
    with GradCAM(model) as cam:
        for col, (source, ds, i) in enumerate(panels):
            x = ds[i][0].unsqueeze(0).to(device)
            heat, logit = cam(x)
            prob = float(sigmoid(logit.float().cpu().numpy() / ckpt["temperature"])[0])
            img = x[0, 0].cpu().numpy()
            axes[0, col].imshow(img, cmap="gray")
            axes[0, col].set_title(f"{source} · P={prob:.2f}", fontsize=9)
            axes[1, col].imshow(img, cmap="gray")
            axes[1, col].imshow(heat[0].cpu().numpy(), cmap="inferno", alpha=0.45)
            if source == "rsna":
                r = ds.meta.iloc[i]
                for bx, by, bw, bh in json.loads(r["boxes"]):
                    sx, sy = size / r["orig_w"], size / r["orig_h"]
                    axes[1, col].add_patch(
                        Rectangle((bx * sx, by * sy), bw * sx, bh * sy, fill=False, ec="#4fc3f7", lw=1.4)
                    )
            for ax in axes[:, col]:
                ax.axis("off")
    axes[0, 0].text(-0.08, 0.5, "input", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.08, 0.5, "Grad-CAM", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
    fig.suptitle("Grad-CAM on correctly detected pneumonia (blue: radiologist boxes, RSNA)", fontsize=10)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_file

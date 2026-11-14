"""Evaluate a trained run on the held-out test split of every source.

For each source we report threshold-free metrics (AUROC, AUPRC), metrics at the operating
point chosen on the *training sources'* validation data (so external results show the
real cost of a threshold that was not tuned for the new hospital), patient-clustered
bootstrap 95 % CIs, and calibration before / after temperature scaling. On RSNA we add
subgroup results (sex, age, view position) and Lung Opacity vs strictly Normal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from .data.dataset import make_dataset
from .metrics import binary_metrics, bootstrap_ci, sigmoid
from .paths import DATA_PROCESSED, SOURCES
from .train import _loader, get_device, load_checkpoint, predict_logits


def _age_group(age: int) -> str:
    if age < 0 or age > 110:
        return "unknown"
    if age < 40:
        return "<40"
    if age < 60:
        return "40-59"
    return "60+"


def subgroup_metrics(df: pd.DataFrame, threshold: float) -> dict[str, dict]:
    out: dict[str, dict] = {}
    groupings = {
        "sex": df["sex"],
        "age": df["age"].map(_age_group),
        "view": df["view"],
    }
    for name, key in groupings.items():
        for value, g in df.groupby(key):
            if value == "unknown" or g["label"].nunique() < 2 or len(g) < 50:
                continue
            m = binary_metrics(g["label"].to_numpy(), g["prob"].to_numpy(), threshold)
            out[f"{name}={value}"] = {k: m[k] for k in ("n", "prevalence", "auroc", "sensitivity", "specificity")}
    strict = df[df["rsna_class"].isin(["Lung Opacity", "Normal"])]
    m = binary_metrics(strict["label"].to_numpy(), strict["prob"].to_numpy(), threshold)
    out["lung_opacity_vs_normal_only"] = {k: m[k] for k in ("n", "prevalence", "auroc", "sensitivity", "specificity")}
    return out


def evaluate_run(
    run_dir: Path,
    sources: tuple[str, ...] = SOURCES,
    split: str = "test",
    processed_dir: Path = DATA_PROCESSED,
    n_boot: int = 1000,
) -> dict[str, dict]:
    run_dir = Path(run_dir)
    device = get_device()
    model, ckpt = load_checkpoint(run_dir, device)
    cfg = ckpt["config"]
    temperature, threshold = ckpt["temperature"], ckpt["threshold"]
    results = {}
    for source in sources:
        ds = make_dataset(source, split, False, cfg["img_size"], processed_dir)
        logits = predict_logits(model, _loader(ds, 64, False, 2), device)
        y = ds.labels.astype(int)
        p_raw, p_cal = sigmoid(logits), sigmoid(logits / temperature)

        preds = ds.meta.copy()
        preds["logit"], preds["prob"] = logits, p_cal
        preds.to_csv(run_dir / f"preds_{source}_{split}.csv", index=False)

        metrics = binary_metrics(y, p_cal, threshold)
        metrics["ece_uncalibrated"] = binary_metrics(y, p_raw, 0.5)["ece"]
        metrics["in_distribution"] = source in cfg["sources"]
        metrics["ci95"] = bootstrap_ci(y, p_cal, threshold, n_boot=n_boot, groups=preds["patient_id"].to_numpy())
        if source == "rsna":
            metrics["subgroups"] = subgroup_metrics(preds, threshold)
        results[source] = metrics
        (run_dir / f"eval_{source}_{split}.json").write_text(json.dumps(metrics, indent=2))
        tag = "in-distribution" if metrics["in_distribution"] else "EXTERNAL"
        lo, hi = metrics["ci95"]["auroc"]
        print(
            f"[{run_dir.name}] {source:8s} ({tag}): AUROC {metrics['auroc']:.4f} [{lo:.4f}-{hi:.4f}] "
            f"sens {metrics['sensitivity']:.3f} spec {metrics['specificity']:.3f} "
            f"acc {metrics['accuracy']:.3f} ECE {metrics['ece_uncalibrated']:.3f}->{metrics['ece']:.3f}"
        )
    torch.cuda.empty_cache()
    return results

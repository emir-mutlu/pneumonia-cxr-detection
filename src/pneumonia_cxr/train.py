"""Training loop: AdamW + warmup/cosine schedule, bf16 autocast, early stopping on validation
AUROC, then temperature scaling and threshold selection on the validation set only."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .data.dataset import CXRDataset, make_dataset
from .metrics import binary_metrics, fit_temperature, sigmoid, youden_threshold
from .models import build_model, count_parameters
from .paths import DATA_PROCESSED, RUNS


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global RNG used by third-party code
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")


def _loader(ds, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            out.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _lr_lambda(warmup_steps: int, total_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return f


def train(cfg: TrainConfig, runs_dir: Path = RUNS, processed_dir: Path = DATA_PROCESSED, mlflow: bool = False) -> Path:
    seed_everything(cfg.seed)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    run_dir = runs_dir / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    train_sets = [
        make_dataset(s, "train", True, cfg.img_size, processed_dir, cfg.max_train_samples, cfg.seed)
        for s in cfg.sources
    ]
    val_sets: dict[str, CXRDataset] = {
        s: make_dataset(
            s, "val", False, cfg.img_size, processed_dir,
            None if cfg.max_train_samples is None else max(64, cfg.max_train_samples // 4), cfg.seed,
        )
        for s in cfg.sources
    }  # fmt: skip
    train_ds = ConcatDataset(train_sets)
    train_loader = _loader(train_ds, cfg.batch_size, True, cfg.num_workers)
    val_workers = min(2, cfg.num_workers)  # eval transforms are cheap; fewer spawned processes
    val_loaders = {s: _loader(ds, cfg.batch_size * 2, False, val_workers) for s, ds in val_sets.items()}

    model = build_model(cfg.arch, cfg.pretrained, cfg.drop_path_rate).to(device)
    model = model.to(memory_format=torch.channels_last)
    n_params = count_parameters(model)
    print(f"[{cfg.name}] {cfg.arch}: {n_params / 1e6:.1f}M params | train {len(train_ds)} images | device {device}")

    optimizer = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(int(cfg.warmup_epochs * steps_per_epoch), total_steps)
    )
    loss_fn = nn.BCEWithLogitsLoss()
    amp = cfg.amp and device.type == "cuda"

    tracker = None
    if mlflow:
        import mlflow as tracker

        tracker.set_experiment("pneumonia-cxr")
        tracker.start_run(run_name=cfg.name)
        tracker.log_params({k: str(v) for k, v in cfg.to_dict().items()})

    history = []
    best_score, best_epoch, bad_epochs = -1.0, -1, 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        running, seen = 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False)
        for x, y in bar:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = loss_fn(model(x).float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(y)
            seen += len(y)
            bar.set_postfix(loss=f"{running / seen:.4f}")

        row = {"epoch": epoch + 1, "train_loss": running / seen, "lr": scheduler.get_last_lr()[0]}
        aurocs = []
        for s, loader in val_loaders.items():
            logits = predict_logits(model, loader, device, amp)
            y_val = val_sets[s].labels
            m = binary_metrics(y_val, sigmoid(logits), 0.5)
            p = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)
            row[f"val_loss_{s}"] = float(-np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p)))
            row[f"val_auroc_{s}"] = m["auroc"]
            aurocs.append(m["auroc"])
        # Mean of per-source AUROCs: pooling sources would reward separating hospitals.
        row["val_score"] = float(np.mean(aurocs))
        row["elapsed_s"] = time.time() - t0
        history.append(row)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
        if tracker:
            tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch + 1)

        if row["val_score"] > best_score:
            best_score, best_epoch, bad_epochs = row["val_score"], epoch + 1, 0
            torch.save(model.state_dict(), run_dir / "best_weights.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stopping after epoch {epoch + 1} (best epoch {best_epoch}).")
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # Post-hoc calibration and operating point, on validation data only.
    model.load_state_dict(torch.load(run_dir / "best_weights.pt", map_location=device, weights_only=True))
    val_logits = np.concatenate([predict_logits(model, val_loaders[s], device, amp) for s in cfg.sources])
    val_y = np.concatenate([val_sets[s].labels for s in cfg.sources])
    temperature = fit_temperature(val_logits, val_y)
    val_probs = sigmoid(val_logits / temperature)
    threshold = youden_threshold(val_y, val_probs)
    val_metrics = binary_metrics(val_y, val_probs, threshold)

    checkpoint = {
        "config": cfg.to_dict(),
        "state_dict": model.state_dict(),
        "temperature": temperature,
        "threshold": threshold,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "n_params": n_params,
        "train_seconds": time.time() - t0,
    }
    torch.save(checkpoint, run_dir / "model.pt")
    (run_dir / "best_weights.pt").unlink(missing_ok=True)
    summary = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[{cfg.name}] best epoch {best_epoch} | T={temperature:.3f} | threshold={threshold:.3f} | "
        f"val AUROC {val_metrics['auroc']:.4f}"
    )
    if tracker:
        tracker.log_metrics(
            {"temperature": temperature, "threshold": threshold, "val_auroc_final": val_metrics["auroc"]}
        )
        tracker.log_artifact(str(run_dir / "train_summary.json"))
        tracker.end_run()
    return run_dir


def load_checkpoint(run_dir: Path, device: torch.device | None = None) -> tuple[nn.Module, dict]:
    device = device or get_device()
    ckpt = torch.load(Path(run_dir) / "model.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["arch"], pretrained=False, drop_path_rate=0.0)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt

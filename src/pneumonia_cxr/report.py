"""Aggregate every evaluated run into ``reports/results.md``, ``reports/results.json`` and figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from .metrics import sigmoid
from .paths import REPORTS, RUNS

# Validated categorical palette (fixed order) and recessive chart chrome, light surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

ARCH_NAMES = {
    "simple_cnn": "Simple CNN (scratch)",
    "resnet50.a1_in1k": "ResNet-50 (ImageNet)",
    "efficientnet_b0.ra_in1k": "EfficientNet-B0 (ImageNet)",
    "xrv_densenet121": "DenseNet-121 (CXR-pretrained)",
    "convnext_tiny.fb_in22k_ft_in1k": "ConvNeXt-T (ImageNet-22k)",
}
SOURCE_NAMES = {"kermany": "Kermany (pediatric)", "rsna": "RSNA (adult)"}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "text.color": INK,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "lines.linewidth": 2.0,
            "savefig.facecolor": SURFACE,
        }
    )


def model_label(cfg: dict) -> str:
    arch = ARCH_NAMES.get(cfg["arch"], cfg["arch"])
    sources = "+".join(s.capitalize() if s == "kermany" else s.upper() for s in cfg["sources"])
    return f"{arch} · trained on {sources}"


def collect_runs(runs_dir: Path = RUNS) -> list[dict]:
    runs = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if (p / "train_summary.json").exists()):
        summary = json.loads((run_dir / "train_summary.json").read_text())
        evals = {
            src: json.loads((run_dir / f"eval_{src}_test.json").read_text())
            for src in ("kermany", "rsna")
            if (run_dir / f"eval_{src}_test.json").exists()
        }
        loc_file = run_dir / "localisation_rsna_test.json"
        runs.append(
            {
                "name": run_dir.name,
                "dir": run_dir,
                "config": summary["config"],
                "label": model_label(summary["config"]),
                "summary": summary,
                "eval": evals,
                "localisation": json.loads(loc_file.read_text()) if loc_file.exists() else None,
            }
        )
    return sorted(runs, key=_run_order)


def _run_order(run: dict) -> tuple[int, int, str]:
    """Single-source runs first (Kermany, then RSNA), multi-source last; within, by architecture."""
    sources = run["config"]["sources"]
    group = 2 if len(sources) > 1 else (0 if sources == ["kermany"] else 1)
    archs = list(ARCH_NAMES)
    arch = archs.index(run["config"]["arch"]) if run["config"]["arch"] in archs else len(archs)
    return group, arch, run["name"]


def _fmt_ci(m: dict, key: str) -> str:
    lo, hi = m["ci95"][key]
    return f"{m[key]:.3f} ({lo:.3f}–{hi:.3f})"


def results_table(runs: list[dict]) -> str:
    lines = [
        "| Model | Train data | Params | Kermany test AUROC (95% CI) | Sens / Spec | RSNA test AUROC (95% CI) | Sens / Spec |",
        "|---|---|---:|---|---|---|---|",
    ]
    for r in runs:
        cells = [ARCH_NAMES.get(r["config"]["arch"], r["config"]["arch"]), " + ".join(r["config"]["sources"]),
                 f"{r['summary']['n_params'] / 1e6:.1f}M"]  # fmt: skip
        for src in ("kermany", "rsna"):
            m = r["eval"].get(src)
            if m is None:
                cells += ["–", "–"]
                continue
            star = "" if m["in_distribution"] else " †"
            cells += [_fmt_ci(m, "auroc") + star, f"{m['sensitivity']:.3f} / {m['specificity']:.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "† external test set: hospital / population never seen in training; threshold transferred from the training data's validation split."
    )
    return "\n".join(lines)


def calibration_table(runs: list[dict]) -> str:
    lines = [
        "| Model | Train data | Test set | ECE before | ECE after temperature scaling | Brier |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in runs:
        for src, m in r["eval"].items():
            if m["in_distribution"]:
                lines.append(
                    f"| {ARCH_NAMES.get(r['config']['arch'], r['config']['arch'])} | {' + '.join(r['config']['sources'])} | "
                    f"{src} | {m['ece_uncalibrated']:.3f} | {m['ece']:.3f} | {m['brier']:.3f} |"
                )
    return "\n".join(lines)


def localisation_table(runs: list[dict]) -> str:
    lines = [
        "| Model | Train data | Pointing game hit rate | Random-point baseline | CAM energy in boxes | Uniform-map baseline |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in runs:
        loc = r["localisation"]
        if loc is None:
            continue
        a = loc["all_positives"]
        lines.append(
            f"| {ARCH_NAMES.get(r['config']['arch'], r['config']['arch'])} | {' + '.join(r['config']['sources'])} | "
            f"{a['pointing_game_hit_rate']:.3f} | {a['random_point_hit_rate']:.3f} | {a['energy_in_box']:.3f} | {a['uniform_map_energy_in_box']:.3f} |"
        )
    return "\n".join(lines)


def subgroup_table(run: dict) -> str:
    sub = run["eval"]["rsna"]["subgroups"]
    lines = [
        "| Subgroup (RSNA test) | n | Prevalence | AUROC | Sensitivity | Specificity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, m in sub.items():
        lines.append(
            f"| {k} | {m['n']} | {m['prevalence']:.2f} | {m['auroc']:.3f} | {m['sensitivity']:.3f} | {m['specificity']:.3f} |"
        )
    return "\n".join(lines)


def plot_roc(runs: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, src in zip(axes, ("kermany", "rsna"), strict=True):
        for color, r in zip(SERIES, runs, strict=False):
            f = r["dir"] / f"preds_{src}_test.csv"
            if not f.exists():
                continue
            p = pd.read_csv(f)
            fpr, tpr, _ = roc_curve(p["label"], p["prob"])
            m = r["eval"][src]
            ls = "-" if m["in_distribution"] else (0, (4, 2))
            ax.plot(fpr, tpr, color=color, ls=ls, label=f"{r['label']}  AUROC {m['auroc']:.3f}")
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=":")
        ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate (1 − specificity)", ylabel="Sensitivity")
        ax.set_title(f"{SOURCE_NAMES[src]} test set", loc="left")
        ax.legend(loc="lower right", fontsize=7.2)
    fig.text(
        0.01, -0.02, "Dashed lines: external evaluation (test hospital not seen in training).", color=MUTED, fontsize=8
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_domain_shift(runs: list[dict], out: Path) -> None:
    """Dumbbell: AUROC on each test set, one row per model."""
    rows = [r for r in runs if {"kermany", "rsna"} <= set(r["eval"])]
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(rows) + 1.4))
    for y, r in enumerate(rows):
        k, s = r["eval"]["kermany"]["auroc"], r["eval"]["rsna"]["auroc"]
        ax.plot([k, s], [y, y], color=AXIS, lw=2, zorder=1)
        close = abs(k - s) < 0.02  # push the two labels apart when the dots nearly touch
        for value, other, color, src in ((k, s, SERIES[0], "kermany"), (s, k, SERIES[1], "rsna")):
            external = not r["eval"][src]["in_distribution"]
            ax.scatter(
                [value],
                [y],
                s=70,
                facecolor=SURFACE if external else color,
                edgecolor=color,
                linewidth=2,
                zorder=2,
            )
            left = close and value < other
            ax.annotate(
                f"{value:.3f}",
                (value, y),
                xytext=((-5 if left else 5) if close else 0, 9),
                textcoords="offset points",
                ha=("right" if left else "left") if close else "center",
                fontsize=7.5,
                color=INK_2,
            )
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=8, color=SERIES[0], label="Kermany test (pediatric)"),
        plt.Line2D([], [], ls="", marker="o", ms=8, color=SERIES[1], label="RSNA test (adult)"),
        plt.Line2D([], [], ls="", marker="o", ms=8, mfc=SURFACE, mec=MUTED, mew=2, label="external (not in training)"),
    ]
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=8.5, color=INK_2)
    ax.set_ylim(len(rows) - 0.4, -0.7)
    ax.set_xlabel("AUROC")
    ax.grid(axis="y", visible=False)
    ax.set_title("AUROC on each hospital's test set", loc="left")
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(run: dict, out: Path, n_bins: int = 10) -> None:
    srcs = [s for s, m in run["eval"].items() if m["in_distribution"]]
    fig, axes = plt.subplots(1, len(srcs), figsize=(4.6 * len(srcs), 4.3), squeeze=False)
    T = run["summary"]["temperature"]
    for ax, src in zip(axes[0], srcs, strict=True):
        p = pd.read_csv(run["dir"] / f"preds_{src}_test.csv")
        y = p["label"].to_numpy()
        for color, name, probs in ((SERIES[1], "before (raw)", sigmoid(p["logit"].to_numpy())),
                                   (SERIES[0], f"after temperature scaling (T={T:.2f})", p["prob"].to_numpy())):  # fmt: skip
            bins = np.linspace(0, 1, n_bins + 1)
            ids = np.clip(np.digitize(probs, bins[1:-1]), 0, n_bins - 1)
            xs, ys = [], []
            for b in range(n_bins):
                m = ids == b
                if m.sum() >= 10:
                    xs.append(probs[m].mean())
                    ys.append(y[m].mean())
            ax.plot(xs, ys, marker="o", ms=5, color=color, label=name)
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=":", label="perfect calibration")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted P(pneumonia)", ylabel="Observed frequency")
        ax.set_title(f"Reliability · {SOURCE_NAMES[src]} test", loc="left")
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(run: dict, out: Path) -> None:
    srcs = list(run["eval"])
    fig, axes = plt.subplots(1, len(srcs), figsize=(3.9 * len(srcs), 3.6), squeeze=False)
    thr = run["summary"]["threshold"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    for ax, src in zip(axes[0], srcs, strict=True):
        p = pd.read_csv(run["dir"] / f"preds_{src}_test.csv")
        cm = confusion_matrix(p["label"], p["prob"] >= thr, labels=[0, 1])
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.1%}", ha="center", va="center", fontsize=9.5,
                        color="#ffffff" if share[i, j] > 0.55 else INK)  # fmt: skip
        ax.set_xticks([0, 1], ["Normal", "Pneumonia"])
        ax.set_yticks([0, 1], ["Normal", "Pneumonia"])
        ax.set(xlabel="Predicted", ylabel="Actual")
        ax.grid(False)
        tag = "" if run["eval"][src]["in_distribution"] else " (external)"
        ax.set_title(f"{SOURCE_NAMES[src]}{tag}", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_history(run: dict, out: Path) -> None:
    h = pd.read_csv(run["dir"] / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(h["epoch"], h["train_loss"], color=SERIES[0], label="train")
    for color, src in zip(SERIES[1:], run["config"]["sources"], strict=False):
        axes[0].plot(h["epoch"], h[f"val_loss_{src}"], color=color, label=f"val · {src}")
        axes[1].plot(h["epoch"], h[f"val_auroc_{src}"], color=color, label=f"val · {src}")
    axes[0].set(xlabel="Epoch", ylabel="BCE loss")
    axes[0].set_title("Loss", loc="left")
    axes[1].set(xlabel="Epoch", ylabel="AUROC")
    axes[1].set_title("Validation AUROC", loc="left")
    best = run["summary"]["best_epoch"]
    for ax in axes:
        ax.axvline(best, color=MUTED, lw=1, ls=":")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_report(runs_dir: Path = RUNS, out_dir: Path = REPORTS, featured: str | None = None) -> Path:
    _style()
    runs = [r for r in collect_runs(runs_dir) if r["eval"]]
    if not runs:
        raise RuntimeError("No evaluated runs found. Run `pneumo evaluate` first.")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_roc(runs, fig_dir / "roc_curves.png")
    plot_domain_shift(runs, fig_dir / "domain_shift.png")
    feat = next((r for r in runs if r["name"] == featured), runs[-1])
    plot_reliability(feat, fig_dir / "reliability.png")
    plot_confusion(feat, fig_dir / "confusion_matrices.png")
    plot_history(feat, fig_dir / "training_curves.png")

    sections = [
        "# Results\n",
        "_Generated by `pneumo report` from the files in `runs/`. Do not edit by hand._\n",
        "## Discrimination (held-out test sets)\n",
        results_table(runs),
        "\n## Calibration (in-distribution test sets)\n",
        calibration_table(runs),
        "\n## Grad-CAM localisation vs radiologist boxes (RSNA test, all Lung Opacity images)\n",
        localisation_table(runs),
    ]
    if "rsna" in feat["eval"] and "subgroups" in feat["eval"]["rsna"]:
        sections += [f"\n## Subgroups – {feat['label']}\n", subgroup_table(feat)]
    md = out_dir / "results.md"
    md.write_text("\n".join(sections) + "\n", encoding="utf-8")

    serialisable = [{k: v for k, v in r.items() if k != "dir"} for r in runs]
    (out_dir / "results.json").write_text(json.dumps(serialisable, indent=2, default=str))
    print(f"Wrote {md} and {len(list(fig_dir.glob('*.png')))} figures to {fig_dir}")
    return md

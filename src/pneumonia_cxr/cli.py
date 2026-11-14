"""Command line interface: ``pneumo --help``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from .paths import DATA_RAW, MODELS, REPORTS, RUNS, SOURCES

# Windows consoles default to cp1252; progress bars of some libraries print Unicode blocks.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Pneumonia detection from chest X-rays.")


def _sources(source: str) -> list[str]:
    if source == "all":
        return list(SOURCES)
    if source not in SOURCES:
        raise typer.BadParameter(f"source must be one of {SOURCES} or 'all'")
    return [source]


@app.command()
def download(source: str = typer.Argument("all", help="kermany | rsna | all")) -> None:
    """Download raw datasets from their public sources (resumable)."""
    from .data.download import download_kermany, download_rsna

    for s in _sources(source):
        (download_kermany if s == "kermany" else download_rsna)(DATA_RAW)


@app.command()
def prepare(
    source: str = typer.Argument("all", help="kermany | rsna | all"),
    size: int = typer.Option(256, help="Side of the cached square images."),
    seed: int = 42,
) -> None:
    """Decode, resize and cache images; build patient-level splits."""
    for s in _sources(source):
        if s == "kermany":
            from .data.kermany import prepare_kermany

            prepare_kermany(size=size, seed=seed)
        else:
            from .data.rsna import prepare_rsna

            prepare_rsna(size=size, seed=seed)
        typer.echo((Path("data/processed") / f"{s}_summary.json").read_text())


@app.command()
def train(
    config: Path = typer.Argument(..., exists=True, help="YAML file in configs/"),
    epochs: int | None = None,
    batch_size: int | None = None,
    max_train_samples: int | None = typer.Option(None, help="Cap per source, for smoke tests."),
    name: str | None = None,
    mlflow: bool = typer.Option(False, help="Also log to MLflow (pip install '.[track]')."),
) -> None:
    """Train a model from a config file."""
    from .config import TrainConfig
    from .train import train as run_training

    cfg = TrainConfig.from_yaml(
        config, epochs=epochs, batch_size=batch_size, max_train_samples=max_train_samples, name=name
    )
    run_training(cfg, mlflow=mlflow)


@app.command()
def evaluate(
    run_dirs: list[Path] = typer.Argument(..., help="One or more runs/<name> folders."),
    n_boot: int = 1000,
) -> None:
    """Evaluate runs on the test split of every dataset (with bootstrap CIs)."""
    from .evaluate import evaluate_run

    for run_dir in run_dirs:
        evaluate_run(run_dir, n_boot=n_boot)


@app.command()
def explain(
    run_dir: Path = typer.Argument(..., exists=True),
    figure: Path = typer.Option(REPORTS / "figures" / "gradcam_examples.png"),
) -> None:
    """Grad-CAM: pointing game vs RSNA radiologist boxes + example figure."""
    from .explain import localisation_eval, plot_examples

    localisation_eval(run_dir)
    plot_examples(run_dir, figure)


@app.command("compare-cam")
def compare_cam(
    run_dirs: list[Path] = typer.Argument(..., exists=True, help="Runs already processed by `pneumo explain`."),
    labels: str = typer.Option(..., help="Comma-separated row labels, one per run."),
    figure: Path = typer.Option(REPORTS / "figures" / "gradcam_comparison.png"),
) -> None:
    """Side-by-side Grad-CAM of several models on the same RSNA test images."""
    from .explain import plot_cam_comparison

    plot_cam_comparison(run_dirs, [s.strip() for s in labels.split(",")], figure)


@app.command()
def export(
    run_dir: Path = typer.Argument(..., exists=True),
    out_dir: Path = typer.Option(MODELS),
) -> None:
    """Export a run to ONNX (+ model_meta.json) and check parity with ONNX Runtime."""
    from .export import export_onnx

    path = export_onnx(run_dir, out_dir)
    typer.echo(f"Exported {path} ({path.stat().st_size / 1e6:.1f} MB)")


@app.command("export-browser")
def export_browser(
    run_dir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(MODELS / "model_browser.onnx"),
) -> None:
    """Export ONNX with probability + closed-form Grad-CAM outputs, for ONNX Runtime Web."""
    from .export import export_browser_onnx

    path = export_browser_onnx(run_dir, out)
    typer.echo(f"Exported {path} ({path.stat().st_size / 1e6:.1f} MB)")


@app.command()
def report(featured: str | None = typer.Option(None, help="Run name to feature in the figures.")) -> None:
    """Aggregate all evaluated runs into reports/results.md and figures."""
    from .report import build_report

    build_report(RUNS, REPORTS, featured)


@app.command()
def demo(
    model_dir: Path = typer.Option(MODELS, help="Folder with model.pt and model_meta.json."),
    host: str = "127.0.0.1",
    port: int = 7860,
) -> None:
    """Start the clinical web demo (viewer + Grad-CAM + findings) at http://HOST:PORT."""
    import os

    import uvicorn

    os.environ["MODEL_DIR"] = str(model_dir)
    uvicorn.run("app.web:app", host=host, port=port)


@app.command()
def predict(
    images: list[Path] = typer.Argument(..., exists=True),
    model_dir: Path = typer.Option(MODELS),
) -> None:
    """Predict with the exported ONNX model (no PyTorch needed)."""
    from PIL import Image

    from .inference import OnnxPredictor

    predictor = OnnxPredictor(model_dir)
    for path in images:
        with Image.open(path) as im:
            typer.echo(json.dumps({"file": str(path), **predictor.predict(im).to_dict()}))


if __name__ == "__main__":
    app()

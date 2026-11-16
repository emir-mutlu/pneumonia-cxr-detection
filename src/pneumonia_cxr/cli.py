"""Command line interface: ``pneumo --help``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from .paths import DATA_RAW, MODELS, PUBLIC_SOURCES, REPORTS, RUNS, SOURCES

# Windows consoles default to cp1252; progress bars of some libraries print Unicode blocks.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Pneumonia detection from chest X-rays.")


def _sources(source: str, allowed: tuple[str, ...] = SOURCES) -> list[str]:
    """'all' means the public sources; access-controlled ones must be named explicitly."""
    if source == "all":
        return list(PUBLIC_SOURCES)
    if source not in allowed:
        raise typer.BadParameter(f"source must be one of {allowed} or 'all'")
    return [source]


@app.command()
def download(source: str = typer.Argument("all", help="kermany | rsna | all")) -> None:
    """Download the public datasets (resumable). VinDr-PCXR: see `pneumo physionet-import`."""
    from .data.download import download_kermany, download_rsna

    for s in _sources(source, PUBLIC_SOURCES):
        (download_kermany if s == "kermany" else download_rsna)(DATA_RAW)


@app.command("physionet-import")
def physionet_import(
    zip_file: Path = typer.Argument(..., exists=True, dir_okay=False, help="ZIP from the project's Files section."),
    project: str = typer.Option(..., help="Folder name under data/raw, e.g. vindr-pcxr"),
) -> None:
    """Extract a PhysioNet ZIP into data/raw/<project> and verify it against SHA256SUMS.txt."""
    from .data.physionet import import_zip, verify

    out = import_zip(zip_file, DATA_RAW / project)
    report = verify(out)
    typer.echo(f"{out}\n{report}")
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def prepare(
    source: str = typer.Argument("all", help="kermany | rsna | vindr_pcxr | all (= kermany + rsna)"),
    size: int = typer.Option(256, help="Side of the cached square images."),
    seed: int = 42,
) -> None:
    """Decode, resize and cache images; build patient-level splits."""
    for s in _sources(source):
        if s == "kermany":
            from .data.kermany import prepare_kermany

            prepare_kermany(size=size, seed=seed)
        elif s == "rsna":
            from .data.rsna import prepare_rsna

            prepare_rsna(size=size, seed=seed)
        else:
            from .data.vindr_pcxr import prepare_vindr_pcxr

            prepare_vindr_pcxr(size=size, seed=seed)
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
    source: list[str] | None = typer.Option(None, "--source", help="Only these test sets; repeatable."),
    n_boot: int = 1000,
) -> None:
    """Evaluate runs on the test split of every prepared dataset (with bootstrap CIs)."""
    from .evaluate import evaluate_run

    sources = tuple(dict.fromkeys(x for s in source for x in _sources(s))) if source else SOURCES
    for run_dir in run_dirs:
        evaluate_run(run_dir, sources=sources, n_boot=n_boot)


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
    site_profile: Path | None = typer.Option(None, exists=True, help="JSON from `pneumo calibrate`."),
) -> None:
    """Predict with the exported ONNX model (no PyTorch needed); PNG, JPEG or DICOM."""
    from .inference import OnnxPredictor, decode_image

    predictor = OnnxPredictor(model_dir, site_profile=site_profile)
    for path in images:
        image = decode_image(path.read_bytes(), path.name)
        typer.echo(json.dumps({"file": str(path), **predictor.predict(image).to_dict()}))


@app.command()
def calibrate(
    labels_csv: Path = typer.Argument(..., exists=True, help="CSV with columns `file` and `label` (1 = pneumonia)."),
    site: str = typer.Option(..., help="Short site name, used for the profile file."),
    images_dir: Path = typer.Option(Path("."), exists=True, file_okay=False, help="Folder the `file` paths are in."),
    model_dir: Path = typer.Option(MODELS),
    target_sensitivity: float = typer.Option(0.90, help="Local threshold for this sensitivity; 0 = Youden's J."),
    out: Path | None = typer.Option(None, help="Default: site_profiles/<site>.json"),
) -> None:
    """Validate and recalibrate the released model on a new hospital's own labelled X-rays."""
    import pandas as pd
    from tqdm import tqdm

    from .inference import OnnxPredictor, decode_image
    from .site_calibration import calibrate_site, format_report

    table = pd.read_csv(labels_csv)
    missing = {"file", "label"} - set(table.columns)
    if missing:
        raise typer.BadParameter(f"{labels_csv} lacks columns {sorted(missing)}")
    predictor = OnnxPredictor(model_dir)
    probs = [
        float(predictor.predict_raw([decode_image((images_dir / f).read_bytes(), str(f))])[0])
        for f in tqdm(table["file"], desc="predict")
    ]
    profile = calibrate_site(
        probs, table["label"].to_numpy(), site, predictor.model_threshold, target_sensitivity or None
    )
    path = profile.save(out or Path("site_profiles") / f"{site}.json")
    typer.echo(f"{format_report(profile)}\nSaved {path}. Use it with: pneumo predict IMAGE --site-profile {path}")


if __name__ == "__main__":
    app()

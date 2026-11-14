"""Publish the web demo to Hugging Face.

Static mode (default, free): the model runs in the visitor's browser with ONNX Runtime Web.

* model repo  <user>/pneumonia-cxr-convnext       model_browser.onnx (+ closed-form Grad-CAM), card
* Space       <user>/pneumonia-cxr-detection      static site: viewer UI + browser engine

Docker mode (Hugging Face PRO): the FastAPI + PyTorch demo (app/web.py) on a CPU Space.

    hf auth login                                    # once, with a write token
    python scripts/deploy_space.py                   # static Space + model repo
    python scripts/deploy_space.py --local site      # build the static site with a local model, no upload
    python scripts/deploy_space.py --mode docker     # needs a PRO subscription
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ORT_VERSION = "1.31.1"
ORT_CDN = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/"
SPACE_NAME = "pneumonia-cxr-detection"
MODEL_REPO_NAME = "pneumonia-cxr-convnext"

DOCKER_FILES = {
    "deploy/huggingface/docker/README.md": "README.md",
    "deploy/huggingface/docker/Dockerfile": "Dockerfile",
    "deploy/huggingface/docker/requirements.txt": "requirements.txt",
    "LICENSE": "LICENSE",
    "app/__init__.py": "app/__init__.py",
    "app/web.py": "app/web.py",
    "src/pneumonia_cxr/__init__.py": "src/pneumonia_cxr/__init__.py",
    "src/pneumonia_cxr/models.py": "src/pneumonia_cxr/models.py",
    "src/pneumonia_cxr/gradcam.py": "src/pneumonia_cxr/gradcam.py",
    "src/pneumonia_cxr/inference.py": "src/pneumonia_cxr/inference.py",
    "src/pneumonia_cxr/data/__init__.py": "src/pneumonia_cxr/data/__init__.py",
    "src/pneumonia_cxr/data/imaging.py": "src/pneumonia_cxr/data/imaging.py",
    "models/model.pt": "models/model.pt",
    "models/model_meta.json": "models/model_meta.json",
}


def _copy(files: dict[str, str], out: Path) -> None:
    for src, dst in files.items():
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, out / dst)


def build_docker(out: Path) -> Path:
    _copy(DOCKER_FILES, out)
    for d in ("app/static", "app/examples"):
        shutil.copytree(ROOT / d, out / d, dirs_exist_ok=True)
    return out


def build_static(out: Path, model_url: str) -> Path:
    """Static site: index.html + viewer assets + examples + model_meta.json (model loaded from model_url)."""
    config = {"engine": "browser", "modelUrl": model_url, "metaUrl": "model_meta.json", "wasmPaths": ORT_CDN}
    engine = (
        f'<script src="{ORT_CDN}ort.min.js"></script>\n'
        f"  <script>window.CXR_CONFIG = {json.dumps(config)};</script>\n"
        '  <script src="static/engine-browser.js" defer></script>'
    )
    html = (ROOT / "app/static/index.html").read_text(encoding="utf-8")
    if "<!-- ENGINE -->" not in html:
        raise RuntimeError("index.html has no <!-- ENGINE --> placeholder")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html.replace("<!-- ENGINE -->", engine), encoding="utf-8")
    for name in ("app.js", "styles.css", "engine-browser.js"):
        (out / "static").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "app/static" / name, out / "static" / name)
    shutil.copytree(ROOT / "app/examples", out / "examples", dirs_exist_ok=True)
    _copy({"models/model_meta.json": "model_meta.json", "LICENSE": "LICENSE"}, out)
    shutil.copy2(ROOT / "deploy/huggingface/static/README.md", out / "README.md")
    return out


def export_browser_model(out_dir: Path) -> Path:
    from pneumonia_cxr.export import export_browser_onnx

    return export_browser_onnx(ROOT / "models", out_dir / "model_browser.onnx")


def _list(folder: Path) -> None:
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB in {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["static", "docker"], default="static")
    parser.add_argument("--local", type=Path, help="static mode: build the site here with a local model, no upload")
    parser.add_argument("--dry-run", action="store_true", help="build in a temp folder and list it, no upload")
    args = parser.parse_args()

    if args.local:
        site = build_static(args.local, "model_browser.onnx")
        export_browser_model(site)
        _list(site)
        print(f"Serve it with: python -m http.server 8008 --directory {site}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if args.dry_run:
            _list(build_static(tmp / "space", "<model url>") if args.mode == "static" else build_docker(tmp / "space"))
            return

        from huggingface_hub import HfApi

        api = HfApi()
        user = api.whoami()["name"]
        space = f"{user}/{SPACE_NAME}"
        message = "Deploy from github.com/emir-mutlu/pneumonia-cxr-detection"

        if args.mode == "docker":
            folder = build_docker(tmp / "space")
            api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
            api.upload_folder(
                repo_id=space, repo_type="space", folder_path=folder, commit_message=message, delete_patterns=["*"]
            )
            print(f"https://huggingface.co/spaces/{space}")
            return

        model_repo = f"{user}/{MODEL_REPO_NAME}"
        model_dir = tmp / "model"
        model_dir.mkdir()
        export_browser_model(model_dir)
        shutil.copy2(ROOT / "models/model_meta.json", model_dir / "model_meta.json")
        shutil.copy2(ROOT / "deploy/huggingface/model/README.md", model_dir / "README.md")
        api.create_repo(model_repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=model_repo, folder_path=model_dir, commit_message=message)
        print(f"https://huggingface.co/{model_repo}")

        model_url = f"https://huggingface.co/{model_repo}/resolve/main/model_browser.onnx"
        folder = build_static(tmp / "space", model_url)
        _list(folder)
        api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(
            repo_id=space, repo_type="space", folder_path=folder, commit_message=message, delete_patterns=["*"]
        )
        print(f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()

"""Publish the web demo to a Hugging Face Space (Docker SDK).

Assembles only what the demo needs (no training code, no ONNX file) and uploads it:

    hf auth login                                   # once, with a *write* token
    python scripts/deploy_space.py                  # -> huggingface.co/spaces/<user>/pneumonia-cxr-detection
    python scripts/deploy_space.py --dry-run        # just build the folder and list it
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "deploy/huggingface/README.md": "README.md",
    "deploy/huggingface/Dockerfile": "Dockerfile",
    "deploy/huggingface/requirements.txt": "requirements.txt",
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
DIRS = {"app/static": "app/static", "app/examples": "app/examples"}


def build(out: Path) -> Path:
    for src, dst in FILES.items():
        target = out / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, target)
    for src, dst in DIRS.items():
        shutil.copytree(ROOT / src, out / dst, dirs_exist_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", default=None, help="owner/name (default: <your user>/pneumonia-cxr-detection)")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        folder = build(Path(tmp) / "space")
        files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
        print(f"{len(files)} files, {sum((folder / f).stat().st_size for f in files) / 1e6:.1f} MB")
        if args.dry_run:
            print("\n".join(files))
            return

        from huggingface_hub import HfApi

        api = HfApi()
        space = args.space or f"{api.whoami()['name']}/pneumonia-cxr-detection"
        api.create_repo(space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=space,
            repo_type="space",
            folder_path=folder,
            commit_message="Deploy web demo from github.com/emir-mutlu/pneumonia-cxr-detection",
            delete_patterns=["*"],  # the Space mirrors this build exactly
        )
        print(f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()

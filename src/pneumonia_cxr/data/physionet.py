"""Import PhysioNet projects downloaded as a ZIP from the project page.

PhysioNet data is not public: sign the project's data use agreement (and complete
credentialing for credentialed projects), then use "Download the ZIP file" in the Files
section of the project page while signed in. ``import_zip`` extracts the archive safely into
``data/raw/<project>`` and ``verify`` checks every file against the project's
``SHA256SUMS.txt``. Both can be re-run: files already extracted with the right size are
skipped.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from tqdm import tqdm

SUMS_FILE = "SHA256SUMS.txt"


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse ``sha256sum`` output (``<hash>  <path>`` or ``<hash> *<path>``) into {path: hash}."""
    sums = {}
    for line in text.splitlines():
        digest, _, path = line.strip().partition(" ")
        path = path.strip().lstrip("*")
        if len(digest) == 64 and path and path != SUMS_FILE:
            sums[path] = digest.lower()
    return sums


def _safe_relpath(path: str) -> PurePosixPath:
    rel = PurePosixPath(path.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or ":" in rel.parts[0]:
        raise ValueError(f"Refusing unsafe path in archive: {path!r}")
    return rel


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def import_zip(zip_path: Path, out: Path) -> Path:
    """Extract a PhysioNet ZIP into ``out``, dropping the archive's single top-level folder."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        parts = [PurePosixPath(m.filename).parts for m in members]
        strip = int(len({p[0] for p in parts}) == 1 and all(len(p) > 1 for p in parts))
        targets = [(m, out / _safe_relpath("/".join(p[strip:]))) for m, p in zip(members, parts)]  # check all first
        with tqdm(total=sum(m.file_size for m in members), unit="B", unit_scale=True, desc="extract") as bar:
            for member, dest in targets:
                if not (dest.exists() and dest.stat().st_size == member.file_size):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                bar.update(member.file_size)
    return out


@dataclass
class VerifyReport:
    checked: int = 0
    missing: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.checked > 0 and not self.missing and not self.mismatched

    def __str__(self) -> str:
        status = "OK" if self.ok else "FAILED"
        text = f"{status}: {self.checked} files checked, {len(self.missing)} missing, {len(self.mismatched)} corrupt"
        for name, items in (("missing", self.missing), ("corrupt", self.mismatched)):
            if items:
                text += f"\n  first {name}: " + ", ".join(sorted(items)[:5])
        return text


def verify(folder: Path, workers: int = 8) -> VerifyReport:
    """Check every file listed in ``folder/SHA256SUMS.txt``."""
    sums_path = folder / SUMS_FILE
    if not sums_path.exists():
        raise FileNotFoundError(f"{sums_path} not found; is this the extracted PhysioNet project folder?")
    expected = parse_sha256sums(sums_path.read_text(encoding="utf-8", errors="replace"))

    def check(item: tuple[str, str]) -> tuple[str, str]:
        rel, digest = item
        path = folder / _safe_relpath(rel)
        if not path.exists():
            return rel, "missing"
        return rel, "ok" if _sha256(path) == digest else "mismatch"

    report = VerifyReport(checked=len(expected))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rel, status in tqdm(pool.map(check, expected.items()), total=len(expected), unit="file", desc="verify"):
            if status == "missing":
                report.missing.append(rel)
            elif status == "mismatch":
                report.mismatched.append(rel)
    return report

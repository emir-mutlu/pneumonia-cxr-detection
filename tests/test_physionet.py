"""PhysioNet ZIP import: safe extraction, checksum verification, re-runs."""

import hashlib
import zipfile

import pytest

from pneumonia_cxr.data.physionet import import_zip, parse_sha256sums, verify

FILES = {
    "image_labels_test.csv": b"image_id,Pneumonia\n",
    "test/b.dicom": b"B" * 3000,
    "train/a.dicom": b"A" * 5000,
}
TOP = "vindr-pcxr-an-open-dataset-1.0.0"


def _sums(files: dict[str, bytes]) -> str:
    return "".join(f"{hashlib.sha256(d).hexdigest()}  {p}\n" for p, d in files.items())


def _zip(path, files: dict[str, bytes], top: str | None = TOP):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in {**files, "SHA256SUMS.txt": _sums(files).encode()}.items():
            zf.writestr(f"{top}/{name}" if top else name, data)
    return path


def test_parse_sha256sums_formats():
    h = "a" * 64
    text = f"{h}  train/x.dicom\n\n{h} *test/y.dicom\n{h}  SHA256SUMS.txt\nnot a line\n"
    assert parse_sha256sums(text) == {"train/x.dicom": h, "test/y.dicom": h}


@pytest.mark.parametrize("top", [TOP, None])
def test_import_strips_top_folder_and_verifies(tmp_path, top):
    out = import_zip(_zip(tmp_path / "p.zip", FILES, top), tmp_path / "raw" / "vindr-pcxr")
    for rel, data in FILES.items():
        assert (out / rel).read_bytes() == data
    report = verify(out)
    assert report.ok and report.checked == 3


def test_reimport_repairs_truncated_file(tmp_path):
    zip_path = _zip(tmp_path / "p.zip", FILES)
    out = import_zip(zip_path, tmp_path / "raw")
    (out / "train" / "a.dicom").write_bytes(b"A" * 10)  # interrupted extraction
    assert not verify(out).ok
    import_zip(zip_path, out)
    assert verify(out).ok


def test_verify_reports_missing_and_corrupt(tmp_path):
    out = import_zip(_zip(tmp_path / "p.zip", FILES), tmp_path / "raw")
    (out / "test" / "b.dicom").unlink()
    (out / "train" / "a.dicom").write_bytes(b"X" * 5000)
    report = verify(out)
    assert not report.ok
    assert report.missing == ["test/b.dicom"] and report.mismatched == ["train/a.dicom"]
    assert "FAILED" in str(report)


def test_zip_slip_is_refused(tmp_path):
    with zipfile.ZipFile(tmp_path / "evil.zip", "w") as zf:
        zf.writestr("../evil.txt", b"x")
        zf.writestr("ok.txt", b"y")
    with pytest.raises(ValueError, match="unsafe"):
        import_zip(tmp_path / "evil.zip", tmp_path / "raw")
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "raw").exists()


def test_verify_needs_checksum_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify(tmp_path)

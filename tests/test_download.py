"""Dataset download behaviour: extra files, Drive-backed raw storage, interrupted downloads, free space, `--all`."""

from __future__ import annotations

import io
import sys
import types
import zipfile
from pathlib import Path

import pytest

from mmibc.cli import main
from mmibc.data import download as dl
from mmibc.data.registry import REGISTRY, DatasetSpec
from mmibc.utils.paths import ProjectPaths, resolve_paths


@pytest.fixture
def paths(tmp_path):
    return ProjectPaths(tmp_path / "drive", tmp_path / "local", raw_on_drive=True).ensure()


def _png(p: Path, n: int = 8) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG" + b"0" * n)


# ----------------------------------------------------------------------------- where raw data goes
def test_raw_downloads_default_to_drive(tmp_path):
    p = resolve_paths(tmp_path / "root", tmp_path / "loc")
    assert p.raw == (tmp_path / "root").resolve() / "raw"
    assert resolve_paths(tmp_path / "root", tmp_path / "loc", raw_on_drive=False).raw == (tmp_path / "loc").resolve() / "raw"


def test_config_default_keeps_raw_on_drive():
    from mmibc.utils.config import load_config

    assert load_config().paths.raw_on_drive is True


# ----------------------------------------------------------------------------- BrEaST-Lesions-USG: a zip AND a spreadsheet
def test_breast_lesions_usg_fetches_the_archive_and_the_clinical_spreadsheet(paths, monkeypatch):
    spec = REGISTRY["breast_lesions_usg"]
    assert spec.kind == "url" and spec.ref.endswith("images_and_masks-Dec-15-2023.zip")
    (name, xurl), = spec.extra_files
    assert name.endswith("clinical-data-Dec-15-2023.xlsx") and xurl.endswith(name)
    calls = []

    def fake_zip(url, dest):
        calls.append(("zip", url))
        _png(dest / "images" / "case001.png")

    def fake_file(url, target):
        calls.append(("file", url))
        target.write_bytes(b"PK")

    monkeypatch.setattr(dl, "_download_url", fake_zip)
    monkeypatch.setattr(dl, "_download_file", fake_file)
    dest = dl.download_dataset("breast_lesions_usg", paths)
    assert [c[0] for c in calls] == ["zip", "file"] and (dest / name).is_file()
    assert not (dest / dl.INCOMPLETE_MARKER).exists()
    calls.clear()
    dl.download_dataset("breast_lesions_usg", paths)                                # complete: nothing is fetched again
    assert calls == []
    (dest / name).unlink()                                                          # only the missing spreadsheet is re-fetched
    dl.download_dataset("breast_lesions_usg", paths)
    assert [c[0] for c in calls] == ["file"]


def test_a_missing_spreadsheet_is_reported_by_name(paths, monkeypatch):
    monkeypatch.setattr(dl, "_download_url", lambda url, dest: _png(dest / "a.png"))

    def boom(url, target):
        raise RuntimeError("returned a web page")

    monkeypatch.setattr(dl, "_download_file", boom)
    with pytest.raises(RuntimeError) as e:
        dl.download_dataset("breast_lesions_usg", paths)
    assert "clinical-data-Dec-15-2023.xlsx" in str(e.value) and "web page" in str(e.value)


# ----------------------------------------------------------------------------- interrupted downloads on Drive
def test_an_interrupted_download_is_never_mistaken_for_a_finished_one(paths, monkeypatch):
    dest = paths.raw / "bus_bra"
    _png(dest / "half.png")                                                         # looks valid to a glob...
    (dest / dl.INCOMPLETE_MARKER).write_text("interrupted")                         # ...but the marker says it was not finished
    fetched = []
    monkeypatch.setattr(dl, "_download_url", lambda url, d: (fetched.append(url), _png(d / "full.png")))
    dl.download_dataset("bus_bra", paths)
    assert fetched and not (dest / dl.INCOMPLETE_MARKER).exists()
    fetched.clear()
    dl.download_dataset("bus_bra", paths)                                           # and a finished one is skipped
    assert fetched == []


def test_a_failed_download_leaves_the_marker(paths, monkeypatch):
    def fail(url, d):
        _png(d / "partial.png")
        raise OSError("connection reset")

    monkeypatch.setattr(dl, "_download_url", fail)
    with pytest.raises(RuntimeError):
        dl.download_dataset("bus_bra", paths)
    # the partial file satisfies the glob, so the marker is the only thing standing between it and being trusted
    assert (paths.raw / "bus_bra" / dl.INCOMPLETE_MARKER).exists()


# ----------------------------------------------------------------------------- Kaggle -> Drive
def test_kaggle_download_is_copied_to_drive_and_the_local_cache_is_removed(paths, tmp_path, monkeypatch):
    cache = tmp_path / "kagglehub_cache" / "busi"
    _png(cache / "Dataset" / "benign" / "b1.png")
    _png(cache / "Dataset" / "malignant" / "m1.png")
    monkeypatch.setitem(sys.modules, "kagglehub", types.SimpleNamespace(dataset_download=lambda ref: str(cache)))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "t")
    dest = dl.download_dataset("busi", paths)
    assert dest == paths.root / "raw" / "busi" and (dest / "Dataset" / "benign" / "b1.png").is_file()
    assert not cache.exists()                                                       # Drive copy made, local cache dropped
    assert not (dest / dl.INCOMPLETE_MARKER).exists()


def test_not_enough_drive_space_is_refused_before_anything_is_copied(paths, tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "busi"
    _png(cache / "benign" / "b.png", n=5000)
    _png(cache / "malignant" / "m.png")
    monkeypatch.setitem(sys.modules, "kagglehub", types.SimpleNamespace(dataset_download=lambda ref: str(cache)))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "t")
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: (10_000, 9_990, 10))      # a nearly full Drive
    with pytest.raises(RuntimeError, match=r"Not enough free space.*raw_on_drive=false"):
        dl.download_dataset("busi", paths)
    assert not list((paths.raw / "busi").rglob("*.png"))                            # nothing was copied
    assert cache.exists()                                                           # and the downloaded cache is still there


# ----------------------------------------------------------------------------- the zip is staged locally, not on Drive
def test_zip_is_extracted_into_dest_and_never_stored_there(tmp_path, monkeypatch):
    pytest.importorskip("requests")
    pytest.importorskip("tqdm")
    import requests

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("set/benign/a.png", b"x" * 20)
        z.writestr("set/malignant/b.png", b"y" * 20)
    data = buf.getvalue()

    class Resp:
        headers = {"content-length": str(len(data))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, n):
            yield data

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    made = []
    real_mkdtemp = dl.tempfile.mkdtemp
    monkeypatch.setattr(dl.tempfile, "mkdtemp", lambda **k: made.append(real_mkdtemp(**k)) or made[-1])
    dest = tmp_path / "drive" / "raw" / "x"
    dl._download_url("https://example.org/x.zip", dest)
    assert (dest / "set" / "benign" / "a.png").is_file() and not list(dest.rglob("*.zip"))
    assert made and not Path(made[0]).exists()                                      # temporary staging area cleaned up


# ----------------------------------------------------------------------------- inspect shows what is inside a spreadsheet
def test_inspect_prints_spreadsheet_columns_and_label_counts(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    root = tmp_path / "raw" / "d"
    root.mkdir(parents=True)
    pd.DataFrame({"CaseID": [f"c{i}" for i in range(30)], "Classification": ["benign"] * 20 + ["malignant"] * 10}).to_excel(root / "clin.xlsx", index=False)
    out = dl.inspect_dataset(root)
    assert "clin.xlsx" in out and "30 rows x 2 columns" in out
    assert "columns: CaseID | Classification" in out
    assert "values of 'Classification': benign=20, malignant=10" in out
    assert "values of 'CaseID'" not in out                                          # an ID column is not a label: too many values


def test_inspect_survives_a_file_that_is_not_a_spreadsheet(tmp_path):
    root = tmp_path / "raw" / "d"
    root.mkdir(parents=True)
    (root / "fake.xlsx").write_text("<html>moved</html>")
    assert "could not read as a spreadsheet" in dl.inspect_dataset(root)


# ----------------------------------------------------------------------------- download everything at once
def test_download_all_continues_past_failures_and_summarises(tmp_path, monkeypatch, capsys):
    seen = []

    def fake(key, paths, force=False):
        seen.append(key)
        if key in ("busi", "cmmd"):
            raise RuntimeError("no credentials")
        return paths.raw / key

    monkeypatch.setattr(dl, "download_dataset", fake)
    rc = main(["download", "--all", "--root", str(tmp_path / "r"), "--local", str(tmp_path / "l")])
    cap = capsys.readouterr()
    assert rc == 1 and seen == list(REGISTRY)                                       # every dataset was attempted
    assert "summary:" in cap.out and "FAILED  busi" in cap.out and "ok      vindr_images" in cap.out
    assert "2 of 8 failed" in cap.err and "mmibc download busi cmmd" in cap.err
    assert "Google Drive" in cap.out                                                # says where the data is going


def test_download_all_returns_zero_when_everything_succeeds(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dl, "download_dataset", lambda key, paths, force=False: paths.raw / key)
    assert main(["download", "--all", "--root", str(tmp_path / "r"), "--local", str(tmp_path / "l")]) == 0
    assert "failed" not in capsys.readouterr().err


def test_unknown_dataset_name_is_rejected_with_the_known_names(tmp_path, capsys):
    assert main(["download", "bussi", "--root", str(tmp_path / "r"), "--local", str(tmp_path / "l")]) == 2
    assert "busi" in capsys.readouterr().err


def test_raw_on_drive_can_be_switched_off_for_download_and_inspect(tmp_path, monkeypatch, capsys):
    where = []
    monkeypatch.setattr(dl, "download_dataset", lambda key, paths, force=False: where.append(paths.raw) or paths.raw / key)
    main(["download", "busi", "--root", str(tmp_path / "r"), "--local", str(tmp_path / "l"), "--set", "paths.raw_on_drive=false"])
    main(["download", "busi", "--root", str(tmp_path / "r"), "--local", str(tmp_path / "l")])
    assert where[0] == (tmp_path / "l").resolve() / "raw" and where[1] == (tmp_path / "r").resolve() / "raw"


def test_a_manual_dataset_still_fails_with_instructions(tmp_path, monkeypatch, capsys):
    """No registered dataset is manual any more, but the mechanism stays for sources without a download link."""
    monkeypatch.setitem(REGISTRY, "zz_manual", DatasetSpec("zz_manual", "manual", "https://example.org", ("**/*.png",), note="get it by hand"))
    assert main(["download", "zz_manual", "--root", str(tmp_path / "r"), "--local", str(tmp_path / "l")]) == 1
    assert "manually" in capsys.readouterr().err


# ----------------------------------------------------------------------------- never silently save to the temporary disk
def test_colab_without_drive_fails_loudly_instead_of_using_the_temporary_disk(monkeypatch, tmp_path):
    from mmibc.utils import paths as P

    monkeypatch.delenv("MMIBC_ROOT", raising=False)
    monkeypatch.setattr(P, "in_colab", lambda: True)
    monkeypatch.setattr(P, "mount_drive", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match=r"(?s)Google Drive is not mounted.*drive\.mount"):
        P.resolve_paths(None, tmp_path / "loc")
    # an explicit location is the user's decision and is respected
    assert P.resolve_paths(tmp_path / "r", tmp_path / "loc").root == (tmp_path / "r").resolve()


def test_mount_failure_in_a_subprocess_is_reported_not_raised_raw(monkeypatch, tmp_path):
    from mmibc.utils import paths as P

    def broken_mount(point):
        raise AttributeError("'NoneType' object has no attribute 'kernel'")

    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=broken_mount)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(P, "in_colab", lambda: True)
    assert P.mount_drive(str(tmp_path / "nodrive")) is False


def test_a_credential_failure_is_not_buried_under_the_dataset_note(paths, monkeypatch):
    for n in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("HOME", str(paths.root))
    monkeypatch.setattr(dl, "in_colab", lambda: False)
    monkeypatch.setitem(sys.modules, "kagglehub", types.SimpleNamespace(dataset_download=lambda ref: ""))
    with pytest.raises(RuntimeError) as e:
        dl.download_dataset("cmmd", paths)
    msg = str(e.value)
    assert "No Kaggle credentials found" in msg and "UNKNOWN" not in msg and len(msg) < 1400

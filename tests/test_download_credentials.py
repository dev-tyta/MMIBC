"""Kaggle credential handling, including the Colab case that bit a real run.

Colab secrets (``google.colab.userdata``) only work inside the notebook's kernel process. A ``!mmibc download`` shell
command is a separate process: ``userdata.get`` raises ``AttributeError: 'NoneType' object has no attribute 'kernel'``
there, which used to be swallowed and reported as a generic "No Kaggle credentials found".
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from mmibc.data import download as dl

ALL = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "HF_TOKEN")


@pytest.fixture
def clean(monkeypatch, tmp_path):
    for n in ALL:
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(dl, "in_colab", lambda: False)
    return tmp_path


def _fake_userdata(monkeypatch, secrets: dict[str, str] | None = None, error: Exception | None = None):
    """Install a fake ``google.colab.userdata``. ``error`` is raised by every get(); otherwise a missing name raises KeyError."""

    def get(name):
        if error is not None:
            raise error
        return secrets[name]

    ud = types.SimpleNamespace(get=get)
    colab = types.ModuleType("google.colab")
    colab.userdata = ud
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(dl, "in_colab", lambda: True)


# ----------------------------------------------------------------------------- what counts as "present"
def test_credentials_present_for_each_supported_form(clean, monkeypatch):
    assert not dl.kaggle_credentials_present()
    monkeypatch.setenv("KAGGLE_USERNAME", "u")                    # half of the legacy pair is not enough
    assert not dl.kaggle_credentials_present()
    monkeypatch.setenv("KAGGLE_KEY", "k")
    assert dl.kaggle_credentials_present()
    monkeypatch.delenv("KAGGLE_USERNAME"), monkeypatch.delenv("KAGGLE_KEY")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "t")
    assert dl.kaggle_credentials_present()
    monkeypatch.delenv("KAGGLE_API_TOKEN")
    (clean / ".kaggle").mkdir()
    (clean / ".kaggle" / "access_token").write_text("t")
    assert dl.kaggle_credentials_present()
    (clean / ".kaggle" / "access_token").unlink()
    (clean / ".kaggle" / "kaggle.json").write_text("{}")
    assert dl.kaggle_credentials_present()


# ----------------------------------------------------------------------------- the failure from the real run
def test_shell_subprocess_in_colab_gets_a_message_that_says_what_to_do(clean, monkeypatch):
    _fake_userdata(monkeypatch, error=AttributeError("'NoneType' object has no attribute 'kernel'"))
    with pytest.raises(RuntimeError) as e:
        dl.setup_kaggle_credentials()
    msg = str(e.value)
    assert "export_colab_secrets" in msg and "notebook cells" in msg      # tells the user the fix
    assert "KAGGLE_API_TOKEN" in msg and "KAGGLE_USERNAME" in msg         # names both credential styles
    assert "kernel" not in msg                                            # not the opaque internal error


def test_in_colab_with_secrets_not_defined_gives_the_generic_message(clean, monkeypatch):
    _fake_userdata(monkeypatch, secrets={})                               # reachable kernel, no secrets stored
    with pytest.raises(RuntimeError, match="No Kaggle credentials found") as e:
        dl.setup_kaggle_credentials()
    assert "Notebook access" in str(e.value)


# ----------------------------------------------------------------------------- the working paths
def test_export_in_a_notebook_cell_sets_env_and_never_prints_values(clean, monkeypatch, capsys):
    _fake_userdata(monkeypatch, secrets={"KAGGLE_API_TOKEN": "SECRET-TOKEN-123", "HF_TOKEN": "SECRET-HF-456"})
    status = dl.export_colab_secrets()
    out = capsys.readouterr().out
    assert status["KAGGLE_API_TOKEN"] == "exported" and status["HF_TOKEN"] == "exported"
    assert status["KAGGLE_USERNAME"].startswith("missing")                # absent secrets are reported, not fatal
    assert "SECRET-TOKEN-123" not in out and "SECRET-HF-456" not in out
    assert "KAGGLE_API_TOKEN" in out
    import os

    assert os.environ["KAGGLE_API_TOKEN"] == "SECRET-TOKEN-123"
    dl.setup_kaggle_credentials()                                         # now succeeds without touching userdata again


def test_setup_reads_secrets_itself_when_called_inside_the_kernel(clean, monkeypatch):
    _fake_userdata(monkeypatch, secrets={"KAGGLE_API_TOKEN": "t"})
    dl.setup_kaggle_credentials()
    assert dl.kaggle_credentials_present()


def test_legacy_pair_is_written_as_kaggle_json_in_colab_only(clean, monkeypatch):
    target = clean / ".kaggle" / "kaggle.json"
    monkeypatch.setenv("KAGGLE_USERNAME", "alice")
    monkeypatch.setenv("KAGGLE_KEY", "legacy-key")
    dl.setup_kaggle_credentials()                                         # not in Colab: do not touch the home directory
    assert not target.exists()
    monkeypatch.setattr(dl, "in_colab", lambda: True)
    dl.setup_kaggle_credentials()
    assert json.loads(target.read_text()) == {"username": "alice", "key": "legacy-key"}
    assert oct(target.stat().st_mode & 0o777) == "0o600"


def test_a_new_style_token_takes_precedence_over_the_legacy_pair(clean, monkeypatch):
    monkeypatch.setattr(dl, "in_colab", lambda: True)
    for k, v in (("KAGGLE_API_TOKEN", "t"), ("KAGGLE_USERNAME", "u"), ("KAGGLE_KEY", "k")):
        monkeypatch.setenv(k, v)
    dl.setup_kaggle_credentials()
    assert not (clean / ".kaggle" / "kaggle.json").exists()


# ----------------------------------------------------------------------------- a URL that serves a web page instead of a zip
def test_url_returning_html_is_reported_as_such(tmp_path, monkeypatch):
    pytest.importorskip("requests")
    pytest.importorskip("tqdm")
    import requests

    class Resp:
        headers = {"content-length": "40"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, n):
            yield b"<!DOCTYPE html><html><body>moved</body></html>"

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    with pytest.raises(RuntimeError, match=r"web page \(HTML\).*instead of a zip"):
        dl._download_url("https://example.org/Dataset.zip", tmp_path / "d")
    assert not (tmp_path / "d" / "download.zip").exists()                 # the bad file is not left behind

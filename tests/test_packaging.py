"""Guards against shipping an incomplete package.

A broad ``.gitignore`` rule (``data/``) once matched the source package ``src/mmibc/data/``: git skipped it silently,
the push looked fine, and a fresh clone failed with ``No module named 'mmibc.data'``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import mmibc

PKG = Path(mmibc.__file__).resolve().parent
REPO = PKG.parents[1]


def _py_files() -> list[Path]:
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def test_every_package_directory_has_an_init():
    missing = sorted({str(p.parent.relative_to(PKG)) for p in _py_files() if not (p.parent / "__init__.py").exists()})
    assert not missing, f"directories with .py files but no __init__.py: {missing}"


def test_missing_configs_dir_gives_an_actionable_error(monkeypatch, tmp_path):
    """A non-editable install has no configs/ next to the package: say how to fix it instead of a bare FileNotFoundError."""
    from mmibc.utils import config

    monkeypatch.delenv("MMIBC_CONFIG_DIR", raising=False)
    monkeypatch.setattr(config, "__file__", str(tmp_path / "a" / "b" / "c" / "d" / "config.py"))
    with pytest.raises(FileNotFoundError, match=r"pip install -e .*MMIBC_CONFIG_DIR"):
        config.config_dir()
    monkeypatch.setenv("MMIBC_CONFIG_DIR", str(tmp_path))  # the documented escape hatch still works
    assert config.config_dir() == tmp_path


@pytest.mark.skipif(shutil.which("git") is None or not (REPO / ".git").exists(), reason="needs a git checkout")
def test_no_source_file_is_git_ignored():
    files = [str(p.relative_to(REPO)) for p in _py_files()]
    done = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=REPO, input="\n".join(files), text=True, capture_output=True)
    ignored = [line for line in done.stdout.splitlines() if line.strip()]
    assert not ignored, f"git would not commit these package files (see .gitignore): {ignored}"


@pytest.mark.skipif(shutil.which("git") is None or not (REPO / ".git").exists(), reason="needs a git checkout")
def test_all_package_files_are_tracked_or_stageable():
    """Every .py file under src/ is either already tracked or untracked-but-not-ignored (i.e. `git add` would take it)."""
    ls = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "src"], cwd=REPO, text=True, capture_output=True)
    known = set(ls.stdout.splitlines())
    missing = [str(p.relative_to(REPO)) for p in _py_files() if str(p.relative_to(REPO)) not in known]
    assert not missing, f"package files git does not see: {missing}"

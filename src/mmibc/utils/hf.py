"""Hugging Face access for gated backbones (DINOv3).

The DINOv3 repositories are gated: you must accept the licence on the model page AND give the process a token. Resolution order:
``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` in the environment, a token saved by ``huggingface-cli login``, and (inside a Colab
notebook kernel) the Colab secret ``HF_TOKEN``. A shell command such as ``!mmibc suite backbones`` is a separate process that cannot
read Colab secrets; run ``from mmibc.data.download import export_colab_secrets; export_colab_secrets()`` in a cell first (the
quickstart notebook does).

A cached download can be left damaged (a runtime that disappeared mid-write, Drive's file system returning a partial file). Loading it then fails
with ``Error while deserializing header: header too large`` and the same failure repeats on every later run, because the cache is never rewritten.
``looks_corrupt`` recognises those errors and ``purge_cached_repo`` removes that ONE repository from the cache so it is downloaded again; the
backbones do this by themselves, once.

``unavailable_reason(name)`` is the cheap pre-flight check suites use: it says WHY a backbone cannot be loaded right now (gated and no
token, and not already in the local cache), so the run can be skipped instead of failing after the data pipeline has been built.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)
TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def hf_token() -> str | None:
    """The token if any source has one; the value is never logged."""
    for k in TOKEN_ENV:
        if os.environ.get(k):
            return os.environ[k]
    try:
        from huggingface_hub import get_token

        tok = get_token()
        if tok:
            return tok
    except Exception:  # noqa: BLE001 - old huggingface_hub or unreadable token file
        pass
    try:
        from mmibc.utils.paths import in_colab

        if in_colab():
            from google.colab import userdata  # type: ignore

            return userdata.get("HF_TOKEN") or None
    except Exception:  # noqa: BLE001 - not in a kernel, secret missing, or 'Notebook access' off
        pass
    return None


def ensure_hf_token() -> str | None:
    """Make the token visible to ``from_pretrained`` (exports it as HF_TOKEN) and return it, or None."""
    tok = hf_token()
    if tok and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = tok
    return tok


def is_cached(repo_id: str, filename: str = "config.json") -> bool:
    """True if ``repo_id`` already sits in the local Hugging Face cache (a gated model that was downloaded once loads without a token)."""
    try:
        from huggingface_hub import try_to_load_from_cache

        home = os.environ.get("HF_HOME")
        kw = {"cache_dir": str(Path(home) / "hub")} if home else {}
        return isinstance(try_to_load_from_cache(repo_id, filename, **kw), str)
    except Exception:  # noqa: BLE001
        return False


def unavailable_reason(backbone: str) -> str | None:
    """None if ``backbone`` can be loaded now; otherwise a one-line reason (and how to fix it)."""
    from mmibc.models.backbones import BACKBONES

    spec = BACKBONES.get(backbone)
    if spec is None or not spec.gated or backbone.endswith("_tiny_test"):
        return None
    if ensure_hf_token() or is_cached(spec.hf_id):
        return None
    return (f"'{backbone}' ({spec.hf_id}) is gated and no Hugging Face token was found. Accept the licence on its model page, add HF_TOKEN "
            "(Colab: Secrets, 'Notebook access' on, then export_colab_secrets()), and rerun: finished runs are not repeated.")


# ------------------------------------------------------------------------------------------------ damaged cache entries
CORRUPT_MARKERS = (
    "deserializing header", "header too large", "incomplete metadata", "pytorchstreamreader", "central directory", "invalid load key",
    "unexpected eof", "unexpected end of", "not a zip", "truncated", "corrupt", "jsondecodeerror", "expecting value",
)


def looks_corrupt(exc: BaseException) -> bool:
    """True if ``exc`` (or what caused it) reads like a damaged or half-written cached file rather than a network or permission problem."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(m in text for m in CORRUPT_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def hub_cache_dirs() -> list[Path]:
    """Every place a Hugging Face download may have been cached: ``$HF_HOME/hub`` (Drive, via mmibc), the library default and ``$HF_HUB_CACHE``."""
    out: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        out.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        out.append(Path(os.environ["HF_HOME"]) / "hub")
    try:
        from huggingface_hub import constants

        out.append(Path(constants.HF_HUB_CACHE))
    except Exception:  # noqa: BLE001
        pass
    out.append(Path.home() / ".cache" / "huggingface" / "hub")
    return list(dict.fromkeys(out))


def hf_cache_kwargs() -> dict:
    """``cache_dir`` for ``from_pretrained``: ``$HF_HOME/hub`` when mmibc pointed HF_HOME at Drive, whatever order the libraries were imported in."""
    home = os.environ.get("HF_HOME")
    return {"cache_dir": str(Path(home) / "hub")} if home else {}


def purge_cached_repo(repo_id: str) -> list[Path]:
    """Delete the cached copy of ONE model repository (``models--org--name``) from every cache location. Returns what was removed.

    Only that folder is touched, and only if it is named exactly like the repository, so nothing else in the cache or on Drive can be hit.
    """
    folder = "models--" + repo_id.replace("/", "--")
    removed = []
    for root in hub_cache_dirs():
        target = root / folder
        if target.name == folder and target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            if not target.exists():
                removed.append(target)
            else:
                log.error("could not fully remove the damaged cache entry %s (delete it by hand)", target)
    return removed

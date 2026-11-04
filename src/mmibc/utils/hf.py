"""Hugging Face access for gated backbones (DINOv3).

The DINOv3 repositories are gated: you must accept the licence on the model page AND give the process a token. Resolution order:
``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` in the environment, a token saved by ``huggingface-cli login``, and (inside a Colab
notebook kernel) the Colab secret ``HF_TOKEN``. A shell command such as ``!mmibc suite backbones`` is a separate process that cannot
read Colab secrets; run ``from mmibc.data.download import export_colab_secrets; export_colab_secrets()`` in a cell first (the
quickstart notebook does).

``unavailable_reason(name)`` is the cheap pre-flight check suites use: it says WHY a backbone cannot be loaded right now (gated and no
token, and not already in the local cache), so the run can be skipped instead of failing after the data pipeline has been built.
"""

from __future__ import annotations

import logging
import os
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

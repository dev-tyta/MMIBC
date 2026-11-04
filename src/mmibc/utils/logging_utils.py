from __future__ import annotations

import logging
import platform
import subprocess
import sys
from pathlib import Path


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=handlers, force=True)


def git_info(repo: Path | None = None) -> dict:
    repo = repo or Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:  # not a git checkout (e.g. pip-installed)
        return {"commit": None, "dirty": None}


def env_info() -> dict:
    import numpy
    import pandas
    import sklearn
    import torch

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info

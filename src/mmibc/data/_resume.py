"""Resumable, threaded per-item processing with an append-only progress log (safe against Colab disconnects)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def run_resumable(
    items: list[dict], fn: Callable[[dict], dict], progress_path: Path, key: str = "uid", workers: int = 8,
    desc: str = "", force: bool = False, validate: Callable[[dict], bool] = lambda rec: True,
) -> dict[str, dict]:
    progress_path = Path(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if progress_path.exists():
        if force:
            progress_path.unlink()
        else:
            for line in progress_path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if validate(rec):
                        done[rec[key]] = rec
    todo = [it for it in items if it[key] not in done]
    if todo:
        with open(progress_path, "a") as pf, ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for rec in tqdm(ex.map(fn, todo), total=len(todo), desc=desc):
                done[rec[key]] = rec
                pf.write(json.dumps(rec, default=_default) + "\n")
                pf.flush()
    return done

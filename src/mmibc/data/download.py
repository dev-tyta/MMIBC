"""Download datasets and inspect what actually arrived."""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from mmibc.data.registry import REGISTRY, DatasetSpec
from mmibc.utils.paths import ProjectPaths, in_colab

log = logging.getLogger(__name__)


def setup_kaggle_credentials() -> None:
    """Make Kaggle credentials available from env, Colab secrets, or ~/.kaggle/kaggle.json (never printed)."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return
    if in_colab():
        try:
            from google.colab import userdata  # type: ignore

            os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
            os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
            return
        except Exception as exc:  # secret missing or notebook access not granted
            log.warning("Could not read Kaggle secrets from Colab: %s", exc)
    raise RuntimeError(
        "No Kaggle credentials found. In Colab: add secrets KAGGLE_USERNAME and KAGGLE_KEY (key icon in the left bar) "
        "and enable notebook access. Elsewhere: set the env vars or create ~/.kaggle/kaggle.json."
    )


def _glob_any(root: Path, pattern: str) -> list[Path]:
    hits: list[Path] = []
    for alt in pattern.split("|"):
        hits.extend(root.glob(alt))
    return hits


def validate_download(spec: DatasetSpec, path: Path) -> list[str]:
    """Return the list of expected patterns that matched nothing (empty list = OK)."""
    return [pat for pat in spec.expected if not _glob_any(path, pat)]


def _place(src: Path, dest: Path, copy: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dest / item.name
        if copy:
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        else:
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), str(target))


def _download_url(url: str, dest: Path) -> None:
    import requests
    from tqdm import tqdm

    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "download.zip"
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zpath, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    zpath.unlink()


def download_dataset(key: str, paths: ProjectPaths, force: bool = False) -> Path:
    """Fetch dataset ``key`` into ``paths.raw/<key>`` and validate it. Idempotent."""
    spec = REGISTRY[key]
    dest = paths.raw / key
    if dest.exists() and not force and not validate_download(spec, dest):
        log.info("%s already present at %s", key, dest)
        return dest

    if spec.kind == "manual":
        raise RuntimeError(f"'{key}' must be downloaded manually into {dest}. {spec.note}")

    errors: list[str] = []
    if spec.kind == "kaggle":
        try:
            setup_kaggle_credentials()
            import kagglehub  # imported lazily: only needed for downloads

            cache_path = Path(kagglehub.dataset_download(spec.ref))
            # Local raw dir: move (cheap, same disk). Drive raw dir: copy, then drop the local cache copy.
            _place(cache_path, dest, copy=paths.raw_on_drive)
            if paths.raw_on_drive:
                shutil.rmtree(cache_path, ignore_errors=True)
        except Exception as exc:
            errors.append(f"kaggle:{spec.ref}: {exc}")
    for url in ([spec.ref] if spec.kind == "url" else []) + list(spec.fallback_urls):
        if dest.exists() and not validate_download(spec, dest):
            break
        try:
            log.info("Downloading %s ...", url)
            _download_url(url, dest)
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    missing = validate_download(spec, dest) if dest.exists() else list(spec.expected)
    if missing:
        raise RuntimeError(
            f"Download of '{key}' incomplete. Missing: {missing}. Errors: {errors}. "
            f"If this is the Kaggle slug, check {spec.note}"
        )
    log.info("%s ready at %s", key, dest)
    return dest


def inspect_dataset(root: Path, max_names: int = 6) -> str:
    """Human-readable structure report: directory tree (2 levels), file-type counts, sample file names, CSV headers."""
    root = Path(root)
    if not root.exists():
        return f"{root} does not exist"
    lines = [f"Structure report for {root}", ""]
    ext_count: Counter[str] = Counter()
    for p in root.rglob("*"):
        if p.is_file():
            ext_count[p.suffix.lower()] += 1
    lines.append("File types: " + ", ".join(f"{e or '(none)'}={n}" for e, n in ext_count.most_common()))
    lines.append("")

    def walk(d: Path, depth: int) -> None:
        children = sorted(d.iterdir())
        files = [c for c in children if c.is_file()]
        dirs = [c for c in children if c.is_dir()]
        names = ", ".join(f.name for f in files[:max_names]) + (" ..." if len(files) > max_names else "")
        lines.append(f"{'  ' * depth}{d.name}/  [{len(files)} files, {len(dirs)} dirs]  {names}")
        if depth < 3:
            for sub in dirs[:30]:
                walk(sub, depth + 1)

    walk(root, 0)
    csvs = list(root.rglob("*.csv"))[:8] + list(root.rglob("*.xlsx"))[:4]
    if csvs:
        lines += ["", "Tabular files (first lines):"]
        for c in csvs:
            lines.append(f"- {c.relative_to(root)}")
            if c.suffix == ".csv":
                try:
                    with open(c, encoding="utf-8", errors="replace") as f:
                        lines += ["    " + next(f, "").strip()[:200] for _ in range(3)]
                except OSError:
                    pass
    return "\n".join(lines)

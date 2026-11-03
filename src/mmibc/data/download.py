"""Download datasets and inspect what actually arrived."""

from __future__ import annotations

import json
import logging
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from mmibc.data.registry import REGISTRY, DatasetSpec
from mmibc.utils.paths import ProjectPaths, in_colab

log = logging.getLogger(__name__)


KAGGLE_SECRETS = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")
EXPORTABLE_SECRETS = (*KAGGLE_SECRETS, "HF_TOKEN")
NOT_IN_KERNEL = "not in a notebook kernel"

_HOW_TO_EXPORT = (
    "Colab secrets can only be read by notebook cells, not by `!mmibc ...` shell commands (those are separate processes). "
    "Run this in a cell first; it copies the secrets into environment variables, which shell commands do inherit:\n"
    "    from mmibc.data.download import export_colab_secrets; export_colab_secrets()"
)
_WHICH_CREDENTIALS = (
    "Provide ONE of: (a) KAGGLE_API_TOKEN, the token from 'Generate New Token' in your Kaggle account Settings "
    "(current kagglehub); (b) the legacy pair KAGGLE_USERNAME + KAGGLE_KEY from 'Create Legacy API Key'; "
    "(c) a file ~/.kaggle/access_token or ~/.kaggle/kaggle.json. In Colab, store them as secrets (key icon, left bar) "
    "with 'Notebook access' switched ON for each one."
)


def kaggle_credentials_present() -> bool:
    """True if kagglehub will find credentials: env token, legacy env pair, or a token / kaggle.json file."""
    env, home = os.environ, Path.home() / ".kaggle"
    return bool(
        env.get("KAGGLE_API_TOKEN")
        or (env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"))
        or (home / "access_token").exists()
        or (home / "kaggle.json").exists()
    )


def export_colab_secrets(names: tuple[str, ...] = EXPORTABLE_SECRETS, verbose: bool = True) -> dict[str, str]:
    """Copy Colab secrets into environment variables so that ``!mmibc ...`` subprocesses can use them.

    Must run in a notebook cell: ``google.colab.userdata`` only works inside the notebook's own kernel process. Returns
    ``{name: status}`` and prints the status only (a secret's value is never printed or logged). A secret that is not
    defined, or whose 'Notebook access' toggle is off, is reported and skipped; it is not an error here.
    """
    status: dict[str, str] = {}
    for name in names:
        if os.environ.get(name):
            status[name] = "already set"
            continue
        try:
            from google.colab import userdata  # type: ignore

            os.environ[name] = userdata.get(name)
            status[name] = "exported"
        except ImportError:
            status[name] = "not in Colab"
        except Exception as exc:  # SecretNotFoundError, NotebookAccessError, or no kernel at all
            kernel_missing = isinstance(exc, AttributeError) and "kernel" in str(exc)
            status[name] = NOT_IN_KERNEL if kernel_missing else f"missing ({type(exc).__name__}: check the name and the Notebook-access toggle)"
    if verbose:
        for name, s in status.items():
            print(f"  {name:<18} {s}")
    return status


def _write_legacy_kaggle_json() -> None:
    """Colab only: materialise the legacy KAGGLE_USERNAME/KAGGLE_KEY pair as ~/.kaggle/kaggle.json (the documented legacy route)."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    target = Path.home() / ".kaggle" / "kaggle.json"
    if in_colab() and user and key and not os.environ.get("KAGGLE_API_TOKEN") and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"username": user, "key": key}))
        target.chmod(0o600)


def setup_kaggle_credentials() -> None:
    """Make sure Kaggle credentials are available (never printed); raise an actionable error if they are not."""
    if not kaggle_credentials_present() and in_colab():
        status = export_colab_secrets(KAGGLE_SECRETS, verbose=False)
        if not kaggle_credentials_present() and NOT_IN_KERNEL in status.values():
            raise RuntimeError(f"Kaggle credentials not visible from this process. {_HOW_TO_EXPORT}\n{_WHICH_CREDENTIALS}")
    if not kaggle_credentials_present():
        raise RuntimeError(f"No Kaggle credentials found. {_WHICH_CREDENTIALS}")
    _write_legacy_kaggle_json()


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
    if not zipfile.is_zipfile(zpath):
        size = zpath.stat().st_size
        head = zpath.read_bytes()[:200].lstrip().lower()
        kind = "a web page (HTML)" if head.startswith((b"<!doctype", b"<html")) or b"<html" in head else "something that is not a zip archive"
        zpath.unlink()
        raise RuntimeError(f"{url} returned {kind} of {size:,} bytes instead of a zip file: the link is probably dead, moved, or needs a browser")
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
            hint = ""
            if any(t in str(exc) for t in ("401", "403", "Unauthorized", "Forbidden")):
                hint = " (Kaggle rejected the credentials: generate a new token in your Kaggle Settings and store it as the secret KAGGLE_API_TOKEN)"
            errors.append(f"kaggle:{spec.ref}: {exc}{hint}")
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

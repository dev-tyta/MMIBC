"""Download datasets and inspect what actually arrived."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from mmibc.data.registry import REGISTRY, DatasetSpec
from mmibc.utils.paths import ProjectPaths, in_colab

log = logging.getLogger(__name__)


INCOMPLETE_MARKER = ".mmibc_incomplete"
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


def _main_missing(spec: DatasetSpec, path: Path) -> list[str]:
    """Expected patterns of the main archive that matched nothing."""
    return [pat for pat in spec.expected if not _glob_any(path, pat)]


def validate_download(spec: DatasetSpec, path: Path) -> list[str]:
    """Return what is missing: expected patterns that matched nothing, plus extra files that are absent (empty list = OK)."""
    return _main_missing(spec, path) + [name for name, _ in spec.extra_files if not (Path(path) / name).is_file()]


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


_UA_DEFAULT = "Mozilla/5.0"
_UA_TOOL = "mmibc/0.2 (+https://github.com/dev-tyta/MMIBC; python-requests)"


def _http_get(url: str):
    """GET ``url`` as a streaming response, or raise a RuntimeError that says what the server actually answered.

    A 403 is retried once with a User-Agent that names this tool (some hosts refuse anonymous or browser-like agents). The
    status and the start of the reply are put in the error: a bare "403 Forbidden" cannot be diagnosed.
    """
    import requests

    r = None
    for ua in (_UA_DEFAULT, _UA_TOOL):
        r = requests.get(url, stream=True, timeout=60, headers={"User-Agent": ua})
        if getattr(r, "status_code", 200) != 403 or ua == _UA_TOOL:
            break
        r.close()
    status = getattr(r, "status_code", 200)
    if status >= 400:
        try:
            body = " ".join(r.content[:400].decode("utf-8", "replace").split())[:160]
        except Exception:  # noqa: BLE001
            body = ""
        r.close()
        raise RuntimeError(f"{url} answered HTTP {status} {getattr(r, 'reason', '') or ''}".rstrip() + (f" with: {body!r}" if body else ""))
    return r


def _save_stream(r, path: Path) -> None:
    from tqdm import tqdm

    total = int(r.headers.get("content-length", 0))
    with r, open(path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))


def _md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324  integrity check against a published checksum, not security
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unpack_zip(zpath: Path, dest: Path, md5: str, label: str) -> None:
    """Check a zip (is it a zip? does it match the published MD5?) and extract it into ``dest``."""
    if not zipfile.is_zipfile(zpath):
        size = zpath.stat().st_size
        head = zpath.read_bytes()[:200].lstrip().lower()
        kind = "a web page (HTML)" if head.startswith((b"<!doctype", b"<html")) or b"<html" in head else "something that is not a zip archive"
        raise RuntimeError(f"{label} is {kind} of {size:,} bytes, not a zip file: the link is probably dead, moved, or needs a browser")
    if md5 and (got := _md5(zpath)) != md5.lower():
        raise RuntimeError(f"{label}: MD5 {got} does not match the published {md5}; the file is corrupt, incomplete or a different version")
    with zipfile.ZipFile(zpath) as z:
        _check_space(dest, sum(i.file_size for i in z.infolist()), label)
        z.extractall(dest)


def _download_url(url: str, dest: Path, md5: str = "") -> None:
    """Download a zip to local temporary storage and extract it into ``dest`` (which may be on Drive).

    The archive itself is never written to ``dest``: that halves the writes to Drive and leaves no stray zip behind.
    """
    dest.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="mmibc_dl_"))
    try:
        zpath = tmpdir / "download.zip"
        _save_stream(_http_get(url), zpath)
        _unpack_zip(zpath, dest, md5, url)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _unpack_local_zips(spec: DatasetSpec, dest: Path) -> tuple[bool, list[str]]:
    """Use a zip that was placed in the dataset folder by hand (e.g. downloaded in a browser and uploaded to Drive).

    This is the way out when a host refuses automated downloads. The zip is checked against the published MD5 when known and
    is left in place.
    """
    errors: list[str] = []
    for z in sorted(Path(dest).glob("*.zip")):
        try:
            log.info("Using the zip found in %s: %s", dest, z.name)
            _unpack_zip(z, dest, spec.md5, z.name)
            if not _main_missing(spec, dest):
                return True, errors
            errors.append(f"{z.name}: unpacked, but it does not contain what '{spec.key}' needs ({spec.expected})")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{z.name}: {exc}")
    return False, errors


def _tree_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def _check_space(dest: Path, need: int, what: str) -> None:
    """Refuse to start a copy that cannot fit (a Drive that fills up halfway leaves a broken dataset)."""
    probe = dest
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        total, _, free = shutil.disk_usage(probe)
    except OSError:
        return
    if total > 0 and need > free:
        raise RuntimeError(
            f"Not enough free space for {what}: needs {need / 1e9:.1f} GB, {free / 1e9:.1f} GB free at {probe}. "
            "Free up Google Drive space, or set paths.raw_on_drive=false to keep raw downloads on the (temporary) local disk."
        )


def _download_file(url: str, target: Path) -> None:
    """Fetch one plain file (not an archive). A web page served in its place is reported as such and not kept."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    _save_stream(_http_get(url), tmp)
    head = tmp.read_bytes()[:200].lstrip().lower()
    if head.startswith((b"<!doctype", b"<html")) or b"<html" in head or tmp.stat().st_size == 0:
        size = tmp.stat().st_size
        tmp.unlink()
        raise RuntimeError(f"{url} returned a web page or an empty body ({size:,} bytes) instead of {target.name}: the link may be dead or need a browser")
    tmp.replace(target)


def is_downloaded(key: str, paths: ProjectPaths) -> bool:
    """True if dataset ``key`` is fully present (not merely partly: a folder with the in-progress marker never counts)."""
    dest = paths.raw / key
    return dest.exists() and not (dest / INCOMPLETE_MARKER).exists() and not validate_download(REGISTRY[key], dest)


def download_dataset(key: str, paths: ProjectPaths, force: bool = False, use_kaggle: bool = True) -> Path:
    """Fetch dataset ``key`` into ``paths.raw/<key>`` and validate it. Idempotent."""
    spec = REGISTRY[key]
    dest = paths.raw / key
    marker = dest / INCOMPLETE_MARKER  # present while a download is running or was interrupted: never trust such a folder
    if dest.exists() and not force and not marker.exists() and not validate_download(spec, dest):
        log.info("%s already present at %s", key, dest)
        return dest

    if spec.kind == "manual":
        raise RuntimeError(f"'{key}' must be downloaded manually into {dest}. {spec.note}")

    # ``main_ok`` is True only for an archive that is known complete: a folder without the in-progress marker that matches the
    # expected patterns, or one this run fetched in full. Files left behind by an interrupted attempt can satisfy a glob (half a
    # dataset still contains PNGs), so they never count. When only an extra file is missing the archive is not fetched again.
    main_ok = dest.exists() and not force and not marker.exists() and not _main_missing(spec, dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not main_ok:
        marker.write_text("download in progress or interrupted; delete this folder or re-run `mmibc download`\n")
    errors: list[str] = []
    if not main_ok:  # a zip placed in the folder by hand beats any network access
        main_ok, local_errors = _unpack_local_zips(spec, dest)
        errors += local_errors
    if spec.kind == "kaggle" and not main_ok and not use_kaggle:
        errors.append("kaggle: skipped (no credentials, see the message above)")
    elif spec.kind == "kaggle" and not main_ok:
        try:
            setup_kaggle_credentials()
            import kagglehub  # imported lazily: only needed for downloads

            cache_path = Path(kagglehub.dataset_download(spec.ref))
            if paths.raw_on_drive:
                _check_space(dest, _tree_bytes(cache_path), f"'{key}' on Drive")
            # Local raw dir: move (cheap, same disk). Drive raw dir: copy, then drop the local cache copy.
            _place(cache_path, dest, copy=paths.raw_on_drive)
            if paths.raw_on_drive:
                shutil.rmtree(cache_path, ignore_errors=True)
            main_ok = not _main_missing(spec, dest)
        except Exception as exc:
            hint = ""
            if any(t in str(exc) for t in ("401", "403", "Unauthorized", "Forbidden")):
                hint = " (Kaggle rejected the credentials: generate a new token in your Kaggle Settings and store it as the secret KAGGLE_API_TOKEN)"
            errors.append(f"kaggle:{spec.ref}: {exc}{hint}")
    for url in ([spec.ref] if spec.kind == "url" else []) + list(spec.fallback_urls):
        if main_ok:
            break
        try:
            log.info("Downloading %s ...", url)
            _download_url(url, dest, spec.md5)
            main_ok = not _main_missing(spec, dest)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    for name, url in spec.extra_files:
        if (dest / name).is_file() and not force:
            continue
        try:
            log.info("Downloading %s ...", url)
            _download_file(url, dest / name)
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    missing = _main_missing(spec, dest) if not main_ok else []
    if not main_ok and not missing:
        missing = ["(the archive did not finish downloading; files from an earlier attempt may be present but cannot be trusted)"]
    missing += [name for name, _ in spec.extra_files if not (dest / name).is_file()]
    if missing:
        # A missing credential explains the failure on its own; the dataset note (slug, layout caveats) would only bury it.
        no_creds = bool(errors) and all(("No Kaggle credentials" in e or "Kaggle credentials not visible" in e or "no credentials" in e) for e in errors)
        detail = " Add the credentials (see above), then re-run." if no_creds else f" If this is the Kaggle slug, check {spec.note}"
        raise RuntimeError(f"Download of '{key}' incomplete. Missing: {missing}. Errors: {errors}.{detail}")
    marker.unlink(missing_ok=True)
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
            else:
                lines += _describe_spreadsheet(c)
    return "\n".join(lines)


def _describe_spreadsheet(path: Path, rows: int = 3, max_unique: int = 12) -> list[str]:
    """Per sheet: size, column names, the first rows, and the value counts of low-cardinality columns (likely labels)."""
    import pandas as pd

    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except ImportError:
        return ["    (reading .xlsx needs openpyxl: pip install openpyxl)"]
    except Exception as exc:  # corrupt file, HTML saved as .xlsx, ...
        return [f"    (could not read as a spreadsheet: {type(exc).__name__}: {exc})"]
    out: list[str] = []
    for name, df in sheets.items():
        out.append(f"    sheet '{name}': {len(df)} rows x {df.shape[1]} columns")
        out.append("    columns: " + " | ".join(str(c) for c in df.columns))
        for _, r in df.head(rows).iterrows():
            out.append("    > " + " | ".join(str(v)[:24] for v in r.tolist())[:240])
        for col in df.columns:
            s = df[col].dropna()
            if 0 < s.nunique() <= max_unique:
                counts = ", ".join(f"{k}={v}" for k, v in s.astype(str).value_counts().items())
                out.append(f"    values of '{col}': {counts}"[:240])
    return out

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging(log_path: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("toxicity_screening")


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_transient_windows_lock(error: OSError) -> bool:
    """Return whether an OS error is consistent with a temporary Windows lock."""
    if isinstance(error, PermissionError):
        return True
    return getattr(error, "winerror", None) in {5, 32, 33}


def atomic_write_json(
    data: Any,
    path: str | Path,
    *,
    attempts: int = 8,
    initial_delay: float = 0.05,
) -> Path:
    """Write JSON atomically with retry protection for transient Windows locks.

    Dropbox, antivirus software, file previews, and indexers can briefly lock an
    existing destination file on Windows. A unique temporary file avoids
    collisions between consecutive writes, while bounded exponential backoff
    allows ``os.replace`` to succeed once the lock is released.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if initial_delay < 0:
        raise ValueError("initial_delay must be non-negative")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, default=str)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)

        for attempt in range(attempts):
            try:
                os.replace(temporary_path, destination)
                temporary_path = None
                return destination
            except OSError as error:
                if not _is_transient_windows_lock(error) or attempt == attempts - 1:
                    raise
                delay = min(initial_delay * (2**attempt), 1.0)
                time.sleep(delay)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    return destination


def require_paths(paths: Iterable[str | Path], message: str | None = None) -> None:
    missing = [str(Path(path)) for path in paths if not Path(path).exists()]
    if missing:
        extra = f" {message}" if message else ""
        raise FileNotFoundError(f"Required artifacts are missing: {missing}.{extra}")


def package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def environment_snapshot(path: str | Path) -> dict[str, Any]:
    packages = [
        "numpy", "pandas", "scipy", "rdkit", "scikit-learn", "xgboost",
        "torch", "torch-geometric", "optuna", "shap", "captum", "pytdc",
        "matplotlib", "joblib", "PyYAML",
    ]
    payload = {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": package_versions(packages),
        "environment": {
            key: os.environ.get(key)
            for key in ["CONDA_PREFIX", "CUDA_VISIBLE_DEVICES", "TOX_SCREEN_PROFILE"]
        },
    }
    atomic_write_json(payload, path)
    return payload


def stable_id(*parts: object, length: int = 16) -> str:
    text = "||".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]

from __future__ import annotations

import gzip
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .utils import sha256_file, utc_now


class DownloadError(RuntimeError):
    """Raised when a dataset cannot be acquired reproducibly."""


def download_file(
    url: str,
    destination: str | Path,
    *,
    timeout: int = 120,
    retries: int = 3,
    chunk_size: int = 1024 * 1024,
    user_agent: str = "toxicity-screening-research/0.1",
) -> dict[str, Any]:
    """Download bytes without transformation, using atomic replacement and retries."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": user_agent}
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, timeout=timeout, stream=True, headers=headers) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
            temporary.replace(destination)
            return {
                "path": str(destination),
                "url": url,
                "download_date": utc_now(),
                "checksum": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "content_type": response.headers.get("content-type"),
            }
        except Exception as exc:  # network libraries raise heterogeneous exceptions
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise DownloadError(f"Failed to download {url}: {'; '.join(errors)}")


def acquire_tdc_dataset(name: str, destination: str | Path) -> dict[str, Any]:
    """Acquire an ML-ready TDC table and preserve it as the raw distributed table."""
    try:
        from tdc.single_pred import Tox
    except ImportError as exc:
        raise DownloadError(
            "pytdc is required for TDC acquisition. Install the pinned environment first."
        ) from exc
    try:
        dataset = Tox(name=name)
        frame = dataset.get_data(format="df")
    except Exception as exc:
        raise DownloadError(f"TDC acquisition failed for {name}: {exc}") from exc
    if frame.empty:
        raise DownloadError(f"TDC returned an empty dataset for {name}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return {
        "path": str(destination),
        "provider": "Therapeutics Data Commons",
        "tdc_name": name,
        "download_date": utc_now(),
        "checksum": sha256_file(destination),
        "number_of_records": int(len(frame)),
        "columns": list(frame.columns),
        "note": "Raw means the unmodified table returned by pytdc, not the primary experimental source.",
    }


def read_csv_preserve_missing(path: str | Path) -> pd.DataFrame:
    """Read CSV/CSV.GZ while preserving blank labels as NaN."""
    return pd.read_csv(path, keep_default_na=True, na_values=["", "NA", "NaN", "nan"])


def decompress_gzip(source: str | Path, destination: str | Path) -> Path:
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return destination

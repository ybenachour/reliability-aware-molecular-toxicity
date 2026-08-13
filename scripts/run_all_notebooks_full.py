from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Reuse the validated notebook runner, but replace its JSON writer with a
# Windows/Dropbox-safe implementation. Setting runner.__file__ to this wrapper
# also ensures every isolated worker process receives the same patch.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_remaining_notebooks as runner  # noqa: E402


def robust_atomic_write_json(
    payload: dict[str, Any],
    path: Path,
    attempts: int = 12,
    initial_delay: float = 0.20,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    delay = initial_delay
    last_error: OSError | None = None

    for attempt in range(attempts):
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
            return
        except OSError as exc:
            last_error = exc
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt + 1 < attempts:
                time.sleep(delay)
                delay = min(delay * 1.7, 3.0)

    # Reproducible status and summary files can safely use a direct-write
    # fallback when Dropbox keeps the destination locked during os.replace().
    try:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError:
        if last_error is not None:
            raise last_error
        raise


runner.atomic_write_json = robust_atomic_write_json
runner.__file__ = str(Path(__file__).resolve())
os.environ["TOX_SCREEN_PROFILE"] = "full"

# Permit direct execution without the batch launcher while retaining all
# original runner options, including --resume and --continue-on-error.
if "--worker" not in sys.argv:
    defaults = {
        "--root": str(PROJECT_ROOT),
        "--start": "0",
        "--end": "25",
        "--profile": "full",
        "--expected-python": str(Path(sys.executable).resolve()),
        "--cell-timeout": "43200",
    }
    for flag, value in defaults.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])


if __name__ == "__main__":
    raise SystemExit(runner.main())

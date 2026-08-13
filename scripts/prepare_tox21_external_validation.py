from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _project_root(start: str | Path) -> Path:
    current = Path(start).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate project root from {current}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the official NCATS Tox21 Challenge final-evaluation "
            "files for locked external validation."
        )
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root or a path inside the project.",
    )
    args = parser.parse_args()

    root = _project_root(args.root)
    os.chdir(root)
    source_dir = root / "src"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    from toxicity_screening.external_validation import (
        prepare_tox21_external_validation,
    )

    summary = prepare_tox21_external_validation(root)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

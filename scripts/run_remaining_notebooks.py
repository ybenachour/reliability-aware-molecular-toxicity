from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NOTEBOOK_PATTERN = re.compile(r"^(?P<number>\d{2})_.*\.ipynb$")
DEFAULT_START = 9
DEFAULT_END = 25


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def locate_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "notebooks").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate the project root from {current}")


def notebook_number(path: Path) -> int | None:
    match = NOTEBOOK_PATTERN.match(path.name)
    return int(match.group("number")) if match else None


def discover_notebooks(root: Path, start: int, end: int) -> list[Path]:
    selected: list[tuple[int, Path]] = []
    for path in (root / "notebooks").glob("*.ipynb"):
        number = notebook_number(path)
        if number is not None and start <= number <= end:
            selected.append((number, path))
    selected.sort(key=lambda item: (item[0], item[1].name))
    return [path for _, path in selected]


def create_private_kernelspec(root: Path) -> tuple[str, Path]:
    kernel_name = "toxicity-screening-runner"
    jupyter_root = root / "reports" / "notebook_runner_jupyter"
    kernel_directory = jupyter_root / "kernels" / kernel_name
    kernel_directory.mkdir(parents=True, exist_ok=True)
    kernel_specification = {
        "argv": [
            str(Path(sys.executable).resolve()),
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": "Python (toxicity-screening runner)",
        "language": "python",
        "metadata": {"debugger": True},
    }
    atomic_write_json(kernel_specification, kernel_directory / "kernel.json")
    return kernel_name, jupyter_root


def build_environment(root: Path, profile: str, jupyter_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "TOX_SCREEN_PROFILE": profile,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(root / "reports" / "_matplotlib_config"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONFAULTHANDLER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    source_path = str(root / "src")
    current_python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        source_path
        if not current_python_path
        else source_path + os.pathsep + current_python_path
    )
    current_jupyter_path = environment.get("JUPYTER_PATH", "")
    environment["JUPYTER_PATH"] = (
        str(jupyter_root)
        if not current_jupyter_path
        else str(jupyter_root) + os.pathsep + current_jupyter_path
    )
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return environment


def read_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def execute_one_notebook(
    *,
    root: Path,
    notebook_path: Path,
    output_path: Path,
    log_path: Path,
    status_path: Path,
    kernel_name: str,
    profile: str,
    cell_timeout: int,
    environment: dict[str, str],
) -> dict[str, Any]:
    started_at = utc_now()
    started_clock = time.monotonic()
    command = [
        str(Path(sys.executable).resolve()),
        "-X",
        "faulthandler",
        str(Path(__file__).resolve()),
        "--worker",
        "--root",
        str(root),
        "--notebook",
        str(notebook_path),
        "--output",
        str(output_path),
        "--status",
        str(status_path),
        "--kernel-name",
        kernel_name,
        "--profile",
        profile,
        "--cell-timeout",
        str(cell_timeout),
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        header = (
            f"Notebook: {notebook_path.name}\n"
            f"Started: {started_at}\n"
            f"Python: {sys.executable}\n"
            f"Profile: {profile}\n"
            f"Kernel: {kernel_name}\n"
            f"Command: {subprocess.list2cmdline(command)}\n"
            + "-" * 79
            + "\n"
        )
        print(header, end="")
        log_handle.write(header)
        log_handle.flush()

        process = subprocess.Popen(
            command,
            cwd=str(root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()

    elapsed_seconds = round(time.monotonic() - started_clock, 3)
    worker_status = read_status(status_path)
    result = {
        "notebook": notebook_path.name,
        "notebook_number": notebook_number(notebook_path),
        "input_path": str(notebook_path.relative_to(root)),
        "output_path": str(output_path.relative_to(root)),
        "log_path": str(log_path.relative_to(root)),
        "status_path": str(status_path.relative_to(root)),
        "profile": profile,
        "python": str(Path(sys.executable).resolve()),
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "return_code": return_code,
        "status": "passed" if return_code == 0 else "failed",
        "current_cell_index": worker_status.get("current_cell_index"),
        "current_code_cell_number": worker_status.get("current_code_cell_number"),
        "current_cell_id": worker_status.get("current_cell_id"),
        "error_type": worker_status.get("error_type"),
        "error_message": worker_status.get("error_message"),
    }
    if return_code not in (0, 1):
        result["failure_class"] = "process_or_native_failure"
    elif return_code == 1:
        result["failure_class"] = "notebook_execution_failure"
    return result


def write_summary(root: Path, results: list[dict[str, Any]], profile: str) -> tuple[Path, Path]:
    report_directory = root / "reports"
    report_directory.mkdir(parents=True, exist_ok=True)
    json_path = report_directory / "remaining_notebooks_execution_summary.json"
    csv_path = report_directory / "remaining_notebooks_execution_summary.csv"
    payload = {
        "generated_at": utc_now(),
        "profile": profile,
        "python": str(Path(sys.executable).resolve()),
        "total": len(results),
        "passed": sum(item["status"] == "passed" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "results": results,
    }
    atomic_write_json(payload, json_path)

    fieldnames = [
        "notebook_number",
        "notebook",
        "status",
        "return_code",
        "failure_class",
        "current_cell_index",
        "current_code_cell_number",
        "current_cell_id",
        "error_type",
        "error_message",
        "elapsed_seconds",
        "profile",
        "input_path",
        "output_path",
        "log_path",
        "status_path",
        "started_at",
        "finished_at",
        "python",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in results:
            writer.writerow(item)
    return json_path, csv_path


def run_parent(arguments: argparse.Namespace) -> int:
    root = locate_project_root(Path(arguments.root) if arguments.root else None)
    notebooks = discover_notebooks(root, arguments.start, arguments.end)
    if not notebooks:
        raise RuntimeError(
            f"No notebooks numbered {arguments.start:02d}-{arguments.end:02d} "
            f"were found under {root / 'notebooks'}"
        )

    if Path(sys.executable).resolve() != Path(arguments.expected_python).resolve():
        raise RuntimeError(
            "The runner is using the wrong Python interpreter.\n"
            f"Expected: {Path(arguments.expected_python).resolve()}\n"
            f"Actual:   {Path(sys.executable).resolve()}"
        )

    kernel_name, jupyter_root = create_private_kernelspec(root)
    environment = build_environment(root, arguments.profile, jupyter_root)
    results: list[dict[str, Any]] = []

    print(f"Project root: {root}")
    print(f"Runner Python: {Path(sys.executable).resolve()}")
    print(f"Execution profile: {arguments.profile}")
    print(f"Notebook range: {arguments.start:02d}-{arguments.end:02d}")
    print(f"Notebooks selected: {len(notebooks)}")
    print()

    for position, notebook_path in enumerate(notebooks, start=1):
        stem = notebook_path.stem
        output_path = root / "reports" / "executed_notebooks" / notebook_path.name
        log_path = root / "logs" / "notebook_runs" / f"{stem}.log"
        status_path = root / "logs" / "notebook_runs" / f"{stem}.status.json"
        if status_path.exists():
            status_path.unlink()

        print("=" * 79)
        print(f"[{position}/{len(notebooks)}] {notebook_path.name}")
        print("=" * 79)
        result = execute_one_notebook(
            root=root,
            notebook_path=notebook_path,
            output_path=output_path,
            log_path=log_path,
            status_path=status_path,
            kernel_name=kernel_name,
            profile=arguments.profile,
            cell_timeout=arguments.cell_timeout,
            environment=environment,
        )
        results.append(result)
        json_path, csv_path = write_summary(root, results, arguments.profile)
        print(
            f"Result: {result['status'].upper()} "
            f"(exit code {result['return_code']}, {result['elapsed_seconds']} s)"
        )
        if result["status"] == "failed":
            if result.get("current_cell_index") is not None:
                print(
                    "Last tracked cell: "
                    f"notebook index {result['current_cell_index']}, "
                    f"code-cell number {result.get('current_code_cell_number')}, "
                    f"cell id {result.get('current_cell_id')}"
                )
            if result.get("error_type") or result.get("error_message"):
                print(
                    f"Error: {result.get('error_type') or 'unknown'}: "
                    f"{result.get('error_message') or 'No Python exception message was captured.'}"
                )
            print(f"Log: {result['log_path']}")
            print(f"Partial executed notebook: {result['output_path']}")
            if not arguments.continue_on_error:
                print("Stopping at the first failure to avoid dependent downstream errors.")
                print(f"Summary JSON: {json_path.relative_to(root)}")
                print(f"Summary CSV: {csv_path.relative_to(root)}")
                return 1
        print()

    json_path, csv_path = write_summary(root, results, arguments.profile)
    failures = [item for item in results if item["status"] == "failed"]
    print("=" * 79)
    print("Notebook execution completed.")
    print(f"Passed: {len(results) - len(failures)}")
    print(f"Failed: {len(failures)}")
    print(f"Summary JSON: {json_path.relative_to(root)}")
    print(f"Summary CSV: {csv_path.relative_to(root)}")
    return 1 if failures else 0


def run_worker(arguments: argparse.Namespace) -> int:
    os.environ["TOX_SCREEN_PROFILE"] = arguments.profile
    root = Path(arguments.root).resolve()
    notebook_path = Path(arguments.notebook).resolve()
    output_path = Path(arguments.output).resolve()
    status_path = Path(arguments.status).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(notebook_path, as_version=4)
    code_cell_counter = 0

    class TrackingNotebookClient(NotebookClient):
        def execute_cell(self, cell: Any, cell_index: int, *args: Any, **kwargs: Any) -> Any:
            nonlocal code_cell_counter
            code_cell_number = None
            if cell.get("cell_type") == "code":
                code_cell_counter += 1
                code_cell_number = code_cell_counter
            atomic_write_json(
                {
                    "status": "running",
                    "notebook": notebook_path.name,
                    "profile": arguments.profile,
                    "python": str(Path(sys.executable).resolve()),
                    "kernel_name": arguments.kernel_name,
                    "current_cell_index": cell_index,
                    "current_code_cell_number": code_cell_number,
                    "current_cell_id": cell.get("id"),
                    "cell_type": cell.get("cell_type"),
                    "updated_at": utc_now(),
                },
                status_path,
            )
            return super().execute_cell(cell, cell_index, *args, **kwargs)

    client = TrackingNotebookClient(
        notebook,
        timeout=arguments.cell_timeout,
        kernel_name=arguments.kernel_name,
        resources={"metadata": {"path": str(root)}},
        allow_errors=False,
        record_timing=True,
    )

    print(f"Worker Python: {Path(sys.executable).resolve()}", flush=True)
    print(f"Notebook: {notebook_path.name}", flush=True)
    print(f"Kernel: {arguments.kernel_name}", flush=True)
    print(f"Profile: {arguments.profile}", flush=True)
    print(f"Cell timeout: {arguments.cell_timeout} seconds", flush=True)

    try:
        executed = client.execute()
        nbformat.write(executed, output_path)
        previous = read_status(status_path)
        atomic_write_json(
            {
                **previous,
                "status": "passed",
                "finished_at": utc_now(),
                "output_path": str(output_path),
            },
            status_path,
        )
        print(f"Created: {output_path}", flush=True)
        print("Notebook completed successfully.", flush=True)
        return 0
    except BaseException as exc:
        try:
            nbformat.write(notebook, output_path)
        except BaseException:
            pass
        previous = read_status(status_path)
        error_message = str(exc)
        atomic_write_json(
            {
                **previous,
                "status": "failed",
                "finished_at": utc_now(),
                "output_path": str(output_path),
                "error_type": type(exc).__name__,
                "error_message": error_message,
                "traceback": traceback.format_exc(),
            },
            status_path,
        )
        print("Notebook execution failed.", flush=True)
        print(f"Error type: {type(exc).__name__}", flush=True)
        print(f"Error message: {error_message}", flush=True)
        traceback.print_exc()
        print(f"Partial notebook saved: {output_path}", flush=True)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute toxicity-screening notebooks in isolated subprocesses."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", default=None)
    parser.add_argument("--start", type=int, default=DEFAULT_START)
    parser.add_argument("--end", type=int, default=DEFAULT_END)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--cell-timeout", type=int, default=3600)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--expected-python",
        default=r"D:\Users\anaconda3\envs\toxicity-screening\python.exe",
    )
    parser.add_argument("--notebook", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--status", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--kernel-name", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.worker:
        required = {
            "root": arguments.root,
            "notebook": arguments.notebook,
            "output": arguments.output,
            "status": arguments.status,
            "kernel_name": arguments.kernel_name,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"Worker mode is missing arguments: {', '.join(missing)}")
        return run_worker(arguments)
    if arguments.start > arguments.end:
        parser.error("--start must be less than or equal to --end")
    if arguments.cell_timeout <= 0:
        parser.error("--cell-timeout must be positive")
    return run_parent(arguments)


if __name__ == "__main__":
    raise SystemExit(main())

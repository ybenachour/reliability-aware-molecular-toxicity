from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration is missing or malformed."""


def project_root(start: str | Path | None = None) -> Path:
    """Find the repository root by locating ``pyproject.toml``."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").exists():
            return candidate
    raise ConfigError(f"Could not locate project root from {current}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Expected mapping in {path}")
    return data


def load_configs(root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    root_path = project_root(root)
    names = ["data_config", "model_config", "training_config", "endpoints"]
    return {name: load_yaml(root_path / "configs" / f"{name}.yaml") for name in names}


def execution_profile(configs: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    requested = os.getenv(
        "TOX_SCREEN_PROFILE",
        str(configs["data_config"].get("execution_profile", "smoke")),
    )
    profiles = configs["training_config"].get("profiles", {})
    if requested not in profiles:
        raise ConfigError(f"Unknown execution profile {requested!r}; available: {sorted(profiles)}")
    return requested, profiles[requested]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw: Path
    interim: Path
    processed: Path
    metadata: Path
    external: Path
    models: Path
    results: Path
    figures: Path
    tables: Path
    reports: Path

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "ProjectPaths":
        r = project_root(root)
        return cls(
            root=r,
            raw=r / "data/raw",
            interim=r / "data/interim",
            processed=r / "data/processed",
            metadata=r / "data/metadata",
            external=r / "data/external",
            models=r / "models",
            results=r / "results",
            figures=r / "figures",
            tables=r / "tables",
            reports=r / "reports",
        )

    def ensure(self) -> None:
        for value in self.__dict__.values():
            Path(value).mkdir(parents=True, exist_ok=True)

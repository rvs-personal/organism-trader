from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_experiment(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONFIG_DIR / "experiment.yaml")


def load_universe(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONFIG_DIR / "universe.yaml")


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)

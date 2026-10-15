"""Configuration + path helpers. Every script imports ROOT from here so paths are stable
whether the code runs from the repo root, a notebook, or a test."""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def path(*parts: str) -> Path:
    """Repo-relative path (absolute Path object)."""
    return ROOT.joinpath(*parts)


@lru_cache(maxsize=None)
def load_yaml(rel: str) -> dict[str, Any]:
    with open(path(rel), encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_cfg() -> dict[str, Any]:
    return load_yaml("configs/data.yaml")


def models_cfg() -> dict[str, Any]:
    return load_yaml("configs/models.yaml")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_env() -> None:
    """Load .env from the repo root into os.environ (without printing anything)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(path(".env"), override=False)
    except Exception:  # noqa: BLE001
        pass


def pick_device() -> str:
    """cuda > mps > cpu (configs/models.yaml: device: auto)."""
    want = models_cfg().get("device", "auto")
    if want != "auto":
        return want
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def cpu_threads() -> int:
    return max(1, os.cpu_count() or 1)

"""Reproducibility manifest: git sha, python version, dataset checksum."""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else "n/a"
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def build_manifest(dataset_checksum: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "dataset_checksum": dataset_checksum,
        "config": config,
    }

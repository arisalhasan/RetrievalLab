"""Portable experiment provenance without recording private machine paths."""

import importlib.metadata
import platform
import subprocess
from pathlib import Path

import psutil

from retrieval_lab.data import sha256_file


def code_state() -> dict:
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        if Path(top).resolve() != Path.cwd().resolve():
            return {"commit": None, "working_tree_dirty": None}
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
        return {"commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "working_tree_dirty": None}


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "machine_ram_bytes": psutil.virtual_memory().total,
        "packages": dict(
            sorted(
                (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
            )
        ),
    }


def source_hashes() -> dict:
    root = Path(__file__).resolve().parent
    return {path.name: sha256_file(path) for path in sorted(root.glob("*.py"))}

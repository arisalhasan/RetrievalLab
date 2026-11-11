"""Install staged artifacts without replacing earlier evidence."""

import time
from pathlib import Path


def install_directory(source: Path, destination: Path, root: Path) -> None:
    """Retry brief Windows sharing locks; never relax file permissions.

    Both resolved paths must stay inside the caller's explicit artifact root.
    Temp files and downloaded assets are sometimes held briefly by scanners.
    """
    resolved_root = root.resolve()
    resolved_source = source.resolve()
    if (
        destination.parent.resolve() != resolved_root
        or not resolved_source.is_relative_to(resolved_root)
        or resolved_source == resolved_root
    ):
        raise ValueError("Staged install paths must stay inside their artifact root")
    for attempt in range(5):
        if destination.exists():
            raise FileExistsError(f"Artifact already exists: {destination}")
        try:
            source.rename(destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32} or attempt == 4:
                raise
            time.sleep(0.1 * 2**attempt)

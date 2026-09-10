"""
scan_io.py -- save a finished scan, reopen it later without recomputing.

Every VQE point in these experiments costs real time (seconds for H2, minutes
per point for H2O in a larger active space). Once a scan is done there is no
reason to ever pay that again just to look at the curve, so each script saves
its results and can reload them straight into the interactive viewer.

Two files are written side by side:

  <name>.json   everything needed to rebuild the plot exactly -- the energies
                *and* the metadata describing what was scanned (which
                coordinate, what was held fixed, which active space). This is
                what ``--load`` reads.
  <name>.csv    the same energies as a flat table, for pulling into a report,
                a spreadsheet, or your own plotting code. Nothing reads this
                back; it exists for you.

Shared by the per-molecule scripts under experiment/ so that both write the
same format -- if the two ever drifted apart, a file saved by one would
silently fail to load in the other.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

FORMAT_VERSION = 1


def display_path(path: Path) -> str:
    """Format a path for printing, relative to where the user actually is.

    ``experiment/h2/h2_scan.json`` is easier to read -- and to copy into the
    next command -- than the full absolute path. Falls back to absolute if
    the target is somewhere far outside the current directory, where a
    relative path would be a long chain of ``../..`` and less clear.
    """
    resolved = Path(path).resolve()
    try:
        relative = Path(os.path.relpath(resolved, Path.cwd()))
    except ValueError:  # different drive on Windows -- no relative path exists
        return str(resolved)
    return str(resolved) if str(relative).startswith(os.path.join("..", "..")) else str(relative)


def _jsonable(value: Any) -> Any:
    """numpy scalars/arrays -> plain Python, so json.dump doesn't choke."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def resolve_target(user_path: str | None, default_dir: Path, default_stem: str) -> Path:
    """Decide where a scan should be written.

    A bare name (``--save fine_scan``) lands next to the script, alongside
    where the default save would have gone -- that is almost always what
    someone means, and it avoids scattering result files into whatever
    directory they happened to launch from. Anything with a directory part
    (``--save out/fine``, ``../fine``, ``/tmp/fine``) is left alone and
    treated as an ordinary path, so explicit locations still work.
    """
    if user_path is None:
        return default_dir / default_stem
    candidate = Path(user_path)
    if candidate.parent == Path("."):
        return default_dir / candidate.name
    return candidate


def save_scan(path: Path, metadata: Dict[str, Any], rows: List[Dict[str, Any]]) -> Tuple[Path, Path]:
    """Write ``<path>.json`` (full, reloadable) and ``<path>.csv`` (flat table).

    Returns the two paths actually written.
    """
    path = Path(path)
    json_path = path.with_suffix(".json")
    csv_path = path.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    clean_rows = [{k: _jsonable(v) for k, v in row.items()} for row in rows]
    payload = {
        "format_version": FORMAT_VERSION,
        "metadata": {k: _jsonable(v) for k, v in metadata.items()},
        "rows": clean_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    # The CSV deliberately drops list-valued columns (e.g. the raw optimized
    # parameters) -- they'd blow out the table and aren't useful in a report.
    scalar_keys = [k for k in clean_rows[0] if not isinstance(clean_rows[0][k], list)] if clean_rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean_rows)

    return json_path, csv_path


def load_scan(path: Path, default_dir: Path | None = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read back a scan saved by :func:`save_scan`. Returns ``(metadata, rows)``.

    Looks for the file where you pointed (relative to the current directory,
    or absolute), and failing that next to the script -- the mirror of how
    :func:`resolve_target` saves. That way ``--load h2_scan.json`` works
    whether you run from the project root or from inside the molecule's own
    folder, instead of only from whichever one happens to match.
    """
    path = Path(path)
    if path.suffix != ".json":
        path = path.with_suffix(".json")

    tried = [path]
    if not path.exists() and default_dir is not None and path.parent == Path("."):
        beside_script = Path(default_dir) / path.name
        tried.append(beside_script)
        if beside_script.exists():
            path = beside_script

    if not path.exists():
        locations = "\n  ".join(display_path(p) for p in tried)
        raise FileNotFoundError(f"no saved scan found. Looked in:\n  {locations}")

    payload = json.loads(path.read_text())
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path} was written in format version {version}, but this code reads "
            f"version {FORMAT_VERSION}."
        )
    return payload["metadata"], payload["rows"]

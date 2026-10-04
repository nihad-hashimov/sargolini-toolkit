"""Shared paths, CLI flags and output helpers.

The raw recordings are never part of this repository.  Point the scripts at
them with ``--data-dir`` or the ``SARGOLINI_DATA`` environment variable (the
folder holding the ``*.mat`` files; see ``scripts/download_data.py``).

Outputs go under ``--out-dir`` (default ``results/``), created only when
something is saved:

    figures/   figures
    tables/    per-cell tables and shuffled nulls (not committed; see README)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # scripts save figures; never pop a window

import numpy as np

OUT_DIR = Path("results")

CMAP = "viridis"
RED, BLUE, GREEN, GREY = "#c0392b", "#1f6f8b", "#2f9e44", "#7f7f7f"


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-dir", type=Path,
                   default=os.environ.get("SARGOLINI_DATA"),
                   help="directory holding the *.mat recordings "
                        "(default: $SARGOLINI_DATA)")
    p.add_argument("--out-dir", type=Path, default=Path("results"),
                   help="directory for figures/ and tables/")
    return p


def parse_args(p: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse the command line, check the data directory, set ``OUT_DIR``."""
    global OUT_DIR
    args = p.parse_args()
    if args.data_dir is None or not Path(args.data_dir).is_dir():
        p.error("no data directory: pass --data-dir or set SARGOLINI_DATA")
    OUT_DIR = args.out_dir
    return args


def save(fig, name: str) -> Path:
    path = OUT_DIR / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  saved {path.as_posix()}")
    return path


def table_path(name: str) -> Path:
    path = OUT_DIR / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_table(columns: dict, name: str) -> Path:
    """Write a dict of equal-length 1-D arrays as CSV into ``OUT_DIR/tables``.

    NaN is written as an **empty cell**, not the text "nan" -- otherwise Excel
    types the whole column as text and every later formula silently fails.
    Non-numeric columns are written verbatim, so string identifiers (session,
    cell) can sit alongside numbers.
    """
    keys = list(columns)
    cols = []
    for k in keys:
        c = np.asarray(columns[k])
        cols.append(c if c.dtype.kind in "USO" else np.asarray(c, float).ravel())
    n = len(cols[0])
    if any(len(c) != n for c in cols):
        raise ValueError("all columns must be the same length")

    def cell(c, i):
        v = c[i]
        if c.dtype.kind in "USO":
            s = str(v)
            return f'"{s}"' if ("," in s or '"' in s) else s
        return "" if not np.isfinite(v) else f"{v:.6g}"

    path = table_path(name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for i in range(n):
            fh.write(",".join(cell(c, i) for c in cols) + "\n")
    print(f"  saved {path.as_posix()}  ({n} rows x {len(keys)} cols)")
    return path

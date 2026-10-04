"""Download the Sargolini et al. 2006 recordings from their archive.

Dataset: "Grid cell data Sargolini et al 2006", NIRD Research Data Archive,
https://doi.org/10.11582/2014.00003

The archive's landing page (reached through the DOI) offers a table of contents
listing every file with its download URL and MD5 checksum.  Save that table
locally, then:

    python scripts/download_data.py --toc table_of_contents.csv --dest data/sargolini

The table is pipe-separated with a header row:

    http_url | s3_url | filename | format | size (bytes) | modified_date | fixity (md5)

Files already present with the right checksum are skipped, so an interrupted
download can be resumed by running the same command again.  Every file is
checked against its MD5 after download; a mismatch is reported and the file
removed.

Read the dataset's licence and citation requirements on the landing page before
using or redistributing the data.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path


def read_toc(path: Path) -> list[dict]:
    """Rows of the archive's table of contents as dicts keyed by header."""
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
             if ln.strip()]
    header = [h.strip() for h in lines[0].split("|")]
    rows = []
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split("|")]
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    return rows


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, retries: int = 5) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
            return
        except OSError as e:
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                raise
            print(f"    retry {attempt + 1} after error: {e}", flush=True)
            time.sleep(2 * (attempt + 1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--toc", type=Path, required=True,
                   help="the archive's table of contents (pipe-separated)")
    p.add_argument("--dest", type=Path, required=True,
                   help="directory to download the .mat files into")
    args = p.parse_args()

    rows = read_toc(args.toc)
    if not rows:
        p.error(f"no rows parsed from {args.toc}")
    args.dest.mkdir(parents=True, exist_ok=True)

    bad = 0
    for i, row in enumerate(rows, 1):
        name, url = row["filename"], row["http_url"]
        want = row.get("fixity (md5)", "").lower()
        dest = args.dest / name
        if dest.exists() and want and md5(dest) == want:
            continue
        print(f"[{i}/{len(rows)}] {name}", flush=True)
        try:
            fetch(url, dest)
        except OSError as e:
            print(f"    FAILED: {e}")
            bad += 1
            continue
        if want and md5(dest) != want:
            print("    checksum mismatch; removed")
            dest.unlink()
            bad += 1

    print(f"done: {len(rows) - bad} of {len(rows)} files present and verified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

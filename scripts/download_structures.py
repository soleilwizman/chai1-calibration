"""Download experimental ground-truth structures for the candidate targets.

Stage 2 of the pipeline. Given the curated candidate list (entity IDs such as
``10AF_1``), download the corresponding experimental mmCIF file for each unique
PDB entry from RCSB into ``data/raw/cif/``.

These files are the *ground truth* against which Chai-1 predictions are scored
(per-residue lDDT). The directory is gitignored; this script rebuilds it.

Examples
--------
Download a small sample to smoke-test connectivity::

    python scripts/download_structures.py --max 5

Download everything (skips files already present)::

    python scripts/download_structures.py
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import requests
from requests import Session
from tqdm import tqdm


# BinaryCIF is ~10x smaller than text mmCIF and parses faster in biotite.
RCSB_BCIF_URL = "https://models.rcsb.org/{entry_lower}.bcif.gz"
RCSB_CIF_URL = "https://files.rcsb.org/download/{entry}.cif.gz"


def load_entries(targets_path: Path) -> List[str]:
    """Return the sorted set of unique PDB entry IDs from a candidate list.

    Accepts either the raw list of ``ENTRY_ENTITY`` strings
    (``candidates_raw.json``) or the covariates file (a list of dicts with an
    ``entry_id`` field).
    """
    with targets_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    entries = set()
    for item in data:
        if isinstance(item, str):
            entries.add(item.split("_")[0])
        elif isinstance(item, dict):
            entry = item.get("entry_id") or (item.get("candidate", "").split("_")[0])
            if entry:
                entries.add(entry)
        else:
            raise ValueError(f"Unrecognised target entry: {item!r}")
    return sorted(entries)


def download_one(session: Session, entry: str, dest: Path, timeout: int, retries: int) -> bool:
    """Download one entry, trying BinaryCIF then falling back to text mmCIF.

    Returns True on success (or if already present), False on failure.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return True

    urls = [
        RCSB_BCIF_URL.format(entry_lower=entry.lower()),
        RCSB_CIF_URL.format(entry=entry),
    ]
    for url in urls:
        for attempt in range(retries):
            try:
                resp = session.get(url, timeout=timeout)
                if resp.status_code == 404:
                    break  # try next URL format; retrying a 404 is pointless
                resp.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(resp.content)
                tmp.rename(dest)
                return True
            except requests.RequestException as exc:
                wait = 2 ** attempt
                print(
                    f"Warning: {entry} attempt {attempt + 1}/{retries} failed ({exc}); "
                    f"retrying in {wait}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="data/targets/candidates_covariates.json")
    parser.add_argument("--out-dir", default="data/raw/cif")
    parser.add_argument("--max", type=int, default=0, help="Limit number of entries (0 = all)")
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds between requests")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    targets_path = root / args.targets
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not targets_path.exists():
        raise FileNotFoundError(f"Missing target list: {targets_path}")

    entries = load_entries(targets_path)
    if args.max > 0:
        entries = entries[: args.max]

    session = requests.Session()
    session.headers.update({"User-Agent": "chai1-calibration/1.0"})

    ok, failed = 0, []
    for entry in tqdm(entries, desc="Downloading structures", unit="entry"):
        dest = out_dir / f"{entry}.bcif.gz"
        if download_one(session, entry, dest, args.timeout, args.retries):
            ok += 1
        else:
            failed.append(entry)
        time.sleep(args.sleep)

    print(f"Downloaded {ok}/{len(entries)} entries to {out_dir}")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

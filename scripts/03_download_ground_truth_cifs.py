#!/usr/bin/env python3
"""Download ground-truth mmCIF files for each candidate target from RCSB."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


DEFAULT_INPUT = Path("data/targets/candidates_covariates.json")
DEFAULT_OUTPUT_DIR = Path("data/raw")
BASE_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ground-truth mmCIF files for candidate targets")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to candidate metadata JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for downloaded mmCIF files")
    parser.add_argument("--max", type=int, default=0, help="Limit number of downloads (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between requests in seconds")
    return parser.parse_args()


def load_candidates(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of candidate records in {input_path}")
    return data


def extract_entry_id(item: Dict[str, Any]) -> Optional[str]:
    candidate = item.get("candidate")
    if isinstance(candidate, str) and candidate:
        base = candidate.split("_", 1)[0]
        if base:
            return base

    entry_id = item.get("entry_id")
    if isinstance(entry_id, str) and entry_id:
        return entry_id

    return None


def download_cif(entry_id: str, output_dir: Path, session: requests.Session, delay: float, overwrite: bool = False) -> Tuple[bool, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{entry_id}.cif"
    if output_path.exists() and not overwrite:
        return False, output_path

    url = BASE_URL.format(pdb_id=entry_id)
    response = session.get(url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download {entry_id}: HTTP {response.status_code}")

    output_path.write_bytes(response.content)
    if delay > 0:
        time.sleep(delay)
    return True, output_path


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else Path(__file__).resolve().parents[1] / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else Path(__file__).resolve().parents[1] / args.output_dir

    candidates = load_candidates(input_path)
    downloaded = 0
    failed: List[str] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "chai1-calibration/1.0"})

    for item in candidates[: args.max] if args.max > 0 else candidates:
        entry_id = extract_entry_id(item)
        if not entry_id:
            continue
        try:
            wrote, path = download_cif(entry_id, output_dir, session, args.delay, overwrite=args.overwrite)
        except Exception as exc:  # pragma: no cover - defensive CLI behavior
            failed.append(f"{entry_id}: {exc}")
            continue

        if wrote:
            downloaded += 1
            print(f"Downloaded {entry_id} -> {path}")

    print(f"Completed. Downloaded {downloaded} files to {output_dir}")
    if failed:
        print(f"Failed downloads: {len(failed)}")
        for item in failed[:10]:
            print(item)


if __name__ == "__main__":
    main()

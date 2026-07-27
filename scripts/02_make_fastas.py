#!/usr/bin/env python3
"""Write one FASTA file per target for Chai-lab inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_INPUT = Path("data/targets/candidates_covariates.json")
DEFAULT_OUTPUT_DIR = Path("data/targets/fastas")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one FASTA file per candidate target")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to candidate covariate JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for FASTA outputs")
    parser.add_argument("--max", type=int, default=0, help="Limit number of targets to write (0 = all)")
    return parser.parse_args()


def load_candidates(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of candidate records in {input_path}")
    return data


def iter_sequences(candidates: Iterable[Dict[str, Any]], limit: int = 0) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in candidates:
        candidate = (item.get("candidate") or "").strip()
        if not candidate:
            continue
        seq = item.get("sequence_canonical") or item.get("sequence") or ""
        if not isinstance(seq, str):
            continue
        seq = seq.strip()
        if not seq:
            continue
        pairs.append((candidate, seq))
        if limit and len(pairs) >= limit:
            break
    return pairs


def wrap_sequence(sequence: str, width: int = 80) -> List[str]:
    return [sequence[index : index + width] for index in range(0, len(sequence), width)]


def write_fasta(candidate: str, sequence: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = output_dir / f"{candidate}.fasta"
    header = f">protein|name={candidate}"
    lines = [header, *wrap_sequence(sequence)]
    fasta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fasta_path


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else Path(__file__).resolve().parents[1] / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else Path(__file__).resolve().parents[1] / args.output_dir

    candidates = load_candidates(input_path)
    pairs = iter_sequences(candidates, limit=args.max)

    written_paths: List[Path] = []
    for candidate, sequence in pairs:
        fasta_path = write_fasta(candidate, sequence, output_dir)
        written_paths.append(fasta_path)

    print(f"Wrote {len(written_paths)} FASTA files to {output_dir}")
    if written_paths:
        print("Example:")
        print(written_paths[0].read_text(encoding="utf-8").splitlines()[0])


if __name__ == "__main__":
    main()

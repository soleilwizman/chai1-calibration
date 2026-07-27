#!/usr/bin/env python3
"""Evaluate predicted structures against reference structures.

The script expects an input inventory of tuples of the form
(target, condition, model_idx). Each item should resolve to:
- a predicted CIF/PDB file
- a reference structure file
- a residue-number mapping from step 31
- optionally, a pLDDT score file or per-residue pLDDT embedded in the prediction

It writes:
- results/per_residue.csv
- results/per_structure.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EvaluationCase:
    target: str
    condition: str
    model_idx: int
    predicted_path: Path
    reference_path: Path
    mapping_path: Optional[Path]
    plddt_path: Optional[Path]


@dataclass
class JoinStats:
    total: int
    joined: int
    fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted structure models")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "data" / "targets" / "evaluation_inventory.json",
        help="JSON file containing list of evaluation cases",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for per-residue and per-structure CSV outputs",
    )
    parser.add_argument(
        "--usalign-bin",
        type=Path,
        default=ROOT / "tools" / "USalign",
        help="Path to the USalign executable",
    )
    parser.add_argument(
        "--openstructure-image",
        default="registry.scicore.unibas.ch/schwede/openstructure:latest",
        help="OpenStructure Docker image",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs",
    )
    return parser.parse_args()


def load_inventory(path: Path) -> List[EvaluationCase]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation inventory not found: {path}")

    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("cases") or []

    cases: List[EvaluationCase] = []
    for item in raw:
        target = item["target"]
        condition = item.get("condition", "")
        model_idx = int(item.get("model_idx", 0))
        predicted_path = ROOT / item["predicted_path"]
        reference_path = ROOT / item["reference_path"]
        mapping_path = ROOT / item["mapping_path"] if item.get("mapping_path") else None
        plddt_path = ROOT / item["plddt_path"] if item.get("plddt_path") else None
        cases.append(
            EvaluationCase(
                target=target,
                condition=condition,
                model_idx=model_idx,
                predicted_path=predicted_path,
                reference_path=reference_path,
                mapping_path=mapping_path,
                plddt_path=plddt_path,
            )
        )
    return cases


def read_mapping(mapping_path: Optional[Path]) -> Dict[int, int]:
    if mapping_path is None or not mapping_path.exists():
        raise FileNotFoundError(f"Residue mapping file not found: {mapping_path}")

    raw = json.loads(mapping_path.read_text())
    if isinstance(raw, dict):
        mapping = {int(k): int(v) for k, v in raw.items()}
    else:
        mapping = {int(item["predicted"]): int(item["reference"]) for item in raw}
    return mapping


def extract_residues_from_cif(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    residues: List[Dict[str, Any]] = []

    parser = MMCIFParser(QUIET=True)
    if path.suffix.lower() == ".pdb":
        parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(path.stem, str(path))
    except Exception:
        lines = path.read_text(errors="ignore").splitlines()
        for line in lines:
            if line.startswith("_atom_site."):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] in {"ATOM", "HETATM"}:
                if len(parts) < 6:
                    continue
                residue_number = None
                residue_name = None
                atom_name = None
                for token in parts:
                    if token == "CA":
                        atom_name = token
                        break
                if atom_name is None:
                    for idx, token in enumerate(parts):
                        if token.startswith("CA"):
                            atom_name = token
                            break
                if atom_name is None:
                    continue
                for idx, token in enumerate(parts):
                    if token.isdigit():
                        residue_number = int(token)
                        break
                if residue_number is None:
                    continue
                residue_name = parts[0] if parts[0] not in {"ATOM", "HETATM"} else "UNK"
                residues.append({"chain": "A", "number": residue_number, "name": residue_name})
        if not residues:
            raise ValueError(f"No CA atoms found in {path}")
    else:
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != " ":
                        continue
                    if not any(atom.get_id() == "CA" for atom in residue):
                        continue
                    residues.append(
                        {
                            "chain": chain.id,
                            "number": int(residue.id[1]),
                            "name": residue.get_resname(),
                        }
                    )

    if not residues:
        raise ValueError(f"No CA atoms found in {path}")

    grouped: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for residue in residues:
        grouped[(residue["chain"], residue["number"])] = residue
    return list(grouped.values())


def load_plddt(path: Optional[Path]) -> Dict[int, float]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"pLDDT file not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            if "plddt" in payload:
                payload = payload["plddt"]
            elif "residue_scores" in payload:
                payload = payload["residue_scores"]
        if isinstance(payload, list):
            return {int(item["residue"]): float(item["plddt"]) for item in payload}
        raise ValueError(f"Unsupported pLDDT JSON format: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "residue" in df.columns and "plddt" in df.columns:
            return {int(row["residue"]): float(row["plddt"]) for _, row in df.iterrows()}
        if "residue_index" in df.columns and "plddt" in df.columns:
            return {int(row["residue_index"]): float(row["plddt"]) for _, row in df.iterrows()}

    raise ValueError(f"Unsupported pLDDT file format: {path}")


def run_usalign(predicted_path: Path, reference_path: Path, usalign_bin: Path) -> Dict[str, Any]:
    if not usalign_bin.exists():
        raise FileNotFoundError(f"USalign executable not found: {usalign_bin}")

    cmd = [str(usalign_bin), str(predicted_path), str(reference_path)]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"USalign failed: {completed.stderr or completed.stdout}")

    stdout = completed.stdout
    tm_match = re.search(r"TM-score=\s+([0-9.]+)", stdout)
    rmsd_match = re.search(r"RMSD=\s+([0-9.]+)", stdout)
    aligned_len_match = re.search(r"Aligned length=\s+(\d+)", stdout)
    if not tm_match or not rmsd_match or not aligned_len_match:
        raise RuntimeError(f"Could not parse USalign output: {stdout}")

    return {
        "tm_score": float(tm_match.group(1)),
        "rmsd": float(rmsd_match.group(1)),
        "aligned_length": int(aligned_len_match.group(1)),
        "alignment": stdout,
    }


def run_openstructure(predicted_path: Path, reference_path: Path, image: str) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        model_copy = tmpdir_path / predicted_path.name
        ref_copy = tmpdir_path / reference_path.name
        model_copy.write_bytes(predicted_path.read_bytes())
        ref_copy.write_bytes(reference_path.read_bytes())

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{tmpdir_path}:/workdir",
            "-w",
            "/workdir",
            "--entrypoint",
            "/usr/local/bin/ost",
            image,
            "compare-structures",
            "-m",
            model_copy.name,
            "-r",
            ref_copy.name,
            "--lddt",
            "--local-lddt",
            "-o",
            "out.json",
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"OpenStructure failed: {completed.stderr or completed.stdout}")

        payload = json.loads((tmpdir_path / "out.json").read_text())
    return {
        "global_lddt": float(payload.get("lddt", math.nan)),
        "local_lddt": payload.get("local_lddt", {}),
    }


def join_residues(
    predicted_residues: Sequence[Dict[str, Any]],
    reference_residues: Sequence[Dict[str, Any]],
    mapping: Dict[int, int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if len(predicted_residues) != len(reference_residues):
        raise ValueError(
            "Predicted and reference residue counts differ: "
            f"{len(predicted_residues)} vs {len(reference_residues)}"
        )

    for pred, ref in zip(predicted_residues, reference_residues):
        pred_num = int(pred["number"])
        ref_num = int(ref["number"])
        mapped_ref = mapping.get(pred_num)
        if mapped_ref is None:
            raise ValueError(
                "Residue mapping missing for predicted residue "
                f"{pred_num} -> reference residue {ref_num}"
            )
        if mapped_ref != ref_num:
            raise ValueError(
                "Residue mapping mismatch at predicted residue "
                f"{pred_num} -> reference residue {ref_num} (mapping says {mapped_ref})"
            )

        if pred.get("name") != ref.get("name"):
            raise ValueError(
                "Residue identity mismatch at predicted residue "
                f"{pred_num}: predicted {pred.get('name')} vs reference {ref.get('name')}"
            )

        rows.append(
            {
                "predicted_residue_number": pred_num,
                "reference_residue_number": ref_num,
                "predicted_residue_name": pred.get("name"),
                "reference_residue_name": ref.get("name"),
            }
        )
    return rows


def build_per_residue_rows(
    predicted_residues: Sequence[Dict[str, Any]],
    reference_residues: Sequence[Dict[str, Any]],
    mapping: Dict[int, int],
    plddt_scores: Dict[int, float],
    structure_metrics: Dict[str, Any],
    case: EvaluationCase,
) -> Tuple[List[Dict[str, Any]], JoinStats]:
    joined_rows = join_residues(predicted_residues, reference_residues, mapping)
    joined = len(joined_rows)
    total = max(1, len(joined_rows))
    stats = JoinStats(total=total, joined=joined, fraction=joined / total)

    rows: List[Dict[str, Any]] = []
    for row in joined_rows:
        pred_num = int(row["predicted_residue_number"])
        plddt = plddt_scores.get(pred_num)
        rows.append(
            {
                "target": case.target,
                "condition": case.condition,
                "model_idx": case.model_idx,
                "predicted_residue_number": pred_num,
                "reference_residue_number": row["reference_residue_number"],
                "predicted_residue_name": row["predicted_residue_name"],
                "reference_residue_name": row["reference_residue_name"],
                "pLDDT": plddt,
                "global_lddt": structure_metrics.get("global_lddt"),
                "local_lddt": structure_metrics.get("local_lddt", {}).get(f"A.{pred_num}.", None),
                "tm_score": structure_metrics.get("tm_score"),
                "rmsd": structure_metrics.get("rmsd"),
            }
        )
    return rows, stats


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_case(case: EvaluationCase, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], JoinStats]:
    predicted_residues = extract_residues_from_cif(case.predicted_path)
    reference_residues = extract_residues_from_cif(case.reference_path)
    mapping = read_mapping(case.mapping_path)
    plddt_scores = load_plddt(case.plddt_path)

    structure_metrics = {
        **run_usalign(case.predicted_path, case.reference_path, args.usalign_bin),
        **run_openstructure(case.predicted_path, case.reference_path, args.openstructure_image),
    }

    per_residue_rows, stats = build_per_residue_rows(
        predicted_residues,
        reference_residues,
        mapping,
        plddt_scores,
        structure_metrics,
        case,
    )
    per_structure_row = {
        "target": case.target,
        "condition": case.condition,
        "model_idx": case.model_idx,
        "tm_score": structure_metrics.get("tm_score"),
        "rmsd": structure_metrics.get("rmsd"),
        "global_lddt": structure_metrics.get("global_lddt"),
        "aligned_length": structure_metrics.get("aligned_length"),
        "joined_fraction": stats.fraction,
    }
    return per_residue_rows, [per_structure_row], stats


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    cases = load_inventory(args.inventory)
    per_residue_rows: List[Dict[str, Any]] = []
    per_structure_rows: List[Dict[str, Any]] = []

    for case in cases:
        try:
            per_residue_chunk, per_structure_chunk, stats = evaluate_case(case, args)
        except Exception as exc:
            raise RuntimeError(f"Failed for case {case.target}/{case.condition}/{case.model_idx}: {exc}") from exc

        if stats.fraction < 0.9:
            print(
                f"WARNING: joined fraction {stats.fraction:.3f} for {case.target} {case.condition} {case.model_idx}",
                file=sys.stderr,
            )

        per_residue_rows.extend(per_residue_chunk)
        per_structure_rows.extend(per_structure_chunk)

    write_csv(args.results_dir / "per_residue.csv", per_residue_rows)
    write_csv(args.results_dir / "per_structure.csv", per_structure_rows)
    print(f"Wrote {len(per_residue_rows)} residue rows and {len(per_structure_rows)} structure rows")
    return 0


if __name__ == "__main__":
    main()

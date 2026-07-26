import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests import Session
from tqdm import tqdm


RCSB_POLYMER_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{entry}/{entity_id}"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{entry}"


def parse_candidate(candidate: str) -> Dict[str, str]:
    try:
        entry, entity_id = candidate.split("_")
    except ValueError as exc:
        raise ValueError(f"Invalid candidate identifier '{candidate}', expected ENTRY_ENTITYID") from exc
    return {"candidate": candidate, "entry": entry, "entity_id": entity_id}


def fetch_json(session: Session, url: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
    response = session.get(url, timeout=timeout)
    if not response.ok:
        return None
    return response.json()


def extract_covariates(item: Dict[str, Any], entry_data: Dict[str, Any]) -> Dict[str, Any]:
    entity_poly = item.get("entity_poly", {}) or {}
    rcsb_polymer_entity = item.get("rcsb_polymer_entity", {}) or {}

    covariates = {
        "candidate": item.get("rcsb_id"),
        "entry_id": item.get("rcsb_id", "").split("_")[0],
        "entity_id": item.get("rcsb_id", "").split("_")[-1],
        "nonpolymer_entity_count": None,
        "rcsb_mutation_count": entity_poly.get("rcsb_mutation_count"),
        "rcsb_non_std_monomer_count": entity_poly.get("rcsb_non_std_monomer_count"),
        "entity_length": entity_poly.get("rcsb_sample_sequence_length"),
        "sequence": entity_poly.get("pdbx_seq_one_letter_code"),
        "sequence_canonical": entity_poly.get("pdbx_seq_one_letter_code_can"),
        "rcsb_entity_polymer_type": entity_poly.get("rcsb_entity_polymer_type"),
        "rcsb_artifact_monomer_count": entity_poly.get("rcsb_artifact_monomer_count"),
        "rcsb_multiple_source_flag": rcsb_polymer_entity.get("rcsb_multiple_source_flag"),
        "rcsb_number_of_molecules": rcsb_polymer_entity.get("pdbx_number_of_molecules"),
        "pdbx_mutation": rcsb_polymer_entity.get("pdbx_mutation"),
        "nonpolymer_entity_count": None,
    }

    if entry_data is not None:
        covariates["nonpolymer_entity_count"] = (
            entry_data.get("rcsb_entry_info", {}).get("nonpolymer_entity_count")
            or entry_data.get("rcsb_entry_info", {}).get("nonpolymer_entity_count", None)
        )

    return covariates


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch RCSB metadata covariates for candidate polymer entities.")
    parser.add_argument("--targets", default="data/targets/candidates_raw.json")
    parser.add_argument("--output", default="data/targets/candidates_covariates.json")
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds to sleep between requests")
    parser.add_argument("--max", type=int, default=0, help="Limit number of candidates to process")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = root / args.targets
    output_path = root / args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Missing candidate list: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        candidates = json.load(f)

    parsed = [parse_candidate(x) for x in candidates]
    if args.max > 0:
        parsed = parsed[: args.max]

    entry_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    covariates: List[Dict[str, Any]] = []

    session = requests.Session()
    session.headers.update({"User-Agent": "chai1-calibration/1.0"})

    for item in tqdm(parsed, desc="Fetching candidates", unit="candidates"):
        entry = item["entry"]
        entity_id = item["entity_id"]
        candidate = item["candidate"]

        poly_url = RCSB_POLYMER_ENTITY_URL.format(entry=entry, entity_id=entity_id)
        poly_data = fetch_json(session, poly_url)
        if poly_data is None:
            print(f"Warning: failed to fetch polymer_entity for {candidate}", file=sys.stderr)
            continue

        if entry not in entry_cache:
            entry_url = RCSB_ENTRY_URL.format(entry=entry)
            entry_cache[entry] = fetch_json(session, entry_url)
            if entry_cache[entry] is None:
                print(f"Warning: failed to fetch entry metadata for {entry}", file=sys.stderr)

        item_cov = extract_covariates(poly_data, entry_cache[entry])
        covariates.append(item_cov)
        time.sleep(args.sleep)

    output_path.write_text(json.dumps(covariates, indent=2), encoding="utf-8")
    print(f"Wrote covariates for {len(covariates)} candidates to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

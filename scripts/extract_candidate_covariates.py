import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests import Session
from tqdm import tqdm


RCSB_POLYMER_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{entry}/{entity_id}"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{entry}"
RCSB_SEQUENCE_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
CACHE_DIR_NAME = "data/targets/rcsb_cache"
MIN_REQUEST_INTERVAL = 0.35  # ~3 requests/second


def parse_candidate(candidate: str) -> Dict[str, str]:
    try:
        entry, entity_id = candidate.split("_")
    except ValueError as exc:
        raise ValueError(f"Invalid candidate identifier '{candidate}', expected ENTRY_ENTITYID") from exc
    return {"candidate": candidate, "entry": entry, "entity_id": entity_id}


def make_cache_path(cache_dir: Path, url: str, body: Optional[Dict[str, Any]] = None) -> Path:
    if url.startswith("https://data.rcsb.org/rest/v1/core/polymer_entity/"):
        entry, entity_id = url.rsplit("/", 2)[-2:]
        return cache_dir / "polymer_entity" / f"{entry}_{entity_id}.json"

    if url.startswith("https://data.rcsb.org/rest/v1/core/entry/"):
        entry = url.rsplit("/", 1)[-1]
        return cache_dir / "entry" / f"{entry}.json"

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    if body is not None:
        body_digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return cache_dir / "search" / f"{digest}_{body_digest}.json"

    return cache_dir / "other" / f"{digest}.json"


def load_cache(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def fetch_json(
    session: Session,
    url: str,
    cache_dir: Path,
    last_request_time: float,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Tuple[Optional[Dict[str, Any]], float]:
    cache_path = make_cache_path(cache_dir, url, body)
    cached = load_cache(cache_path)
    if cached is not None:
        return cached, last_request_time

    elapsed = time.time() - last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)

    if method == "POST":
        response = session.post(url, json=body, timeout=timeout)
    else:
        response = session.get(url, timeout=timeout)
    last_request_time = time.time()

    if response.status_code == 204:
        save_cache(cache_path, {})
        return {}, last_request_time

    if not response.ok:
        return None, last_request_time

    try:
        data = response.json()
    except json.JSONDecodeError:
        # Retry once on malformed or empty JSON
        time.sleep(0.1)
        if method == "POST":
            response = session.post(url, json=body, timeout=timeout)
        else:
            response = session.get(url, timeout=timeout)
        last_request_time = time.time()
        if response.status_code == 204:
            save_cache(cache_path, {})
            return {}, last_request_time
        if not response.ok:
            print(f"Warning: failed after retry for {url} ({response.status_code})", file=sys.stderr)
            return None, last_request_time
        try:
            data = response.json()
        except json.JSONDecodeError:
            print(f"Warning: JSON decode failed twice for {url}", file=sys.stderr)
            return None, last_request_time

    save_cache(cache_path, data)
    return data, last_request_time


def parse_sequence_search_identity(response: Dict[str, Any]) -> Tuple[float, int]:
    idents = [
        mc["sequence_identity"]
        for r in response.get("result_set", [])
        for s in r.get("services", [])
        if s.get("service_type") == "sequence"
        for nd in s.get("nodes", [])
        for mc in nd.get("match_context", [])
        if "sequence_identity" in mc
    ]
    total_count = len(idents)
    return max(idents) if idents else 0.0, total_count


def extract_covariates(
    item: Dict[str, Any],
    entry_data: Optional[Dict[str, Any]],
    max_train_identity: float = 0.0,
    total_train_homologs: int = 0,
) -> Dict[str, Any]:
    entity_poly = item.get("entity_poly", {}) or {}
    rcsb_polymer_entity = item.get("rcsb_polymer_entity", {}) or {}

    raw_sequence = entity_poly.get("pdbx_seq_one_letter_code") or ""
    canonical_sequence = entity_poly.get("pdbx_seq_one_letter_code_can") or raw_sequence
    canonical_sequence = canonical_sequence.replace("\n", "") if isinstance(canonical_sequence, str) else canonical_sequence

    entry_info = entry_data.get("rcsb_entry_info", {}) if entry_data else {}
    accession_info = entry_data.get("rcsb_accession_info", {}) if entry_data else {}
    struct = entry_data.get("struct", {}) if entry_data else {}

    covariates = {
        "candidate": item.get("rcsb_id"),
        "entry_id": item.get("rcsb_id", "").split("_")[0],
        "entity_id": item.get("rcsb_id", "").split("_")[-1],
        "nonpolymer_entity_count": entry_info.get("nonpolymer_entity_count"),
        "rcsb_mutation_count": entity_poly.get("rcsb_mutation_count"),
        "rcsb_non_std_monomer_count": entity_poly.get("rcsb_non_std_monomer_count", 0),
        "entity_length": len(canonical_sequence) if canonical_sequence is not None else None,
        "sequence": raw_sequence,
        "sequence_canonical": canonical_sequence,
        "rcsb_entity_polymer_type": entity_poly.get("rcsb_entity_polymer_type"),
        "rcsb_artifact_monomer_count": entity_poly.get("rcsb_artifact_monomer_count"),
        "rcsb_multiple_source_flag": rcsb_polymer_entity.get("rcsb_multiple_source_flag"),
        "rcsb_number_of_molecules": rcsb_polymer_entity.get("pdbx_number_of_molecules"),
        "pdbx_mutation": rcsb_polymer_entity.get("pdbx_mutation"),
        "resolution_combined": entry_info.get("resolution_combined"),
        "release_date": accession_info.get("deposit_date")
        or accession_info.get("initial_release_date")
        or entry_info.get("deposition_date"),
        "title": struct.get("title") or (entry_data.get("title") if entry_data else None),
        "max_train_identity": max_train_identity,
        "total_train_homologs": total_train_homologs,
    }

    return covariates


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch RCSB metadata covariates for candidate polymer entities.")
    parser.add_argument("--targets", default="data/targets/candidates_raw.json")
    parser.add_argument("--output", default="data/targets/candidates_covariates.json")
    parser.add_argument("--max", type=int, default=0, help="Limit number of candidates to process")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = root / args.targets
    output_path = root / args.output
    cache_dir = root / CACHE_DIR_NAME

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
    last_request_time = 0.0

    for item in tqdm(parsed, desc="Fetching candidates", unit="candidates"):
        entry = item["entry"]
        entity_id = item["entity_id"]
        candidate = item["candidate"]

        poly_url = RCSB_POLYMER_ENTITY_URL.format(entry=entry, entity_id=entity_id)
        poly_data, last_request_time = fetch_json(session, poly_url, cache_dir, last_request_time)
        if poly_data is None:
            print(f"Warning: failed to fetch polymer_entity for {candidate}", file=sys.stderr)
            continue

        entity_poly = poly_data.get("entity_poly", {}) or {}
        if entity_poly.get("rcsb_non_std_monomer_count", 0) > 0:
            print(
                f"Skipping {candidate}: rcsb_non_std_monomer_count={entity_poly.get('rcsb_non_std_monomer_count')}",
                file=sys.stderr,
            )
            continue

        if entry not in entry_cache:
            entry_url = RCSB_ENTRY_URL.format(entry=entry)
            entry_cache[entry], last_request_time = fetch_json(session, entry_url, cache_dir, last_request_time)
            if entry_cache[entry] is None:
                print(f"Warning: failed to fetch entry metadata for {entry}", file=sys.stderr)

        max_train_identity = 0.0
        if entity_poly.get("pdbx_seq_one_letter_code_can") or entity_poly.get("pdbx_seq_one_letter_code"):
            sequence_value = (entity_poly.get("pdbx_seq_one_letter_code_can") or entity_poly.get("pdbx_seq_one_letter_code") or "").replace("\n", "")
            search_body = {
                "query": {
                    "type": "group",
                    "logical_operator": "and",
                    "nodes": [
                        {
                            "type": "terminal",
                            "service": "sequence",
                            "parameters": {
                                "evalue_cutoff": 1,
                                "identity_cutoff": 0.0,
                                "sequence_type": "protein",
                                "value": sequence_value,
                            },
                        },
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "rcsb_accession_info.initial_release_date",
                                "operator": "less",
                                "value": "2021-01-12",
                            },
                        },
                    ],
                },
                "return_type": "polymer_entity",
                "request_options": {
                    "paginate": {"start": 0, "rows": 10},
                    "results_content_type": ["experimental"],
                    "results_verbosity": "verbose",
                },
            }
            search_response, last_request_time = fetch_json(
                session,
                RCSB_SEQUENCE_SEARCH_URL,
                cache_dir,
                last_request_time,
                method="POST",
                body=search_body,
            )
            if search_response is not None:
                max_train_identity, total_train_homologs = parse_sequence_search_identity(search_response)
            else:
                print(f"Warning: failed sequence search for {candidate}", file=sys.stderr)
                total_train_homologs = 0

        item_cov = extract_covariates(
            poly_data,
            entry_cache[entry],
            max_train_identity=max_train_identity,
            total_train_homologs=total_train_homologs,
        )
        covariates.append(item_cov)

    output_path.write_text(json.dumps(covariates, indent=2), encoding="utf-8")
    print(f"Wrote covariates for {len(covariates)} candidates to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

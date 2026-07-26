"""Compute each target's maximum sequence identity to any pre-cutoff PDB entry.

Supports hypothesis 3 ("targets with low maximum sequence identity to any
pre-cutoff PDB entry are both less accurate and worse-calibrated"). This is the
data-leakage / novelty covariate: how close is each 2024-2026 target to something
the model could plausibly have seen during pretraining?

For each target sequence it runs an RCSB sequence search restricted to entries
released *before* ``--cutoff`` and records the best-hit identity. A value near 1.0
means a near-identical structure existed pre-cutoff (little novelty); a low value
means the target is genuinely novel.

The search uses ``search.rcsb.org`` (same service that built the candidate set),
so it needs RCSB egress. ``parse_max_identity`` is separated out and unit-tested
so the response handling is verifiable without the network.

Examples
--------
    python scripts/extract_training_identity.py \
        --targets data/targets/candidates_covariates.json \
        --out data/targets/training_identity.json
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def build_query(sequence: str, cutoff: str, rows: int = 25,
                identity_cutoff: float = 0.0, evalue_cutoff: float = 1.0) -> Dict[str, Any]:
    """RCSB query: sequence search AND released strictly before ``cutoff``."""
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "sequence", "parameters": {
                    "evalue_cutoff": evalue_cutoff,
                    "identity_cutoff": identity_cutoff,
                    "sequence_type": "protein",
                    "value": sequence,
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date",
                    "operator": "less",
                    "value": cutoff,
                }},
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "scoring_strategy": "sequence",
            "return_all_hits": False,
            "paginate": {"start": 0, "rows": rows},
        },
    }


def parse_max_identity(response: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract the best-hit sequence identity from a search response.

    Returns ``{max_identity, best_hit, n_hits}``. ``max_identity`` is None when
    there are no pre-cutoff hits (a maximally novel target).
    """
    if not response or "result_set" not in response:
        return {"max_identity": None, "best_hit": None, "n_hits": 0}

    best_id, best_hit = None, None
    results = response.get("result_set", [])
    for res in results:
        identifier = res.get("identifier")
        for svc in res.get("services", []):
            for node in svc.get("nodes", []):
                for ctx in node.get("match_context", []):
                    sid = ctx.get("sequence_identity")
                    if sid is not None and (best_id is None or sid > best_id):
                        best_id, best_hit = sid, identifier
    return {"max_identity": best_id, "best_hit": best_hit, "n_hits": len(results)}


def query_one(session: requests.Session, sequence: str, cutoff: str,
              timeout: int, retries: int) -> Optional[Dict[str, Any]]:
    payload = build_query(sequence, cutoff)
    for attempt in range(retries):
        try:
            resp = session.post(SEARCH_URL, json=payload, timeout=timeout)
            if resp.status_code == 204:      # RCSB returns 204 for zero hits
                return {"result_set": []}
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"Warning: search attempt {attempt + 1}/{retries} failed ({exc}); "
                  f"retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return None


def load_targets(path: Path) -> List[Dict[str, str]]:
    data = json.load(path.open("r", encoding="utf-8"))
    out = []
    for item in data:
        seq = item.get("sequence_canonical") or item.get("sequence")
        cand = item.get("candidate")
        if cand and seq:
            out.append({"candidate": cand, "sequence": seq.strip().replace("\n", "")})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", default="data/targets/candidates_covariates.json")
    parser.add_argument("--out", default="data/targets/training_identity.json")
    parser.add_argument("--cutoff", default="2024-01-01",
                        help="Entries released before this date count as 'seen'")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    targets = load_targets(root / args.targets)
    if args.max > 0:
        targets = targets[: args.max]

    session = requests.Session()
    session.headers.update({"User-Agent": "chai1-calibration/1.0"})

    rows = []
    for t in targets:
        resp = query_one(session, t["sequence"], args.cutoff, args.timeout, args.retries)
        stats = parse_max_identity(resp)
        rows.append({"candidate": t["candidate"], "cutoff": args.cutoff, **stats})
        time.sleep(args.sleep)

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    n_novel = sum(1 for r in rows if r["max_identity"] is None)
    print(f"Wrote training identity for {len(rows)} targets to {out_path} "
          f"({n_novel} with no pre-{args.cutoff} hit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

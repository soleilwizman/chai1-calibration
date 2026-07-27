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
            # REQUIRED: without "verbose" the API returns only identifier+score,
            # with no match_context, so sequence_identity would be unavailable
            # and every target would silently look maximally novel.
            "results_verbosity": "verbose",
            "paginate": {"start": 0, "rows": rows},
        },
    }


def parse_max_identity(response: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract the best-hit sequence identity from a search response.

    Returns ``{max_identity, best_hit, n_hits, parse_failed}``.

    ``max_identity`` is None either because the target genuinely has no
    pre-cutoff hit, or because the response carried no ``match_context``.
    Those mean opposite things scientifically, so ``parse_failed`` distinguishes
    them: True means hits existed but no identity could be read (a bug or a
    verbosity regression), and the caller should treat the value as missing
    rather than as "maximally novel".
    """
    if not response or "result_set" not in response:
        return {"max_identity": None, "best_hit": None, "n_hits": 0,
                "parse_failed": False}

    results = response.get("result_set", [])
    # total_count is the true number of pre-cutoff hits; result_set is paginated.
    n_hits = int(response.get("total_count", len(results)) or 0)

    best_id, best_hit = None, None
    for res in results:
        identifier = res.get("identifier")
        for svc in res.get("services", []):
            for node in svc.get("nodes", []):
                for ctx in node.get("match_context", []):
                    sid = ctx.get("sequence_identity")
                    if sid is not None and (best_id is None or sid > best_id):
                        best_id, best_hit = sid, identifier

    return {"max_identity": best_id, "best_hit": best_hit, "n_hits": n_hits,
            "parse_failed": bool(results) and best_id is None}


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

    n_scored = sum(1 for r in rows if r["max_identity"] is not None)
    n_novel = sum(1 for r in rows
                  if r["max_identity"] is None and not r["parse_failed"])
    n_broken = sum(1 for r in rows if r["parse_failed"])
    print(f"Wrote training identity for {len(rows)} targets to {out_path} "
          f"({n_scored} with an identity, {n_novel} with no pre-{args.cutoff} hit)")

    if n_broken:
        # Loud failure: silently recording None here would make every target look
        # maximally novel and invalidate the novelty stratification.
        print(f"ERROR: {n_broken}/{len(rows)} responses had hits but no readable "
              f"sequence_identity. Do NOT trust novelty_bin from this run -- check "
              f"that request_options.results_verbosity is 'verbose'.", file=sys.stderr)
        return 1
    if n_scored == 0 and rows:
        print("ERROR: no target got an identity value; the search returned nothing "
              "usable. novelty_bin would be empty.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

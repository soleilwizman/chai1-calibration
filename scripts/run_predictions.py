"""Run Chai-1 structure predictions for the candidate targets.

Stage 3 of the pipeline. For each target this writes a Chai-1 input FASTA to
``predictions/<candidate>/input.fasta``, then (when the ``chai_lab`` package and a
GPU are available) runs the model, writing structures to
``predictions/<candidate>/output/``. Input and output are kept in *separate*
directories because ``run_inference`` requires an empty output directory.

Chai-1 writes its per-atom predicted lDDT into the B-factor column of the output
mmCIF (``predictions/<candidate>/output/pred.model_idx_*.cif``), which
``compute_lddt.py`` later reads as ``plddt``.

GPU note
--------
The prediction step needs a CUDA GPU and the ``chai_lab`` package (plus its model
weights, downloaded on first run). It is intentionally *not* invoked in
environments without a GPU. Use ``--dry-run`` to generate the input FASTAs only
-- that path has no heavy dependencies and is what you run to stage inputs before
handing off to a GPU box.

Examples
--------
Stage FASTA inputs for the first 5 targets (no GPU needed)::

    python scripts/run_predictions.py --dry-run --max 5

Run predictions for all targets (GPU + chai_lab required)::

    python scripts/run_predictions.py
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_targets(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    targets = []
    for item in data:
        seq = item.get("sequence_canonical") or item.get("sequence")
        cand = item.get("candidate")
        if cand and seq:
            targets.append({"candidate": cand, "sequence": seq})
        else:
            print(f"Warning: skipping {item.get('candidate')} (no sequence)", file=sys.stderr)
    return targets


def write_fasta(target: Dict, target_dir: Path) -> Path:
    """Write a single-chain Chai-1 input FASTA and return its path.

    Chai-1 expects headers of the form ``>protein|name=<id>``. The FASTA lives at
    ``<target_dir>/input.fasta``; Chai-1's outputs go in a *separate* subdirectory
    (see :func:`run_chai`) because ``run_inference`` requires an empty output dir.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    fasta = target_dir / "input.fasta"
    seq = target["sequence"].strip().replace("\n", "")
    fasta.write_text(f">protein|name={target['candidate']}\n{seq}\n", encoding="utf-8")
    return fasta


def run_chai(fasta: Path, chai_out_dir: Path) -> None:
    """Invoke Chai-1 on a FASTA. Requires chai_lab + a CUDA GPU.

    ``chai_out_dir`` must be empty (Chai-1 asserts this). Kept import-local so that
    ``--dry-run`` (staging inputs) works without the heavy dependency installed.
    """
    try:
        from chai_lab.chai1 import run_inference  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "chai_lab is not installed. Install it on a GPU host "
            "(pip install chai_lab) or use --dry-run to stage FASTA inputs only."
        ) from exc

    chai_out_dir.mkdir(parents=True, exist_ok=True)
    run_inference(
        fasta_file=fasta,
        output_dir=chai_out_dir,
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        use_msa_server=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", default="data/targets/candidates_covariates.json")
    parser.add_argument("--out-dir", default="predictions")
    parser.add_argument("--max", type=int, default=0, help="Limit number of targets (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only write input FASTAs; do not run the model")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip targets whose output/ subdir already has a .cif")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    targets = load_targets(root / args.targets)
    if args.max > 0:
        targets = targets[: args.max]

    out_root = root / args.out_dir
    staged, predicted = 0, 0
    for t in targets:
        target_dir = out_root / t["candidate"]
        chai_out = target_dir / "output"
        if args.skip_existing and any(chai_out.glob("*.cif")):
            continue
        fasta = write_fasta(t, target_dir)
        staged += 1
        if not args.dry_run:
            run_chai(fasta, chai_out)
            predicted += 1

    mode = "staged FASTA for" if args.dry_run else "predicted"
    print(f"{mode} {staged if args.dry_run else predicted}/{len(targets)} targets "
          f"under {out_root}")
    if args.dry_run:
        print("Dry run: FASTA inputs written. Run without --dry-run on a GPU host "
              "with chai_lab installed to produce structures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

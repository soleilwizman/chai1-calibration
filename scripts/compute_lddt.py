"""Compute per-residue lDDT of a predicted structure vs. its experimental reference.

Stage 4 of the pipeline. This is where the two quantities the whole project is
about get paired up, per residue:

* **lDDT**  -- realized accuracy: local Distance Difference Test of the predicted
  coordinates against the experimental ground truth (0..1, higher = better).
* **pLDDT** -- predicted confidence: Chai-1 writes its per-atom predicted lDDT
  into the B-factor column of the output mmCIF (0..100).

A perfectly calibrated model would have ``pLDDT == 100 * lDDT`` on average within
any bin of pLDDT. The downstream calibration analysis (``calibration.py``)
consumes the tidy table this script emits.

Both structures are reduced to the atoms they have in common -- keyed by
``(chain, res_id, ins_code, atom_name)`` -- so unmodeled reference residues and
any atom-ordering differences are handled. lDDT itself is computed over all
common atoms (its inclusion radius reaches across residues), then aggregated to
one score per residue.

Examples
--------
    python scripts/compute_lddt.py \
        --ref data/raw/cif/10AF.bcif.gz \
        --pred predictions/10AF_1/pred.cif \
        --out data/analysis/10AF_1.csv
"""
import argparse
import gzip
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx


def load_structure(path: Path, model: int = 1) -> struc.AtomArray:
    """Load a single model from a (optionally gzipped) mmCIF / BinaryCIF file.

    Returns protein atoms only, carrying the ``b_factor`` field (Chai-1's pLDDT).
    """
    suffixes = "".join(path.suffixes).lower()
    is_binary = ".bcif" in suffixes
    opener = gzip.open if path.suffix == ".gz" else open

    if is_binary:
        with opener(path, "rb") as fh:
            cif = pdbx.BinaryCIFFile.read(fh)
    else:
        mode = "rt" if path.suffix == ".gz" else "r"
        with opener(path, mode) as fh:
            cif = pdbx.CIFFile.read(fh)

    atoms = pdbx.get_structure(cif, model=model, extra_fields=["b_factor"])
    atoms = atoms[struc.filter_amino_acids(atoms)]
    return atoms


def _atom_keys(atoms: struc.AtomArray) -> np.ndarray:
    """A stable per-atom identity: chain / res_id / insertion code / atom name."""
    ins = atoms.ins_code if "ins_code" in atoms.get_annotation_categories() else \
        np.array([""] * atoms.array_length())
    return np.array(
        [f"{c}|{r}|{i}|{a}" for c, r, i, a in
         zip(atoms.chain_id, atoms.res_id, ins, atoms.atom_name)]
    )


def match_atoms(ref: struc.AtomArray, sub: struc.AtomArray) -> Tuple[struc.AtomArray, struc.AtomArray]:
    """Reduce both structures to their common atoms, in the reference's order."""
    ref_keys = _atom_keys(ref)
    sub_keys = _atom_keys(sub)
    sub_index = {k: idx for idx, k in enumerate(sub_keys)}

    ref_sel, sub_sel = [], []
    for i, k in enumerate(ref_keys):
        j = sub_index.get(k)
        if j is not None:
            ref_sel.append(i)
            sub_sel.append(j)

    if not ref_sel:
        raise ValueError("No common atoms between reference and prediction")
    return ref[ref_sel], sub[sub_sel]


def per_residue_scores(ref: struc.AtomArray, sub: struc.AtomArray) -> pd.DataFrame:
    """Return a tidy per-residue table: res_id, chain, lddt, plddt, n_atoms.

    ``ref`` is the experimental reference, ``sub`` the prediction. They need not
    be pre-matched -- this matches atoms internally.
    """
    ref_m, sub_m = match_atoms(ref, sub)

    # lDDT of the prediction's coordinates against the reference, per residue.
    lddt = struc.lddt(ref_m, sub_m, aggregation="residue")

    # Residue identity + mean pLDDT (B-factor) per residue, in the same order
    # struc.lddt uses for residue aggregation (ascending residue starts).
    res_starts = struc.get_residue_starts(ref_m)
    chain_id = ref_m.chain_id[res_starts]
    res_id = ref_m.res_id[res_starts]
    res_name = ref_m.res_name[res_starts]
    plddt = struc.apply_residue_wise(sub_m, sub_m.b_factor, np.mean)
    n_atoms = struc.apply_residue_wise(ref_m, np.ones(ref_m.array_length()), np.sum)

    df = pd.DataFrame({
        "chain": chain_id,
        "res_id": res_id,
        "res_name": res_name,
        "lddt": np.asarray(lddt, dtype=float),
        "plddt": np.asarray(plddt, dtype=float),
        "n_atoms": np.asarray(n_atoms, dtype=int),
    })
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", required=True, help="Experimental reference (mmCIF/BinaryCIF)")
    parser.add_argument("--pred", required=True, help="Predicted structure (mmCIF, pLDDT in B-factor)")
    parser.add_argument("--out", required=True, help="Output CSV of per-residue scores")
    parser.add_argument("--model", type=int, default=1)
    args = parser.parse_args()

    ref = load_structure(Path(args.ref), model=args.model)
    sub = load_structure(Path(args.pred), model=args.model)
    df = per_residue_scores(ref, sub)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} per-residue scores to {out_path} "
          f"(mean lDDT={df.lddt.mean():.3f}, mean pLDDT={df.plddt.mean():.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compute per-residue lDDT of a predicted structure vs. its experimental reference.

Stage 4 of the pipeline -- where realized accuracy (lDDT) and predicted confidence
(pLDDT) get paired per residue:

* **lDDT**  -- local Distance Difference Test of the predicted coordinates against
  the experimental ground truth (0..1, higher = better).
* **pLDDT** -- Chai-1's per-atom predicted lDDT, written into the B-factor column
  of the output mmCIF (0..100).

A perfectly calibrated model has ``pLDDT == 100 * lDDT`` on average within any
bin of pLDDT. ``calibration.py`` consumes the tidy table this emits.

Two correctness details that matter for the science:

1. **Sequence alignment, not residue-number matching.** The prediction numbers
   residues 1..N from the input FASTA, while the experimental mmCIF may use a
   different numbering (offsets, gaps, expression tags). Residues are therefore
   paired by a global sequence alignment of the two structures, so an N-terminal
   His-tag or a construct that starts at residue 20 does not silently wreck the
   comparison. The fraction of reference residues that align is reported as
   ``coverage`` -- a low value means the pairing is untrustworthy.

2. **Cα vs all-atom lDDT.** AlphaFold-style pLDDT predicts *Cα* lDDT, so the
   default metric here is ``ca`` (Cα-only), matching what the confidence head was
   trained to predict. Pass ``--metric all-atom`` to score every common atom
   instead. Comparing all-atom lDDT to a Cα-lDDT predictor would look like
   miscalibration that is really an apples-to-oranges mismatch.

Examples
--------
    python scripts/compute_lddt.py \
        --ref data/raw/cif/10AF.cif.gz \
        --pred predictions/10AF_1/pred.cif \
        --out data/analysis/10AF_1.csv
"""
import argparse
import gzip
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.sequence as seq
import biotite.sequence.align as align


_SUB_MATRIX = align.SubstitutionMatrix.std_protein_matrix()


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


# Maximum accessible surface area per residue (Tien et al. 2013, theoretical),
# used to turn absolute SASA into a relative (0..1) solvent accessibility.
_MAX_ASA = {
    "ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225,
    "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236,
    "MET": 224, "PHE": 240, "PRO": 159, "SER": 155, "THR": 172, "TRP": 285,
    "TYR": 263, "VAL": 174,
}


def reference_annotations(ref: struc.AtomArray) -> pd.DataFrame:
    """Per-residue disorder proxies computed from the *experimental* structure.

    Supports hypothesis 1 ("pLDDT is overconfident in intrinsically disordered
    regions"). Truly disordered residues are usually *absent* from a crystal
    structure and therefore have no lDDT to compare against, so these are proxies
    for local flexibility among the residues that were modelled:

    * ``ref_bfactor``   -- crystallographic B-factor (higher = more mobile).
    * ``ref_bfactor_z`` -- B-factor z-scored *within this structure*, since raw
      B-factors are not comparable across resolutions and refinement protocols.
      This is the primary disorder proxy.
    * ``rsa``           -- relative solvent accessibility (0..1); disordered
      segments tend to be exposed.
    * ``sse``           -- secondary structure (``a`` helix, ``b`` sheet,
      ``c`` coil); disorder is coil-associated.
    * ``near_chain_gap``-- residue adjacent to a break in residue numbering, i.e.
      flanking an unmodelled (disordered) stretch. This is the most direct
      evidence of disorder available for modelled residues.

    Returned frame is keyed by ``(chain, res_id)`` for joining onto scores.
    """
    starts = struc.get_residue_starts(ref)
    chain_id = ref.chain_id[starts]
    res_id = ref.res_id[starts]
    res_name = ref.res_name[starts]

    bfac = struc.apply_residue_wise(ref, ref.b_factor, np.mean)
    std = float(np.std(bfac))
    bfac_z = (bfac - float(np.mean(bfac))) / std if std > 0 else np.zeros_like(bfac)

    # Relative solvent accessibility from all-atom SASA.
    try:
        atom_sasa = struc.sasa(ref)
        atom_sasa = np.nan_to_num(atom_sasa, nan=0.0)
        res_sasa = struc.apply_residue_wise(ref, atom_sasa, np.sum)
        max_asa = np.array([_MAX_ASA.get(rn, np.nan) for rn in res_name], dtype=float)
        rsa = np.clip(res_sasa / max_asa, 0.0, 1.5)
    except Exception:
        rsa = np.full(len(starts), np.nan)

    # Secondary structure (P-SEA; CA-based).
    try:
        sse = struc.annotate_sse(ref)
        if len(sse) != len(starts):
            sse = np.full(len(starts), "", dtype="U1")
    except Exception:
        sse = np.full(len(starts), "", dtype="U1")

    # Residues flanking a numbering discontinuity = edges of unmodelled regions.
    near_gap = np.zeros(len(starts), dtype=bool)
    for i in range(len(starts) - 1):
        if chain_id[i] == chain_id[i + 1] and (res_id[i + 1] - res_id[i]) > 1:
            near_gap[i] = True
            near_gap[i + 1] = True

    return pd.DataFrame({
        "chain": chain_id,
        "res_id": res_id,
        "ref_bfactor": np.asarray(bfac, dtype=float),
        "ref_bfactor_z": np.asarray(bfac_z, dtype=float),
        "rsa": np.asarray(rsa, dtype=float),
        "sse": np.asarray(sse, dtype=str),
        "near_chain_gap": near_gap,
    })


def _one_letter(res_names: np.ndarray) -> seq.ProteinSequence:
    """Convert a 3-letter residue-name array to a ProteinSequence (unknown -> X)."""
    letters = []
    for name in res_names:
        try:
            letters.append(seq.ProteinSequence.convert_letter_3to1(name))
        except KeyError:
            letters.append("X")
    return seq.ProteinSequence("".join(letters))


def _chain_ids_ordered(atoms: struc.AtomArray) -> list:
    """Unique chain IDs in first-appearance order."""
    seen, out = set(), []
    for c in atoms.chain_id:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _aligned_residue_pairs(ref_c: struc.AtomArray, sub_c: struc.AtomArray):
    """Global-align two single-chain structures; yield aligned (ref_start_idx,
    sub_start_idx) residue-boundary index pairs plus the alignment score."""
    ref_starts = struc.get_residue_starts(ref_c)
    sub_starts = struc.get_residue_starts(sub_c)
    ref_seq = _one_letter(ref_c.res_name[ref_starts])
    sub_seq = _one_letter(sub_c.res_name[sub_starts])
    alignment = align.align_optimal(ref_seq, sub_seq, _SUB_MATRIX,
                                    gap_penalty=(-10, -1), terminal_penalty=False)[0]
    pairs = []
    for r, s in alignment.trace:
        if r != -1 and s != -1:
            pairs.append((ref_starts, r, sub_starts, s))
    return pairs, alignment.score


def _residue_atom_index(atoms: struc.AtomArray, starts: np.ndarray, k: int) -> dict:
    """Map atom_name -> global atom index for the k-th residue of ``atoms``."""
    lo = starts[k]
    hi = starts[k + 1] if k + 1 < len(starts) else atoms.array_length()
    return {atoms.atom_name[i]: i for i in range(lo, hi)}


def match_atoms(ref: struc.AtomArray, sub: struc.AtomArray, metric: str = "ca"
                ) -> Tuple[struc.AtomArray, struc.AtomArray, float, int]:
    """Return reference/subject atom arrays reduced to aligned, name-matched atoms.

    Atoms are ordered by reference residue (contiguous per residue) so downstream
    residue-wise aggregation is well defined. Also returns ``coverage`` (fraction
    of reference residues that aligned *and* contributed atoms) and the reference
    residue count.
    """
    if metric not in ("ca", "all-atom"):
        raise ValueError("metric must be 'ca' or 'all-atom'")

    ref_chains = _chain_ids_ordered(ref)
    sub_chains = _chain_ids_ordered(sub)

    ref_sel, sub_sel = [], []
    total_ref_res = struc.get_residue_count(ref)
    aligned_res = 0

    for rc in ref_chains:
        ref_c = ref[ref.chain_id == rc]
        # Greedily pick the sub chain with the best alignment score to this ref
        # chain (trivial for the single-chain targets in this dataset).
        best = None
        for sc in sub_chains:
            sub_c = sub[sub.chain_id == sc]
            pairs, score = _aligned_residue_pairs(ref_c, sub_c)
            if best is None or score > best[0]:
                best = (score, pairs, sub_c)
        if best is None:
            continue
        _, pairs, sub_c = best

        for ref_starts, r, sub_starts, s in pairs:
            ref_atoms = _residue_atom_index(ref_c, ref_starts, r)
            sub_atoms = _residue_atom_index(sub_c, sub_starts, s)
            if metric == "ca":
                names = ["CA"] if ("CA" in ref_atoms and "CA" in sub_atoms) else []
            else:
                names = [n for n in ref_atoms if n in sub_atoms]
            if not names:
                continue
            aligned_res += 1
            # Global indices back into the original ref/sub arrays.
            ref_c_offset = np.where(ref.chain_id == rc)[0]
            sub_c_offset = np.where(sub.chain_id == sub_c.chain_id[0])[0]
            for n in names:
                ref_sel.append(ref_c_offset[ref_atoms[n]])
                sub_sel.append(sub_c_offset[sub_atoms[n]])

    if not ref_sel:
        raise ValueError("No aligned, name-matched atoms between reference and prediction")

    coverage = aligned_res / total_ref_res if total_ref_res else 0.0
    return ref[ref_sel], sub[sub_sel], coverage, total_ref_res


def per_residue_scores(ref: struc.AtomArray, sub: struc.AtomArray, metric: str = "ca"
                       ) -> pd.DataFrame:
    """Tidy per-residue table: chain, res_id, res_name, lddt, plddt, n_atoms.

    ``ref`` is the experimental reference, ``sub`` the prediction. A ``coverage``
    attribute is attached to the returned frame (``df.attrs['coverage']``).
    """
    # Computed on the FULL reference: SASA and secondary structure need every
    # atom, so this must happen before the Ca-only reduction in match_atoms.
    annotations = reference_annotations(ref)

    ref_m, sub_m, coverage, n_ref_res = match_atoms(ref, sub, metric=metric)

    lddt = struc.lddt(ref_m, sub_m, aggregation="residue")

    res_starts = struc.get_residue_starts(ref_m)
    df = pd.DataFrame({
        "chain": ref_m.chain_id[res_starts],
        "res_id": ref_m.res_id[res_starts],
        "res_name": ref_m.res_name[res_starts],
        "lddt": np.asarray(lddt, dtype=float),
        "plddt": np.asarray(struc.apply_residue_wise(ref_m, sub_m.b_factor, np.mean), dtype=float),
        "n_atoms": np.asarray(struc.apply_residue_wise(ref_m, np.ones(ref_m.array_length()), np.sum), dtype=int),
    })
    # Attach the disorder proxies for the residues that were actually scored.
    df = df.merge(annotations, on=["chain", "res_id"], how="left")

    # Persisted as a column (constant per target) so downstream steps can filter
    # out poorly-aligned targets; attrs alone would be lost on CSV round-trip.
    df["coverage"] = coverage
    df.attrs["coverage"] = coverage
    df.attrs["n_ref_residues"] = n_ref_res
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", required=True, help="Experimental reference (mmCIF/BinaryCIF)")
    parser.add_argument("--pred", required=True, help="Predicted structure (mmCIF, pLDDT in B-factor)")
    parser.add_argument("--out", required=True, help="Output CSV of per-residue scores")
    parser.add_argument("--metric", choices=["ca", "all-atom"], default="ca",
                        help="lDDT flavour: 'ca' (default, matches pLDDT) or 'all-atom'")
    parser.add_argument("--min-coverage", type=float, default=0.8,
                        help="Warn (and flag) if aligned-residue coverage is below this")
    parser.add_argument("--model", type=int, default=1)
    args = parser.parse_args()

    ref = load_structure(Path(args.ref), model=args.model)
    sub = load_structure(Path(args.pred), model=args.model)
    df = per_residue_scores(ref, sub, metric=args.metric)
    coverage = df.attrs["coverage"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    flag = "  [LOW COVERAGE]" if coverage < args.min_coverage else ""
    print(f"Wrote {len(df)} per-residue scores ({args.metric}) to {out_path} "
          f"(mean lDDT={df.lddt.mean():.3f}, mean pLDDT={df.plddt.mean():.1f}, "
          f"coverage={coverage:.2%}){flag}")
    if coverage < args.min_coverage:
        print(f"Warning: only {coverage:.1%} of reference residues aligned to the "
              f"prediction; check sequence/chain correspondence before trusting "
              f"these scores.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

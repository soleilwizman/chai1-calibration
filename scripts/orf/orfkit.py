"""ORF extraction and length-matched control sampling.

Shared library for the ORF age-stratification study. Two jobs live here, and
both are places where a quiet mistake would invalidate every downstream number.

**Finding the ORF.** plm-utils (Borges et al. 2024) takes the longest ORF on each
contig, considering five start codons (ATG, TTG, CTG, GTG, ACG). We reimplement
that rather than shelling out to orfipy so the exact definition is visible and
testable -- in particular whether an ORF must terminate in a stop codon, and
whether the reverse strand is searched. Both are options here because both change
the answer, and the only invariant that actually matters is that *the same
definition is applied to the coding and noncoding classes*. Applying a laxer rule
to one class manufactures the separation the classifier is then credited with
finding.

For annotated transcripts (Ensembl cDNA / ncRNA) the orientation is known, so the
default is forward-strand only. Searching both strands on a transcript set lets a
noncoding transcript contribute the best ORF from either orientation while a
coding transcript is judged on the one that biology actually uses -- an advantage
handed to the negative class.

**Length matching.** ProtiGeno (Tu et al. 2023) sampled noncoding regions "from
the same length distribution as the coding regions per genome" explicitly to stop
ORF length becoming a spurious predictive feature. That control is essential here
and not optional: ORF length is the single strongest classical coding signal
(it is what CPAT leans on), and gene age correlates with length, so an
unmatched comparison cannot separate "the model fails on young genes" from "the
model fails on short sequences".

The sampler therefore reports how well matching actually succeeded
(``ks_statistic``, per-bin shortfalls) instead of silently returning whatever it
could find. When a genome has too few long noncoding ORFs to match its coding
length distribution, matching degrades quietly unless someone is watching -- so
this makes it loud.
"""
from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# Standard genetic code (NCBI translation table 1).
GENETIC_CODE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})

# The five start codons plm-utils passes to orfipy.
DEFAULT_START_CODONS: Tuple[str, ...] = ("ATG", "TTG", "CTG", "GTG", "ACG")

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def translate(seq: str) -> str:
    """Translate a nucleotide sequence in frame 0.

    Codons containing anything outside ACGT (an ``N``, an IUPAC ambiguity code)
    translate to ``X`` rather than raising: real transcriptomes contain them, and
    dropping the whole transcript over one ambiguous base would bias the set
    toward cleanly sequenced -- and therefore better studied, therefore older --
    genes.
    """
    upper = seq.upper()
    return "".join(
        GENETIC_CODE.get(upper[i:i + 3], "X")
        for i in range(0, len(upper) - len(upper) % 3, 3)
    )


@dataclass(frozen=True)
class Orf:
    """A single open reading frame located on a transcript."""

    start: int           # 0-based offset into the searched strand
    end: int             # exclusive; includes the stop codon when present
    strand: str          # "+" or "-"
    frame: int           # 0, 1 or 2
    start_codon: str
    has_stop: bool
    protein: str         # translated, stop codon stripped

    @property
    def aa_length(self) -> int:
        return len(self.protein)


def find_orfs(
    seq: str,
    start_codons: Sequence[str] = DEFAULT_START_CODONS,
    require_stop: bool = True,
    both_strands: bool = False,
    min_aa: int = 1,
) -> List[Orf]:
    """Enumerate ORFs on ``seq``.

    An ORF runs from a start codon to the first in-frame stop. Nested starts in
    the same frame that share a stop are all reported; callers wanting
    plm-utils' behaviour should use :func:`longest_orf`, which picks the longest.

    ``require_stop=True`` discards runs that reach the end of the transcript
    without terminating. This matters more than it looks: 3'-truncated coding
    transcripts and short noncoding transcripts both produce stopless runs, and
    admitting them lets transcript length leak in as ORF length.
    """
    starts = {c.upper() for c in start_codons}
    found: List[Orf] = []

    strands = [("+", seq)] + ([("-", reverse_complement(seq))] if both_strands else [])
    for strand, s in strands:
        upper = s.upper()
        n = len(upper)
        for frame in range(3):
            # Walk the frame once, recording the stop that closes each region so
            # every start in that region can be resolved without a rescan.
            codons = [upper[i:i + 3] for i in range(frame, n - 2, 3)]
            pending: List[int] = []          # indices into `codons` of open starts
            for idx, codon in enumerate(codons):
                if codon in starts:
                    pending.append(idx)
                if codon in STOP_CODONS:
                    for s_idx in pending:
                        found.append(_make_orf(
                            upper, strand, frame, s_idx, idx, has_stop=True))
                    pending = []
            if not require_stop:
                for s_idx in pending:
                    found.append(_make_orf(
                        upper, strand, frame, s_idx, len(codons), has_stop=False))

    return [o for o in found if o.aa_length >= min_aa]


def _make_orf(upper: str, strand: str, frame: int,
              start_idx: int, stop_idx: int, has_stop: bool) -> Orf:
    """Build an Orf from codon indices within a frame.

    ``stop_idx`` is the index of the stop codon (or one past the last codon when
    the frame ran off the end). The protein excludes the stop.
    """
    start = frame + start_idx * 3
    end = frame + (stop_idx + 1) * 3 if has_stop else frame + stop_idx * 3
    coding_end = frame + stop_idx * 3
    return Orf(
        start=start,
        end=end,
        strand=strand,
        frame=frame,
        start_codon=upper[start:start + 3],
        has_stop=has_stop,
        protein=translate(upper[start:coding_end]),
    )


def longest_orf(
    seq: str,
    start_codons: Sequence[str] = DEFAULT_START_CODONS,
    require_stop: bool = True,
    both_strands: bool = False,
    min_aa: int = 1,
) -> Optional[Orf]:
    """The longest ORF on ``seq``, or None. Ties break on earliest start.

    This is the plm-utils unit of analysis: one putative peptide per transcript.
    """
    orfs = find_orfs(seq, start_codons, require_stop, both_strands, min_aa)
    if not orfs:
        return None
    return min(orfs, key=lambda o: (-o.aa_length, o.start))


# --------------------------------------------------------------------- FASTA

def read_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    """Yield (header, sequence) pairs. Handles plain and gzipped files."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:  # type: ignore[operator]
        header: Optional[str] = None
        chunks: List[str] = []
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            elif line:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def write_fasta(path: Path, records: Iterable[Tuple[str, str]], width: int = 60) -> int:
    """Write records, wrapping sequences. Returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w") as handle:
        for header, seq in records:
            handle.write(f">{header}\n")
            for i in range(0, len(seq), width):
                handle.write(seq[i:i + width] + "\n")
            count += 1
    return count


# ------------------------------------------------------------ length matching

@dataclass
class MatchReport:
    """Diagnostics for a length-matched draw.

    ``ks_statistic`` is the two-sample Kolmogorov-Smirnov distance between the
    positive and sampled-negative length distributions. It is reported rather
    than tested: with tens of thousands of sequences any real difference is
    "significant", so the effect size is the informative quantity. Below ~0.05
    the distributions are practically indistinguishable.

    ``shortfall`` records bins where the negative pool could not supply enough
    sequences. A large shortfall at the long end is the failure mode to watch:
    it silently reintroduces the length confound the matching exists to remove.
    """

    requested: int
    sampled: int
    ks_statistic: float
    shortfall: Dict[str, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.sampled == self.requested

    def as_dict(self) -> Dict[str, object]:
        return {
            "requested": self.requested,
            "sampled": self.sampled,
            "ks_statistic": round(self.ks_statistic, 4),
            "complete": self.complete,
            "shortfall": self.shortfall,
        }


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample KS distance, sup|F_a - F_b|. Returns nan if either is empty."""
    a_arr, b_arr = np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))
    if a_arr.size == 0 or b_arr.size == 0:
        return float("nan")
    grid = np.concatenate([a_arr, b_arr])
    cdf_a = np.searchsorted(a_arr, grid, side="right") / a_arr.size
    cdf_b = np.searchsorted(b_arr, grid, side="right") / b_arr.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def length_matched_indices(
    positive_lengths: Sequence[int],
    negative_lengths: Sequence[int],
    bin_width: int = 10,
    seed: int = 0,
) -> Tuple[np.ndarray, MatchReport]:
    """Choose negatives whose length distribution matches the positives'.

    Lengths are binned at ``bin_width`` amino acids; each bin then draws, without
    replacement, as many negatives as there are positives in that bin. This is
    the ProtiGeno control: it equalises the marginal length distribution rather
    than merely equalising the means, so length cannot be read off a sequence to
    predict its class.

    Returns indices into ``negative_lengths`` and a :class:`MatchReport`. Sampling
    is seeded, so a given (input, seed) always yields the same draw.
    """
    rng = np.random.default_rng(seed)
    pos = np.asarray(positive_lengths, dtype=int)
    neg = np.asarray(negative_lengths, dtype=int)

    if pos.size == 0 or neg.size == 0:
        return np.array([], dtype=int), MatchReport(int(pos.size), 0, float("nan"))

    pos_bins = pos // bin_width
    neg_bins = neg // bin_width

    # Group negative indices by bin once, then draw per bin.
    order = np.argsort(neg_bins, kind="stable")
    sorted_bins = neg_bins[order]
    boundaries = np.searchsorted(sorted_bins, np.unique(sorted_bins))
    pools: Dict[int, np.ndarray] = {}
    unique_bins = np.unique(sorted_bins)
    for i, b in enumerate(unique_bins):
        lo = boundaries[i]
        hi = boundaries[i + 1] if i + 1 < len(boundaries) else len(order)
        pools[int(b)] = order[lo:hi]

    chosen: List[np.ndarray] = []
    shortfall: Dict[str, int] = {}
    for b, want in zip(*np.unique(pos_bins, return_counts=True)):
        pool = pools.get(int(b), np.array([], dtype=int))
        take = min(int(want), pool.size)
        if take < int(want):
            lo = int(b) * bin_width
            shortfall[f"{lo}-{lo + bin_width - 1}aa"] = int(want) - take
        if take:
            chosen.append(rng.choice(pool, size=take, replace=False))

    picked = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    picked = np.sort(picked)
    report = MatchReport(
        requested=int(pos.size),
        sampled=int(picked.size),
        ks_statistic=ks_statistic(pos, neg[picked]) if picked.size else float("nan"),
        shortfall=shortfall,
    )
    return picked, report


def match_groups_common_support(
    frame,
    group_col: str,
    match_col: str,
    bin_width: int = 10,
    seed: int = 0,
):
    """Equalise ``match_col`` across every level of ``group_col``.

    Each bin is down-sampled to the *minimum* count across groups, so every group
    ends with an identical histogram and bins where any group is empty drop out
    on their own. Unlike matching one group to another's marginal, this cannot
    come up short: the guarantee is structural rather than best-effort.

    That distinction matters more than it sounds. Matching noncoding ORFs to the
    coding length distribution is impossible at the long end of a full-length
    transcriptome -- a 250-codon run without a stop essentially does not occur by
    chance, so no amount of sampling produces long negatives. ProtiGeno could
    match per genome because it worked only on short genes, where the two classes
    genuinely overlap. Restricting to common support is the honest alternative to
    pretending the tails are comparable, and it makes the restriction explicit in
    the report rather than leaving it as residual imbalance.

    Returns ``(matched_frame, report)``. Kept here rather than in the analysis
    module because both the dataset builder (matching coding vs noncoding) and
    the evaluator (matching age strata) need exactly this operation.
    """
    import pandas as pd  # local: keeps orfkit importable without pandas

    groups = sorted(str(g) for g in frame[group_col].dropna().unique())
    if len(groups) < 2:
        return frame, {"groups": groups, "note": "fewer than two groups; nothing to match"}

    working = frame.dropna(subset=[match_col]).copy()
    working["_bin"] = (working[match_col] // bin_width).astype(int)
    working["_grp"] = working[group_col].astype(str)

    counts = working.groupby(["_grp", "_bin"]).size().unstack("_grp", fill_value=0)
    per_bin_min = counts.min(axis=1)
    usable = per_bin_min[per_bin_min > 0]

    rng = np.random.default_rng(seed)
    kept = []
    for bin_id, take in usable.items():
        for group in groups:
            block = working[(working["_bin"] == bin_id) & (working["_grp"] == group)]
            if len(block) == take:
                kept.append(block)
            else:
                pick = rng.choice(len(block), size=int(take), replace=False)
                kept.append(block.iloc[np.sort(pick)])

    matched = (
        pd.concat(kept).drop(columns=["_bin", "_grp"])
        if kept else working.iloc[0:0].drop(columns=["_bin", "_grp"])
    )

    report: Dict[str, object] = {
        "groups": groups,
        "bin_width": bin_width,
        "bins_available": int(len(counts)),
        "bins_retained": int(len(usable)),
        "n_before": int(len(frame)),
        "n_after": int(len(matched)),
        "retained_fraction": round(len(matched) / len(frame), 4) if len(frame) else 0.0,
        "per_group_n": {
            g: int((matched[group_col].astype(str) == g).sum()) for g in groups
        },
        "complete": bool(len(matched) > 0),
    }
    if len(matched):
        report["common_support"] = [
            int(matched[match_col].min()), int(matched[match_col].max())
        ]
        report["max_pairwise_ks"] = round(max(
            (ks_statistic(
                matched.loc[matched[group_col].astype(str) == a, match_col].to_numpy(),
                matched.loc[matched[group_col].astype(str) == b, match_col].to_numpy())
             for i, a in enumerate(groups) for b in groups[i + 1:]),
            default=0.0,
        ), 4)
    return matched, report

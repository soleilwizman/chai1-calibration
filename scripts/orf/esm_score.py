"""Score putative ORF peptides with ESM-2: embeddings and zero-shot likelihood.

Two scoring modes, deliberately separated, because the comparison between them is
part of the point.

**Embeddings** (``--mode embed``) reproduce the plm-utils / ProtiGeno setup:
mean-pool the final-layer representations and hand them to a supervised head. The
default checkpoint is ``esm2_t6_8M_UR50D`` because that is what plm-utils
hardcodes -- matching it exactly is what makes this a baseline rather than a
different experiment.

**Zero-shot pseudo-log-likelihood** (``--mode pll``) is the axis neither paper
tried. Both read the model's *representations* and train a classifier on top;
neither reads the model's *likelihood* directly. A masked language model already
assigns every sequence a probability, and a real protein should be more probable
under a model trained on real proteins than a spurious translation of noncoding
RNA is. If that holds with no supervised head at all, the classifier those papers
train is partly redundant -- and if it holds on old genes but collapses on young
ones, that is the homology-lookup story showing up in the likelihood itself,
without a trained head to blame it on.

PLL is length-normalised by default. Unnormalised log-likelihood scales with
sequence length, which would smuggle the length confound back in through the
scoring function after the dataset went to such trouble to remove it.

Cost note: exact PLL masks every position in turn, so it is O(L) forward passes
per sequence. At the 8M checkpoint that is cheap; at 650M it is not. ``--stride``
subsamples masked positions for a cheaper unbiased estimate.

**This module has not been executed against real weights.** The sandbox that
produced it has ``huggingface.co`` blocked by egress policy, so the shapes and
the masking logic are argued from the ESM-2 API rather than observed. Run
``--self-test`` on a host with weights before trusting any number it emits.

Examples
--------
    python scripts/orf/esm_score.py --fasta data/orf/hsap.faa \
        --mode embed --out data/orf/hsap.embeddings.npz

    python scripts/orf/esm_score.py --fasta data/orf/hsap.faa \
        --mode pll --out data/orf/hsap.pll.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orfkit import read_fasta  # noqa: E402

DEFAULT_CHECKPOINT = "facebook/esm2_t6_8M_UR50D"  # what plm-utils hardcodes


def load_model(checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cpu"):
    """Load an ESM-2 model and tokenizer, with an actionable error if absent.

    Deliberately imports inside the function: the rest of this repository runs
    CPU-only without torch, and importing it at module scope would make the
    analysis pipeline depend on a GPU-shaped dependency it never uses.
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            f"missing dependency: {exc}. Install with:\n"
            "    pip install torch transformers\n"
            "Note that fetching weights needs access to huggingface.co, which is "
            "blocked by egress policy in the sandbox -- run this stage on the "
            "prediction host (see scripts/orf/README.md)."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    model.eval().to(device)
    return model, tokenizer


def _batches(records: Sequence[Tuple[str, str]], size: int) -> Iterator[Sequence]:
    for i in range(0, len(records), size):
        yield records[i:i + size]


def embed(
    records: Sequence[Tuple[str, str]],
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cpu",
    batch_size: int = 16,
    max_length: int = 1022,
) -> Tuple[List[str], np.ndarray]:
    """Mean-pooled final-layer embeddings, one row per sequence.

    Padding and the BOS/EOS tokens are excluded from the mean. Including them is
    an easy mistake that makes the pooled vector depend on batch composition,
    which would quietly couple every sequence's representation to whatever it
    happened to be batched with.
    """
    import torch

    model, tokenizer = load_model(checkpoint, device)
    ids: List[str] = []
    vectors: List[np.ndarray] = []

    with torch.no_grad():
        for batch in _batches(records, batch_size):
            names = [name for name, _ in batch]
            seqs = [seq[:max_length] for _, seq in batch]
            encoded = tokenizer(seqs, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length + 2)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = model(**encoded, output_hidden_states=True)
            hidden = out.hidden_states[-1]                      # (B, L, D)

            mask = encoded["attention_mask"].clone()
            mask[:, 0] = 0                                      # BOS
            lengths = mask.sum(dim=1, keepdim=True)
            # Drop EOS: it is the last attended position of each row.
            for row, length in enumerate(encoded["attention_mask"].sum(dim=1)):
                mask[row, int(length) - 1] = 0
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)

            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths
            ids.extend(names)
            vectors.append(pooled.cpu().numpy())

    return ids, np.vstack(vectors) if vectors else np.zeros((0, 0))


def pseudo_log_likelihood(
    sequence: str,
    model,
    tokenizer,
    device: str = "cpu",
    stride: int = 1,
    normalise: bool = True,
    max_length: int = 1022,
) -> float:
    """Masked pseudo-log-likelihood of one sequence.

    Each residue is masked in turn and scored by the model's log-probability of
    the true amino acid given the rest. Summing those gives the standard
    pseudo-log-likelihood; dividing by the number of scored positions makes it
    comparable across lengths, which is what ``normalise`` controls.

    ``stride > 1`` scores every n-th position instead of all of them. The
    per-position mean is unbiased under subsampling, so the normalised score
    stays comparable -- the unnormalised sum does not, and is rescaled to the
    full length to keep the two modes on one scale.
    """
    import torch

    sequence = sequence[:max_length]
    if not sequence:
        return float("nan")

    encoded = tokenizer(sequence, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    # Token 0 is BOS and the final token is EOS; residues sit between them.
    positions = list(range(1, input_ids.shape[1] - 1))[::stride]
    if not positions:
        return float("nan")

    total = 0.0
    with torch.no_grad():
        for start in range(0, len(positions), 64):
            chunk = positions[start:start + 64]
            repeated = input_ids.repeat(len(chunk), 1)
            targets = []
            for row, pos in enumerate(chunk):
                targets.append(int(repeated[row, pos].item()))
                repeated[row, pos] = tokenizer.mask_token_id
            logits = model(input_ids=repeated).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            for row, pos in enumerate(chunk):
                total += float(log_probs[row, pos, targets[row]].item())

    if normalise:
        return total / len(positions)
    return total * (len(positions) and (input_ids.shape[1] - 2) / len(positions))


def score_fasta_pll(
    records: Sequence[Tuple[str, str]],
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cpu",
    stride: int = 1,
    normalise: bool = True,
) -> List[Dict[str, object]]:
    model, tokenizer = load_model(checkpoint, device)
    rows: List[Dict[str, object]] = []
    for name, seq in records:
        rows.append({
            "sequence_id": name,
            "aa_length": len(seq),
            "pll": pseudo_log_likelihood(
                seq, model, tokenizer, device=device,
                stride=stride, normalise=normalise),
        })
    return rows


def self_test(checkpoint: str, device: str) -> int:
    """Sanity checks that require real weights. Run this before trusting output.

    A real protein should score higher than a shuffled version of itself: the
    shuffle preserves amino-acid composition exactly, so anything the model
    prefers about the original is about *order*, which is what a language model
    is supposed to have learned. If this fails, the masking indices are wrong.
    """
    model, tokenizer = load_model(checkpoint, device)
    rng = np.random.default_rng(0)
    # Human haemoglobin subunit alpha (P69905), N-terminal 60 residues.
    real = "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKK"
    shuffled = "".join(rng.permutation(list(real)))

    real_pll = pseudo_log_likelihood(real, model, tokenizer, device=device)
    shuffled_pll = pseudo_log_likelihood(shuffled, model, tokenizer, device=device)
    print(f"real     PLL/residue = {real_pll:.4f}")
    print(f"shuffled PLL/residue = {shuffled_pll:.4f}")
    print(f"delta                = {real_pll - shuffled_pll:+.4f}")

    if not real_pll > shuffled_pll:
        print("FAIL: the real sequence did not outscore its own shuffle; "
              "check the mask position indices.", file=sys.stderr)
        return 1
    print("PASS")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fasta", type=Path, help="peptide FASTA to score")
    parser.add_argument("--mode", choices=("embed", "pll"), default="embed")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1,
                        help="score every n-th position (pll mode)")
    parser.add_argument("--no-normalise", dest="normalise", action="store_false",
                        help="report summed PLL instead of per-residue")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true",
                        help="verify masking against a real/shuffled protein")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test(args.checkpoint, args.device)

    if not args.fasta or not args.out:
        parser.error("--fasta and --out are required unless --self-test is given")

    records = list(read_fasta(args.fasta))
    if not records:
        raise SystemExit(f"no sequences in {args.fasta}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "embed":
        ids, matrix = embed(records, args.checkpoint, args.device, args.batch_size)
        np.savez_compressed(args.out, sequence_id=np.array(ids), embedding=matrix)
        print(f"wrote {args.out}  ({matrix.shape[0]} x {matrix.shape[1]})")
    else:
        import pandas as pd
        rows = score_fasta_pll(records, args.checkpoint, args.device,
                               args.stride, args.normalise)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"wrote {args.out}  ({len(rows)} sequences)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

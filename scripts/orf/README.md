# Does a PLM coding/noncoding classifier fail on young genes?

Two papers established that protein language models can call coding versus
noncoding ORFs:

- **plm-utils** — Borges AL, Celebi FM, Cheveralls K, Reiter T (2024),
  [10.57844/arcadia-fa56-ee23](https://doi.org/10.57844/arcadia-fa56-ee23),
  [Arcadia-Science/2024-plm-utils](https://github.com/Arcadia-Science/2024-plm-utils).
  orfipy finds the longest ORF per contig across five start codons
  (ATG, TTG, CTG, GTG, ACG); ESM-2 embeds the translation; a random forest
  classifies. The checkpoint is hardcoded to `esm2_t6_8M_UR50D`. Evaluated as a
  16-species cross-species matrix (Ensembl 111 / EnsemblGenomes 58) on MCC,
  recall and TNR.
- **ProtiGeno** — Tu et al. (2023), ICML WCB,
  [arXiv:2307.10343](https://arxiv.org/abs/2307.10343). ESM-1b embeddings
  (d=1280) into a 7-layer network for short prokaryotic genes. From 145,232
  coding and 3,465,408 noncoding short regions they "sampled the noncoding
  regions from the same length distribution as the coding regions per genome" to
  control for ORF length as a spurious feature.

Both report **average** performance. Neither asks where the errors land. This
asks whether they land on the evolutionarily novel ORFs — the ones a
coding-potential tool exists to find, since anything with detectable homology can
be annotated by homology instead.

## The two traps

**Length.** Young genes are short, and ORF length is the dominant classical
coding signal. A naive age stratification recovers "short ORFs are hard", which
is the stated *motivation* of both papers rather than a critique of them. Every
comparison here runs on length-matched sets.

**Age inference itself.** Phylostratigraphy dates a gene by *failure to detect
homologs*; Moyers & Zhang showed short, fast-evolving genes are systematically
misdated as young for that reason. A BLAST-based age assignment and an
ESM-embedding classifier can fail on the same sequences for the same underlying
cause, which would manufacture the correlation. This is a bound on interpretation,
not something statistics removes. Prefer synteny-based ages (fagin) where
available; `join_gene_age.py --age-method` records which was used.

The sharper hypothesis is that the classifier does not merely fail on young genes
but **succeeds on old ones for the wrong reason**: ESM-2 was pretrained on
UniRef50, where conserved genes are dense and orphans are absent by construction.
`age_stratified_eval.py` tests this by asking whether age still predicts error
once nearest-UniRef50 identity is conditioned on.

## Pipeline

| # | Stage | Script | Needs network |
|---|---|---|---|
| 0 | Fetch transcriptomes | — (see below) | **yes** — ftp.ensembl.org |
| 1 | ORFs + length-matched controls | `build_orf_dataset.py` | no |
| 2 | ESM-2 embeddings / zero-shot PLL | `esm_score.py` | **yes** — huggingface.co |
| 2b | Nearest-UniRef50 identity | MMseqs2/DIAMOND | **yes** — UniRef50 DB |
| 3 | Gene age + covariate join | `join_gene_age.py` | no (needs an age table) |
| 4 | Age-stratified evaluation | `age_stratified_eval.py` | no |

```bash
# stage 1
python scripts/orf/build_orf_dataset.py \
    --coding data/orf/raw/hsap.cdna.fa.gz \
    --noncoding data/orf/raw/hsap.ncrna.fa.gz \
    --species hsap --match common-support \
    --out data/orf/hsap.csv --fasta-out data/orf/hsap.faa

# stage 2 (prediction host)
python scripts/orf/esm_score.py --fasta data/orf/hsap.faa --mode pll \
    --out data/orf/hsap.pll.csv

# stage 3
python scripts/orf/join_gene_age.py --orfs data/orf/hsap.csv \
    --ages data/orf/genorigin_hsap.tsv --uniref data/orf/hsap.uniref50.tsv \
    --out data/orf/hsap.annotated.csv

# stage 4
python scripts/orf/age_stratified_eval.py --table data/orf/hsap.scored.csv \
    --control uniref50_identity --age-numeric age_mya \
    --out data/orf/age_stratified.json
```

### Matching modes

`build_orf_dataset.py --match` chooses how the noncoding class is controlled:

- `none` — reproduces plm-utils. On the synthetic fixture the two classes differ
  in length at **KS 0.725**: length alone is a strong classifier here, which is
  what makes the unmatched setup hard to interpret.
- `sample` — ProtiGeno's control: draw noncoding ORFs matching the coding length
  distribution. **Cannot complete on full-length transcriptomes.** A ~250-codon
  run without a stop essentially does not occur by chance, so no sampling
  produces long negatives; the shortfall is reported per bin. ProtiGeno could
  match per genome because it worked only on short genes, where the classes
  genuinely overlap.
- `common-support` — down-sample both classes per length bin to the smaller.
  Exact by construction (fixture: KS 0.725 → **0.043**) at the cost of the long
  tail (fixture: 9.8% retained). Retention and the surviving length range are
  always reported.

## Validation

`tests/test_orf_age.py` simulates four worlds where the error mechanism is known
by construction and asserts the analysis recovers it:

| scenario | unmatched (young→old recall) | matched | required verdict |
|---|---|---|---|
| length only | 0.451 → 0.754 | 0.492 → 0.500 | spurious effect erased |
| age beyond length | 0.105 → 0.879 | 0.117 → 0.698 | real effect survives |

and for the decomposition: representation-only gives pooled AUC 0.209 →
conditional 0.454 (absorbed); age-beyond-representation gives conditional 0.238
(not explained away). Length equalisation is near-exact at scale
(max pairwise KS 0.0074).

The false-positive guards (length-only, representation-only) and the power checks
(the other two) both matter: an instrument that passes only the guards is one
that can never detect anything.

## What has not been run

The analysis code is tested; the **study has not been executed**. Stages 0, 2 and
2b need hosts blocked by egress policy in the sandbox where this was written:

- `huggingface.co` — ESM-2 weights
- `ftp.ensembl.org` — transcriptomes
- `rest.uniprot.org` — UniRef

All three return 403 on CONNECT (organization policy, not a transient failure).
Run those stages on the prediction host, as with the Chai-1 GPU stages.

`esm_score.py` in particular has **never been executed against real weights** —
its tensor shapes and masking indices are argued from the ESM-2 API, not
observed. Run `python scripts/orf/esm_score.py --self-test` first: it checks that
a real protein outscores a shuffled version of itself, which fails loudly if the
mask positions are off by one.

## Data sources

- Transcriptomes: Ensembl `cdna` and `ncrna` FASTAs, matching the releases
  plm-utils used (111 vertebrates / 58 EnsemblGenomes) so the baseline is
  comparable.
- Gene ages: [GenOrigin](http://genorigin.chenzxlab.cn/) (9.1M genes, 565
  species) is the easiest join; GenTree for human/Drosophila; `phylostratr` to
  compute them directly. Human and Drosophila have the best coverage.
- Pretraining representation: MMseqs2 or DIAMOND against UniRef50 — the same
  release ESM-2 was pretrained on, or the result measures the wrong thing.

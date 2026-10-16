# Artemis

**TransE knowledge-graph embeddings in Nim.**

Artemis trains [TransE](https://papers.nips.cc/paper/5023-translating-embeddings-for-modeling-multi-relational-data)
embeddings (Bordes et al., NeurIPS 2013) on a table of `head relation tail`
triples, evaluates them with standard link-prediction metrics, and exports the
learned vectors — as a small native binary, with no Python anywhere.

Every true fact `(h, r, t)` is trained to satisfy `h + r ≈ t`; facts are pushed
against corrupted negatives with a margin-ranking loss and SGD.

Companion documents: [`DevLog.md`](DevLog.md) (build chronicle) and
[`Decisions.md`](Decisions.md) (decision records). Requirements and the
original plan live in `PRD.md` and `CODING_PLAN.md`.

---

## Install

Requires Nim ≥ 2.0 and nimble.

```sh
git clone https://github.com/unplugandplay/art-mis
cd art-mis && git checkout toy
nimble deps            # installs arraymancer
nimble build -d:release
```

## Quickstart

```sh
# train on the bundled 16-triple family graph
./artemis train --data data/toy_family.tsv --dim 10 --epochs 200 --lr 0.05

# evaluate (self-eval here; see FB15k-237 section for the real numbers)
./artemis eval --model artemis_checkpoint.bin --data data/toy_family.tsv

# semantic sanity check: alice's nearest neighbour is her spouse
./artemis nearest --model artemis_checkpoint.bin --entity alice

# export plain-text embeddings + metadata
./artemis export --model artemis_checkpoint.bin --out out/
```

## Data format

Tab-separated (or `--delimiter comma|space|CH`), one triple per line, in any
entity/relation vocabulary:

```
alice	spouseOf	david
alice	parentOf	bob
```

Malformed lines are skipped and counted. Ids are assigned in first-appearance
order, so runs are reproducible. For proper evaluation, train on one file and
evaluate on held-out triples — with `--valid` and `--test` supplied to `train`,
filtered metrics use all splits as the known-truth set; standalone `eval` needs
`--truth train.tsv,valid.tsv` to do the same.

## Commands

### `artemis train --data train.tsv [options]`

| Option | Default | Meaning |
|--------|---------|---------|
| `--dim N` | 50 | embedding dimension |
| `--margin F` | 1.0 | margin-ranking margin |
| `--lr F` | 0.01 | SGD learning rate |
| `--epochs N` | 1000 | max epochs |
| `--batch N` | 100 | batch size (shuffled each epoch) |
| `--neg N` | 1 | negatives per positive |
| `--norm l1\|l2` | l1 | dissimilarity norm ‖h+r−t‖ₚ |
| `--bern` / `--no-bern` | on | Bernoulli negative sampling |
| `--seed N` | 42 | RNG seed (deterministic runs) |
| `--eval-every N` | 0 | validation MRR + early stopping every N epochs |
| `--ckpt-every N` | 0 | write `out/ckpt_eN.bin` every N epochs |
| `--min-count N` | 1 | drop entities/relations seen fewer than N times |
| `--threads N` | 1 | evaluation workers (0 = one per core; training stays serial) |
| `--valid` / `--test` | — | extra splits (truth set + metrics) |
| `--out DIR` | . | checkpoints, `embeddings.tsv`, `vocab.json` |
| `--quiet` | off | suppress progress lines |

### Other commands

```sh
artemis eval    --model checkpoint.bin --data test.tsv [--threads N] \
                [--truth train.tsv,valid.tsv]
artemis export  --model checkpoint.bin --out dir
artemis nearest --model checkpoint.bin --entity NAME [--k 10] [--metric cosine|l2]
```

`eval` filters ranks against the file it is evaluating; pass `--truth` with the
other splits to get the standard train∪valid∪test protocol — on the FB15k-237
reference checkpoint that is the difference between filtered MRR 0.157
(test-only truth) and the 0.207 reported below.

## Library usage

```nim
import artemis

let hp = HyperParams(dim: 32, epochs: 200)
let (data, entities, relations, report) = loadDataset("train.tsv", hp)
var rng = initRand(42)
var model = initModel(entities, relations, hp, rng)

let truth = buildTruthSet(data.train, data.valid, data.test)
let sampler = initSampler(data.train, relations.len, entities.len, true, truth)
discard train(model, data.train, sampler, hp, data.valid, truth)

let metrics = evaluate(model, data.test, truth, hp.norm)
saveCheckpoint(model, hp, "ckpt.bin")
```

## Evaluation protocol

For each held-out triple, the true head is ranked against all entities (tail
fixed), then the true tail against all entities (head fixed): two ranks per
triple. Reported: Mean Rank, MRR, Hits@1/3/10 — *raw* and *filtered* (candidates
that form any known-true triple are excluded). Rank `= 1 + #better + 0.5·#ties`
(average tie handling; ties are rare with float32 scores).

Implementation: the known-true tails of each `(h, r)` and heads of each `(r, t)`
are indexed once and flattened into per-triple candidate lists, so the filtered
rank is `rawRank − #truth candidates that score better` and the rank loop stays
pure arithmetic (no hash probe per candidate). The pooled triples are then
evaluated in contiguous chunks on `--threads N` workers (default 1; `0` = one per
core). Training itself is always single-threaded, so a seed still reproduces
byte-identical embeddings; a fixed thread count reproduces the metrics (thread
count changes only the order of float64 additions — see `Decisions.md`, D16/D17).

## Results

### Bundled toy graph (16 triples, 8 entities, 3 relations — self-eval)

`--dim 10 --epochs 200 --lr 0.05`

| | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|
| raw | 1.66 | 0.682 | 0.375 | 1.000 | 1.000 |
| filtered | 1.47 | 0.766 | 0.531 | 1.000 | 1.000 |

Optimistic (evaluated on training triples), but a useful smoke signal: during
training the 8-entity ranking improves from MRR ≈ 0.29 to ≈ 0.68, and the nearest
neighbour of `alice` becomes her spouse `david` (cosine 0.943).

### FB15k-237 (14,505 entities · 237 relations · 272,115 train triples)

```sh
scripts/get_data.sh
./artemis train --data data/fb15k237/train.txt --valid data/fb15k237/valid.txt \
  --test data/fb15k237/test.txt --dim 100 --lr 0.01 --epochs 200 --eval-every 20
```

Reference run (`--dim 100 --lr 0.01 --epochs 200 --bern`, ~15 min wall on a
laptop, single thread — training 0.15 s/epoch, validation pass ~80 s):

| test (n = 40,876 ranks) | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|
| raw | 648.5 | 0.155 | 0.098 | 0.160 | 0.268 |
| filtered | 523.4 | **0.207** | 0.140 | 0.221 | **0.343** |

Validation MRR rose monotonically across all ten evaluations (0.171 → 0.208),
so early stopping never triggered and the run ended under-trained. Published,
tuned TransE baselines on FB15k-237 report filtered MRR ≈ 0.29 / Hits@10 ≈
0.46 with larger budgets (500–1000 epochs, sometimes dim 200); this result
lands in a sane band for the budget spent, and the gap is attributable to
training time, not correctness (see `DevLog.md` for the trajectory).

The ten validation passes dominate that wall clock; `--threads 0` runs them on
every core (65 s → 10 s per pass on 8 cores here) and leaves the numbers above
unchanged to the last printed digit. Re-running the identical command with
`--threads 0` finished in **2m01s** and reproduced all ten validation MRRs and
every test metric exactly; evaluating the same checkpoint standalone with
`--truth train.tsv,valid.tsv` reproduces filtered MRR 0.207 / Hits@10 0.343.

Training budget is the remaining gap. The same command with `--epochs 500
--eval-every 25` (best-validation weights restored at epoch 475, 4m14s wall
with `--threads 0`) gives:

| test (n = 40,876 ranks) | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|
| filtered, 200 epochs | 523.4 | 0.207 | 0.140 | 0.221 | 0.343 |
| filtered, 500 epochs | 463.6 | 0.220 | 0.146 | 0.241 | 0.364 |

2.5× the budget buys +0.013 MRR / +0.021 Hits@10, and the validation curve
flattens into ±0.005 noise after ~epoch 300 — so the rest of the distance to
published ≈0.29 needs an lr schedule or dim 200, not just more epochs at a
constant lr.

### Performance (release build, Apple Silicon)

Synthetic scale — 15,000 entities × 200 relations × 500,000 random triples,
dim 50 (`nimble bench`), single thread:

| Metric | Value |
|---|---|
| training throughput | ~850,000 triples/s |
| epoch time (500k triples) | 0.59 s |
| peak RSS | 273 MB |

The per-triple hot loop performs no allocations (preallocated scratch buffers,
raw row-major access), so epoch time scales linearly with triples × dim.

Link-prediction evaluation — FB15k-237 valid (17,526 triples ranked against
14,505 entities, dim 100, L1), one full raw + filtered pass:

| Workers | Wall | Notes |
|---|---|---|
| 1 | 42.3 s | pre-index implementation was 65 s on the same machine |
| 2 | 21.6 s | |
| 4 | 13.2 s | measured with one core busy training, so 8 is conservative |
| 8 | 10.1 s | metrics identical at every thread count |

## Design notes

Deliberate deviations from the original blueprint — rationale in `Decisions.md`:

1. **Manual closed-form gradients, not Arraymancer autograd.** For `x = h+r−t`,
   `‖x‖₂` has gradient `x/‖x‖₂` and `‖x‖₁` has gradient `sign(x)`. The
   blueprint's `discard` placeholder is replaced by this math applied to the
   3–4 affected embedding rows directly. Validated by a finite-difference test
   (`tests/t_gradients.nim`) to 1e-2 on float32.
2. **Unit-norm constraint on entities after every update** (paper requirement,
   absent from the blueprint): entity rows are re-projected onto the unit L2
   sphere; relation rows are unconstrained. An invariant test enforces it.
3. **Fused loops instead of Arraymancer broadcasting** for per-triple scoring:
   identical complexity, zero temporaries, no allocation in the hot path.
4. **Sequential SGD updates on shared rows** (`h` in both pos/neg, `r` in both)
   handle head- vs tail-corruption uniformly by applying the two update terms
   in sequence; the test suite checks both sides numerically.
5. `export.nim` → `persist.nim`: `export` is a Nim keyword and cannot appear in
   an `export` list.
6. **Evaluation is indexed and chunk-parallel** (`--threads`), while SGD stays
   serial: filtered ranks come from a pre-built `(h, r)` / `(r, t)` candidate
   index instead of a hash probe per candidate, and workers share a GC-free raw
   view of the embeddings. Parallelism is spent where it cannot break the
   determinism guarantee.

## Tests

```sh
nimble test        # 35 unit tests
nimble bench       # synthetic-scale throughput benchmark
```

Coverage: TSV parsing, vocab order/min-count, init determinism, unit-norm
invariants, hand-computed L1/L2 distances, negative-sampling distributions,
finite-difference gradient check, convergence + determinism on the toy graph,
a hand-computed evaluation fixture (raw & filtered), bit-exact checkpoint
round-trips, and — for the evaluation rewrite — a per-candidate oracle that the
indexed/threaded ranks must match on both norms at 1, 3, 4, 8 and 16 workers.

## Non-goals (v0.1.0)

Other KGE models (RotatE, ComplEx, …), GPU, distributed training, hyperparameter
search, servers. Architecturally, `transEGrad`/`computePair` is the only place
that would need extending for a new norm-based model.

## License

MIT
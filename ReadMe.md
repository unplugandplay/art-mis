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
| `--lr-schedule S` | constant | `constant`, `exp` (needs `--lr-decay`) or `linear` |
| `--lr-decay F` | 1.0 | per-epoch factor for `exp`, e.g. 0.99 |
| `--lr-min F` | 0.0 | floor for `exp`, final lr for `linear` |
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

let hp = HyperParams(dim: 32, epochs: 200, lr: 0.01'f32,
                     lrSchedule: lsLinear, lrMin: 0.0005'f32)
let (data, entities, relations, report) = loadDataset("train.tsv", hp)
var rng = initRand(42)
var model = initModel(entities, relations, hp, rng)

let truth = buildTruthSet(data.train, data.valid, data.test)
let sampler = initSampler(data.train, relations.len, entities.len, true, truth)
discard train(model, data.train, sampler, hp, data.valid, truth)

# ranked on `threads` workers; lsLinear decays lr to hp.lrMin over hp.epochs
let metrics = evaluate(model, data.test, truth, hp.norm, threads = 0)
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
neighbour of `alice` becomes her spouse `david` (cosine 0.943). Adding
`--lr-schedule linear` to the same command lifts it to raw MRR 0.781 / filtered
0.875 — the schedule helps at every scale.

### FB15k-237 (14,505 entities · 237 relations · 272,115 train triples)

```sh
scripts/get_data.sh

# reference configuration (3m41s wall with --threads 0 on 8 cores)
./artemis train --data data/fb15k237/train.txt --valid data/fb15k237/valid.txt \
  --test data/fb15k237/test.txt --dim 100 --lr 0.01 --epochs 1000 \
  --lr-schedule linear --lr-min 0.0005 --eval-every 100 --threads 0
```

| test (n = 40,876 ranks) | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|
| raw | 522.2 | 0.182 | 0.115 | 0.191 | 0.321 |
| filtered | 386.6 | **0.289** | 0.206 | 0.318 | **0.453** |

That matches published, tuned TransE baselines on FB15k-237 (filtered MRR ≈ 0.29
/ Hits@10 ≈ 0.46), and validation MRR rose monotonically across all ten evals
(0.196 → 0.289) — the budget, not the model, is still the limit.

### Getting there: budget and learning-rate schedule

The v0.1.0 reference was 200 epochs at a constant lr 0.01 (filtered MRR 0.207 /
Hits@10 0.343, ~15 min single-threaded). Everything below is the same data,
dim 100, margin 1.0, Bernoulli sampling and seed; only budget and schedule move:

| run | filtered MRR | Hits@10 | wall (`--threads 0`) |
|---|---|---|---|
| 200 epochs, constant lr | 0.207 | 0.343 | 2m01s |
| 500 epochs, constant lr | 0.218 | 0.364 | 2m38s |
| 500 epochs, exponential 0.99 → 1e-4 | 0.245 | 0.389 | 2m41s |
| 500 epochs, linear → 5e-4 | 0.264 | 0.421 | 2m40s |
| 1000 epochs, linear → 1e-4 | 0.288 | 0.451 | 3m44s |
| 1000 epochs, linear → 5e-4 | **0.289** | **0.453** | 3m41s |

(The two 1000-epoch floors differ by less than run-to-run spread — validation
MRR even favours 1e-4 — so read both rows as ≈0.29 / ≈0.45.)

Two lessons, measured rather than assumed. More epochs at a *constant* lr buy
almost nothing (+0.011 MRR for 2.5× the budget: the run oscillates in ±0.005
validation noise after ~epoch 300), while decaying the lr to a small floor lets
the same budget settle (+0.046 MRR over constant at 500 epochs, +0.071 at
1000). Linear decay beats exponential here (0.264 vs 0.245 at 500 epochs): it
stays in the productive lr range longer and only anneals at the end.

`--threads 0` is what makes the table cheap: ranking is ~95 % of a run's wall
clock, so the 1000-epoch reference costs 3m41s instead of ~15 min single-threaded
(2.5 min of training, ~11 min of validation passes, ~1 min of test ranking), with
numbers identical to the single-threaded path. Evaluating a saved checkpoint
standalone needs `--truth train.tsv,valid.tsv` for the same filtered basis (see
*Other commands*).

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
7. **Learning-rate schedule is a pure function of the epoch** (`scheduledLr`),
   constant by default so v0.1.0 runs stay bit-identical, with linear and
   exponential decay opt-in. On FB15k-237 this single knob is worth more than
   any reasonable epoch increase at a constant lr (see *Results*).

## Tests

```sh
nimble test        # 39 unit tests
nimble bench       # synthetic-scale throughput benchmark
```

Coverage: TSV parsing, vocab order/min-count, init determinism, unit-norm
invariants, hand-computed L1/L2 distances, negative-sampling distributions,
finite-difference gradient check, convergence + determinism on the toy graph,
a hand-computed evaluation fixture (raw & filtered), bit-exact checkpoint
round-trips, learning-rate schedule math (endpoints, decay, floor) plus a
decayed run that still converges, and — for the evaluation rewrite — a
per-candidate oracle that the indexed/threaded ranks must match on both norms at
1, 3, 4, 8 and 16 workers.

## Non-goals (v0.1.0)

Other KGE models (RotatE, ComplEx, …), GPU, distributed training, hyperparameter
search, servers. Architecturally, `transEGrad`/`computePair` is the only place
that would need extending for a new norm-based model.

## License

MIT
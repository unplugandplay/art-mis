# Artemis

**TransE knowledge-graph embeddings in Nim.**

Artemis trains [TransE](https://papers.nips.cc/paper/5023-translating-embeddings-for-modeling-multi-relational-data)
embeddings (Bordes et al., NeurIPS 2013) on a table of `head relation tail`
triples, evaluates them with standard link-prediction metrics, and exports the
learned vectors — as a small native binary, with no Python anywhere.

Every true fact `(h, r, t)` is trained to satisfy `h + r ≈ t`; facts are pushed
against corrupted negatives with a margin-ranking loss and SGD.

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
filtered metrics use all splits as the known-truth set.

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
| `--valid` / `--test` | — | extra splits (truth set + metrics) |
| `--out DIR` | . | checkpoints, `embeddings.tsv`, `vocab.json` |
| `--quiet` | off | suppress progress lines |

### Other commands

```sh
artemis eval    --model checkpoint.bin --data test.tsv
artemis export  --model checkpoint.bin --out dir
artemis nearest --model checkpoint.bin --entity NAME [--k 10] [--metric cosine|l2]
```

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

## Results

### Bundled toy graph (16 triples, 8 entities, 3 relations — self-eval)

`--dim 10 --epochs 200 --lr 0.05`

| | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|
| raw | 1.66 | 0.682 | 0.375 | 1.000 | 1.000 |
| filtered | 1.47 | 0.766 | 0.531 | 1.000 | 1.000 |

Optimistic (evaluated on training triples), but a useful smoke signal: after
training the 8-entity ranking improves from MRR ≈ 0.29 to ≈ 0.68, and the nearest
neighbour of `alice` becomes her spouse `david`.

### FB15k-237 (14,541 entities · 237 relations · 272k train triples)

See `scripts/get_data.sh` and the repro commands below; numbers are recorded in
the table at the bottom of this file after the reference run.

### Performance (release build, Apple Silicon, single thread)

Synthetic scale — 15,000 entities × 200 relations × 500,000 random triples,
dim 50:

| Metric | Value |
|---|---|
| training throughput | ~850,000 triples/s |
| epoch time (500k triples) | 0.59 s |
| peak RSS | 273 MB |

The per-triple hot loop performs no allocations (preallocated scratch buffers,
raw row-major access), so epoch time scales linearly with triples × dim.

## Design notes

Deliberate deviations from the original blueprint — see `PRD.md` / `CODING_PLAN.md`:

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

## Tests

```sh
nimble test        # 28 unit tests
nimble bench       # synthetic-scale throughput benchmark
```

Coverage: TSV parsing, vocab order/min-count, init determinism, unit-norm
invariants, hand-computed L1/L2 distances, negative-sampling distributions,
finite-difference gradient check, convergence + determinism on the toy graph,
a hand-computed evaluation fixture (raw & filtered), and bit-exact checkpoint
round-trips.

## Non-goals (v0.1.0)

Other KGE models (RotatE, ComplEx, …), GPU, distributed training, hyperparameter
search, servers. Architecturally, `transEGrad` is the only place that would need
extending for a new norm-based model.

## License

MIT
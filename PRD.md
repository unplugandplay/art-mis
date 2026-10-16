# Artemis — Product Requirements Document

**Product:** Artemis
**Version:** 0.1.0 (target)
**Type:** Nim CLI tool + library for training TransE knowledge-graph embeddings
**Status:** Draft
**Date:** 2026-10-16

---

## 1. Background

TransE (Bordes et al., NeurIPS 2013) is the canonical baseline for knowledge-graph
embedding: every entity and relation is mapped to a vector in ℝᵈ such that for a
true fact `(h, r, t)` the translation **h + r ≈ t** holds. Training minimizes a
margin-based ranking loss between a positive triple and a corrupted (negative)
triple.

The conceptual blueprint (see appendix A) sketches a TransE training loop in Nim
using Arraymancer tensors, `Table[string, int]` id maps, L2-norm distances, and
margin loss. Artemis turns that sketch into a complete, reproducible, testable
program: data ingestion → training → evaluation → export.

## 2. Problem Statement

Knowledge-graph embedding today effectively requires Python (PyKEEN, DGL-KE,
PyTorch/TensorFlow). There is no lightweight, native, dependency-thin way to train
TransE embeddings from the Nim ecosystem. Artemis fills that gap with a single
static binary that trains TransE on a TSV of triples and exports the resulting
embeddings.

## 3. Goals and Non-Goals

### 3.1 Goals

- **G1.** Train TransE on user-supplied triples (head, relation, tail) with
  configurable dimension, margin, norm (L1/L2), learning rate, epochs, batch size.
- **G2.** Uniform and Bernoulli negative sampling (head- or tail-corruption).
- **G3.** Enforce the TransE unit-norm constraint on entity embeddings
  (‖e‖₂ ≤ 1) after every update — a step the blueprint omits but the paper requires.
- **G4.** Link-prediction evaluation: Mean Rank, MRR, Hits@1/3/10, raw and filtered.
- **G5.** Export embeddings, id maps, and checkpoints in plain-text formats.
- **G6.** Deterministic runs from a user-provided RNG seed.
- **G7.** A clean Nim library API (the `TransEModel` object) usable independently
  of the CLI.

### 3.2 Non-Goals

- Other KGE models (RotatE, ComplEx, DistMult, TransH/R/D) — architecture must
  not preclude them, but only TransE ships in 0.1.0.
- GPU/CUDA training.
- Distributed training.
- A serving/inference server (embeddings are consumed offline after export).
- Hyperparameter search / auto-tuning.

## 4. Target Users

- Nim developers who want KG embeddings without leaving the Nim toolchain.
- Researchers needing a small, auditable TransE reference implementation.
- Pipelines (e.g. the `runes-lang` family of projects) that need locally trained
  embeddings for small-to-medium graphs (≤ ~10M triples).

## 5. Functional Requirements

### FR-1 — Data ingestion

- Accept a triple file in TSV (or configurable delimiter): `head<TAB>relation<TAB>tail`.
- Support one file, or `train` / `valid` / `test` splits (standard KGE layout).
- Build and persist vocabularies: `entity_to_id`, `rel_to_id` (`Table[string, int]`),
  plus reverse `id_to_entity`, `id_to_rel` (`seq[string]`).
- Skip malformed lines with a warning counter, never abort mid-file.
- Optional `--min-count N` filter to drop rare entities/relations.

### FR-2 — Model

- `TransEModel` object per the blueprint: id tables + two matrices
  (`entity_embeddings: [n_entities, d]`, `rel_embeddings: [n_relations, d]`,
  `float32`).
- Initialization: uniform in `[-6/√d, +6/√d]` (paper default), seeded RNG.
- Entity embeddings re-normalized to unit L2 norm after each update step;
  relation embeddings left unconstrained.

### FR-3 — Scoring

- Dissimilarity `d(h, r, t) = ‖h + r − t‖ₚ` with p ∈ {1, 2} (CLI-selectable,
  default L1 as in the paper; L2 supported per the blueprint).

### FR-4 — Negative sampling

- Uniform corruption: corrupt head or tail with a random entity.
- Bernoulli corruption (paper): per-relation head/tail corruption probability
  estimated from average tails-per-head and heads-per-tail on the training set.
- Corrupted triple must not already exist in the training set when avoidable
  ("filtered" generation is best-effort, bounded retries).

### FR-5 — Training

- Margin-based ranking loss: `L = max(0, margin + d(pos) − d(neg))`.
- SGD with **closed-form TransE gradients** (see §7.3), applied in-place to the
  affected embedding rows only. (Arraymancer autograd is *not* used in the hot
  path — see §9 risks; it may be kept behind a flag for cross-validation of the
  manual gradients.)
- Mini-batching (default 100), shuffled each epoch.
- Configurable: dimensions (default 50), margin (1.0), learning rate (0.01),
  epochs (1000, with early-stop option on validation MRR), negatives per positive
  (default 1), norm (L1).
- Progress reporting: epoch, mean loss, optional validation metrics every N epochs.
- Checkpoint save every N epochs (optional).

### FR-6 — Evaluation

- Standard link-prediction protocol on valid/test: for each triple, replace head,
  then tail, with every entity; rank the true one by score.
- Metrics: Mean Rank, MRR, Hits@1, Hits@3, Hits@10 — both **raw** and **filtered**
  (all known-true triples across train+valid+test excluded from ranking).
- Eval is vectorized over batches of candidate scores.

### FR-7 — Export

- `embeddings.tsv`: `<name>\t<d floats space-separated>` for entities and relations.
- `vocab.json`: id maps, hyperparameters, final metrics.
- `checkpoint.bin`: compact binary (n_entities, n_relations, d, raw float32 data)
  for fast reload / resume.

### FR-8 — CLI

```
artemis train   --data train.tsv [--valid valid.tsv] [options]
artemis eval    --model checkpoint.bin --data test.tsv
artemis export  --model checkpoint.bin [--out dir]
artemis nearest --model checkpoint.bin --entity "paris" [--k 10]
```

All hyperparameters available as flags with the defaults of §FR-5.

## 6. Non-Functional Requirements

- **NFR-1 Performance.** FB15k-scale (≈ 15k entities, 1345 relations, 483k
  triples): one epoch ≤ ~30 s single-threaded on a modern laptop; full default
  run (1000 epochs is aspirational — early stopping expected well before).
- **NFR-2 Memory.** O((n_entities + n_relations) × d) floats plus triple arrays;
  ≤ 2 GB for FB15k-scale.
- **NFR-3 Portability.** Builds on macOS (arm64/x86_64) and Linux with Nim ≥ 2.0,
  `--mm:orc`, `-d:release`.
- **NFR-4 Determinism.** Same seed + same data ⇒ identical embeddings.
- **NFR-5 Parallelism (stretch).** Optional `--threads:on` HogWild-style training
  behind a flag; default build stays single-threaded and deterministic.

## 7. Technical Design

### 7.1 Module layout

```
src/artemis/
  types.nim        # Triple, Vocab, Dataset, TransEModel, HyperParams
  io.nim           # TSV parsing, vocab building, dataset splits
  embeddings.nim   # init, row access, unit-norm normalization
  scoring.nim      # dissimilarity d(h,r,t), batch scoring for eval
  sampling.nim     # uniform / Bernoulli negative sampling
  gradients.nim    # closed-form TransE gradients, SGD update step
  train.nim        # epoch loop, early stopping, checkpointing
  evaluate.nim     # ranking protocol, MRR / Hits@k (raw + filtered)
  export.nim       # TSV / JSON / binary writers and readers
  cli.nim          # command dispatch
src/artemis.nim    # public library API (re-exports)
```

### 7.2 Core types (refined from blueprint)

```nim
type
  Triple* = object
    head*, rel*, tail*: int          # internal: ids, not strings

  RawTriple* = tuple[head, relation, tail: string]   # parsing stage

  Vocab* = object
    toId*: Table[string, int]
    names*: seq[string]              # id -> name

  HyperParams* = object
    dim*: int = 50
    margin*: float32 = 1.0
    lr*: float32 = 0.01
    epochs*: int = 1000
    batchSize*: int = 100
    negPerPos*: int = 1
    norm*: NormKind = nkL1           # nkL1 | nkL2
    bernoulli*: bool = true
    seed*: int64 = 42
    evalEvery*: int = 0              # 0 = off

  TransEModel* = object
    entities*, relations*: Vocab
    entityEmb*: Tensor[float32]      # [nEntities, dim]
    relEmb*: Tensor[float32]         # [nRelations, dim]
```

### 7.3 Closed-form gradients (replaces blueprint's `discard`)

For `x = h + r − t`, L2 distance `d = ‖x‖₂`, when margin loss > 0:

- **L2:** `∇ = x / ‖x‖₂` (0 if ‖x‖ = 0)
- **L1:** `∇ = sign(x)`

Updates (η = lr), positive triple pulls together, negative pushes apart:

```
h ← h − η·∇pos + η·∇neg        (same for r, with t and t_neg swapped in sign)
t ← t + η·∇pos                 t_neg ← t_neg − η·∇neg
```

Then re-normalize updated entity rows to unit norm.

### 7.4 Dependencies

| Package     | Role                                  | Risk mitigation                     |
|-------------|---------------------------------------|-------------------------------------|
| arraymancer | Tensor storage + vectorized math      | Thin wrapper; manual grad path      |
| stdlib      | tables, random, parseopt, json, os    | —                                   |

## 8. Success Metrics / Acceptance Criteria

1. `nimble build -d:release` produces a working `artemis` binary with zero warnings.
2. Unit tests (`nimble test`) cover: parsing, vocab, normalization, scoring,
   sampling distribution sanity, gradient correctness (finite-difference check),
   one full training step on a toy graph.
3. On a small synthetic graph (e.g. family-tree rules), learned embeddings satisfy
   the translation property: `score(h, r, t)` for true triples in bottom quartile
   of all candidate scores.
4. On FB15k-237 or a WN18 subset: filtered MRR and Hits@10 within a reasonable
   band of published TransE numbers for the chosen norm (documented in README).
5. Deterministic: two runs with `--seed 42` produce byte-identical checkpoints.

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Arraymancer autograd overhead/ergonomics on row slices | Slow training | Manual closed-form gradients (§7.3); autograd only as test oracle |
| Arraymancer maintenance status | Build breaks on future Nim | Pin version in `.nimble`; keep tensor usage to a small API surface that could be re-implemented over `seq[float32]` |
| Filtered negative sampling collisions | Slightly optimistic metrics | Bounded retries + documented best-effort |
| HogWild races break determinism | NFR-4 violated | Threads are opt-in flag; deterministic single-thread default |
| Memory blow-up in eval (ranking vs. all entities) | OOM on large graphs | Score candidates in batches; stream top-k |

## 10. Milestones

| # | Milestone | Deliverable |
|---|-----------|-------------|
| M1 | Scaffolding + data pipeline | Parses triples, builds vocab, unit tests |
| M2 | Core model | Init, normalization, scoring, gradients, single train step |
| M3 | Training loop | Epochs, batching, sampling, checkpoints, toy-graph convergence |
| M4 | Evaluation | MRR / Hits@k raw + filtered on a real dataset |
| M5 | CLI + export | All four commands, docs, README with repro numbers |

---

## Appendix A — Source Blueprint

The conceptual sketch this PRD is based on: `TransEModel` with
`entity_to_id` / `rel_to_id` tables and Arraymancer `Tensor[float32]` matrices;
a `train_step` computing `pos_dist = ‖v_h + v_r − v_t‖₂`,
`neg_dist = ‖v_h + v_r − v_t_neg‖₂`, margin loss
`max(0, margin + pos_dist − neg_dist)`, and a placeholder for gradient updates
(replaced by §7.3 in the implementation).

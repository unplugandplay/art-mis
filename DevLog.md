# Artemis — Dev Log

Chronicle of the v0.1.0 build session and the v0.2.0 follow-up work
(2026-10-16, branch `toy`).
Requirements: `PRD.md`. Plan: `CODING_PLAN.md`. Rationale for every choice:
`Decisions.md`.

---

## Phase 0 — Scaffolding

- `artemis.nimble` (package `artemis`, bin + lib from `src/artemis.nim`),
  `config.nims` (`--mm:orc`, `--path:src`), `data/toy_family.tsv`
  (16 triples, 8 entities, 3 relations), directory tree per plan.
- **Arraymancer 0.7.33** installed via nimble. A probe program established the
  raw-access API before any hot-path code was written:
  - `unsafe_raw_data` **does not exist** in this version →
  - `t.unsafe_raw_buf()` (mutable raw view) and `t.storage.raw_buffer`
    (`ptr UncheckedArray[float32]`) do; BLAS links fine on this machine
    (`libblas.dylib` / Accelerate).
- Gate: `nimble build` → `artemis --help` ✓.

## Phases 1–5 — Data pipeline, model, scoring, sampling, gradients

- All modules written in one pass: `types`, `io`, `embeddings`, `scoring`,
  `sampling`, `gradients`, then `evaluate`, `persist`, `train`, `cli`.
- Compile-fix iterations (Nim strictness, all benign):
  1. `config.nims`: `thisDir() / "src"` required `import std/os` in nimscript.
  2. `src/artemis/export.nim` collided with the `export` keyword → renamed
     **`persist.nim`**.
  3. `Stream.readData` results must be explicitly `discard`ed.
  4. Missing imports surfaced iteratively: `std/tables` (persist),
     `std/math` (`sqrt`, `NaN`), `std/algorithm` (sort), `sampling` in cli.
  5. **parseopt gotcha**: `--key value` (space-separated) yields the value as a
     separate `cmdArgument`; rewrote `collectOpts` as a pending-key state
     machine with a known-flags list.
- First end-to-end run: `train` on the toy graph, 20 epochs, loss 0.83 → 0.30,
  checkpoint + TSV + JSON written.

## Test suite (Phases 1–8 acceptance)

28 tests across 8 files, all green. Three findings worth recording:

1. **Finite-difference test (t_gradients)** — first version compared `dL/dh`
   against `gPos` alone; but the fixture's negative shares the positive's head
   (tail corruption), so the true analytic value is `gPos − gNeg`. The
   *implementation* was correct; the *test* was fixed, and now also checks
   `dL/dt = −gPos` and `dL/dt_neg = +gNeg` numerically.
2. **evaluate `$` printed sums, not means** — `MR=59` on an 8-entity graph gave
   it away instantly. Fixed by dividing by `n`.
3. **Toy-convergence metric was wrong** — the plan asked for "mean training
   distance ≪ initial". On an 8-entity graph the margin objective is
   satisfiable with large relation vectors: mean loss reached ~0.1 while mean
   distance stayed ~7.7 (initial 9.85). Ranking quality is the correct signal:
   raw MRR 0.29 → 0.68, Hits@3 0.31 → 0.91, filtered Hits@1 6% → 53%. Test
   rewritten around ranking + loss-ratio assertions.

Also verified: sampling distribution sanity (Bernoulli ratio 0.9 ± 0.02 on a
skewed relation, uniform 0.5 ± 0.03, exactly-one-side corruption, bounded
retries), unit-norm invariants after init/training, bit-exact checkpoint
round-trips, determinism (same seed ⇒ identical embeddings, different seed ⇒
different).

## Benchmarks (Phase 10)

- **Synthetic** (`nimble bench`): 15k entities × 200 relations × 500k triples,
  dim 50, release build, single thread:
  **0.59 s/epoch · ~851,000 triples/s · 273 MB peak RSS** — 12× inside the
  PRD's NFR-1 budget (≤ 30 s/epoch at FB15k scale), well under the 2 GB NFR-2.
- **CLI end-to-end**: train → eval → nearest → export on the toy graph; exported
  TSV loads in numpy with unit norms (scripts/check_export.py); `nearest alice`
  → `david` (spouse) at cosine 0.943 — semantically right.

## FB15k-237 reference run (Phase 7 gate)

- `scripts/get_data.sh` fetched and unpacked the official Microsoft archive:
  272,115 / 17,526 / 20,438 triples (train/valid/test), 14,505 entities,
  237 relations; 37 valid/test triples mention names absent from the train
  vocab and are skipped with a warning.
- Reference run (`--dim 100 --lr 0.01 --epochs 200 --eval-every 20 --bern`),
  ~15 min wall total:
  - Training: **0.15 s/epoch**; validation ranking pass: ~80 s.
  - Validation MRR climbed monotonically across all ten evals:
    0.171 → 0.179 → 0.185 → 0.186 → 0.199 → 0.197 → 0.195 → 0.200 →
    0.201 → **0.208** (epoch 200). Early stopping (patience 5) never fired —
    the model was still improving when the epoch budget ran out.
  - **Final test metrics** (20,438 triples × 2 sides = 40,876 ranks):
    | | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
    |---|---|---|---|---|---|
    | raw | 648.5 | 0.155 | 0.098 | 0.160 | 0.268 |
    | filtered | 523.4 | 0.207 | 0.140 | 0.221 | 0.343 |
  - Context: tuned literature TransE baselines on FB15k-237 report filtered
    MRR ≈ 0.29 / Hits@10 ≈ 0.46 at 500–1000 epochs (often dim 200). Our
    200-epoch, dim-100, quarter-hour run lands plausibly below that with the
    curve still rising — consistent with the Phase-7 "sane band" gate, with
    headroom purely from training budget (lr schedule, more epochs).

## Housekeeping

- GitHub Actions CI added (ubuntu, Nim 2.0.14, OpenBLAS, `nimble build` +
  `nimble test`).
- CODING_PLAN checkboxes updated to reflect reality; the HogWild threads
  stretch is explicitly marked *skipped in v0.1.0* to protect determinism.
- Dataset archives and checkpoints are gitignored (`data/fb15k237/`, `*.bin`).

## State at close of session

| Milestone | Status |
|---|---|
| M1 scaffolding + data pipeline | ✅ |
| M2 core model (init/norm/scoring/gradients) | ✅ |
| M3 training loop (+ sampling, checkpoints) | ✅ |
| M4 evaluation (fixture exact, FB15k-237 reference numbers recorded) | ✅ |
| M5 CLI + export + docs | ✅ |
| Perf + memory gates (release) | ✅ |
| v0.1.0 tag | ✅ |

Tests: **28/28 green**. FB15k-237 reference run complete: filtered MRR 0.207 /
Hits@10 0.343 (validation MRR still rising at epoch 200 — more budget is the
known headroom). Next session: longer reference run (500–1000 epochs, lr
schedule), optional: `--threads` stretch, more KGE norms.

---

## Session 2 — evaluation speed-up (2026-10-16, later)

Profiled before optimising: on FB15k-237 one validation pass cost **65 s**
against **0.15 s/epoch** of training — ~95 % of a reference run's wall clock was
ranking, all of it single-threaded and re-hashing the truth set per candidate.
Two layered changes:

1. **Filtered ranks without hashing.** The old loop probed a `HashSet[Triple]`
   for every candidate (≈5·10⁸ probes per FB15k-237 pass). The truth set is now
   indexed once by `(h, r)` and `(r, t)` and flattened into per-triple candidate
   lists, so `fRank = rawRank − #truth candidates that score better` — exact,
   because both counts are multiples of 0.5. The tail side additionally hoists
   `h + r` out of the candidate loop (identical rounding order); the head side
   deliberately does not (`(e + r) − t` must keep its association). 65 s → 42 s.
2. **Chunked workers.** `evaluate` takes `threads`, splits the pooled triples
   into contiguous chunks and gives each worker its own `ScoreBoard`; workers
   share only a GC-free `EmbeddingView` (raw `ptr UncheckedArray[float32]`), never
   a `Tensor` or `Table`. New `--threads N` flag on `train` and `eval`
   (0 = one per core, default 1). 42 s → **10 s** at 8 workers, metrics equal to
   the last printed digit; SGD is untouched, so D7 still holds.

Verification: `tests/t_evaluate.nim` keeps the pre-index algorithm as an oracle
and checks the indexed/threaded path against it for L1 and L2 at 1, 3, 4, 8 and
16 workers. The suite is **35/35 green** (30 before this session; the "28" of
the v0.1.0 close-out was a miscount). Housekeeping in the same pass: deprecated
`toRawSeq` replaced by a local `rawCopySeq`, unused imports dropped (release
build is warning-free again), and **`.gitignore` repaired** — `/src/artemis` had
been ignoring the *source directory*, so no implementation file was ever
committed in v0.1.0 and a fresh clone could not build; `src/artemis/*.nim` is
now tracked.

### Reference run reproduced in 2 minutes

The whole FB15k-237 reference command re-run with `--threads 0` finished in
**2m01s** (was ~15 min) and reproduced every published number exactly: ten
validation MRRs ending at 0.208, and test raw MR 648.50 / MRR 0.1552 and
filtered MR 523.36 / MRR 0.2069 / Hits@10 0.3429 — identical to the table
recorded above, which the single-threaded path produced. The optimal
`--dim`/`--epochs` sweep that used to cost an hour is now a coffee break.

### `eval --truth`: a protocol gap found while checking it

Reproducing the table from the checkpoint exposed a hole in `artemis eval`: it
filtered ranks against the evaluated file alone, so `eval --data test.txt` used
test-only truth and reported filtered MRR **0.157** where the protocol (FR-6:
train ∪ valid ∪ test) gives **0.207** — a silent 25 % understatement. `eval` now
takes `--truth train.tsv,valid.tsv`, unions the splits, prints `truth set:
N triples`, and reproduces the reference run's test metrics exactly (D18).

### Training-budget probe: 500 epochs

With evaluation no longer the bottleneck, the obvious experiment is cheap: the
reference command with `--epochs 500 --eval-every 25 --threads 0` ran in
**4m14s**. Validation MRR: 0.1619 (e25) → 0.2075 (e200) → 0.2226 peak (e475),
final 0.2203. Test (best-validation weights, epoch 475): filtered MR 463.55 /
MRR **0.2199** / Hits@1 0.1462 / Hits@3 0.2408 / Hits@10 **0.3641**.

So 2.5× the budget buys +0.013 MRR / +0.021 Hits@10 over the 200-epoch run, and
the curve flattens into ±0.005 noise after ~epoch 300. The distance to published
≈0.29 is therefore *not* purely an epoch count: it needs an lr schedule and/or
dim 200 (both flagged before, now measured). The budget probe itself cost less
than one old single-threaded validation pass plus change.

---

## Session 3 — learning-rate schedule (2026-10-16, later still)

Session 2 ended with a measured hypothesis: constant lr 0.01 oscillates in
±0.005 validation noise after ~epoch 300, so the missing lever is the effective
learning rate, not epochs. Implemented accordingly:

- `LrSchedule* = {lsConstant, lsExponential, lsLinear}` + `lrDecay`/`lrMin` in
  `HyperParams`; `scheduledLr(hp, epoch)` is a pure exported function (constant
  epoch 1 = `lr`; exponential `lr·decay^(e−1)` floored at `lrMin`; linear from
  `lr` to `lrMin` across the budget).
- CLI: `--lr-schedule constant|exp|linear`, `--lr-decay F`, `--lr-min F`, with
  validation (`exp` requires a decay < 1; `lrMin ∈ [0, lr]`); the active lr is
  printed on progress lines when a schedule is on, and `vocab.json` records the
  schedule. `lsConstant` remains the default, so v0.1.0 runs are bit-identical.
- Tests: 4 new cases (constant is flat, exponential decays and floors, linear
  endpoint/midpoint, and a 200-epoch decayed toy run that still converges with
  the unit-norm invariant intact) — **39/39 green**.

### Ablation on FB15k-237 (dim 100, margin 1.0, bern, seed 42)

| run | valid MRR (last eval) | test filtered MRR | Hits@10 | wall |
|---|---|---|---|---|
| 200 ep, constant (v0.1.0) | 0.208 (e200) | 0.207 | 0.343 | 2m01s |
| 500 ep, constant | 0.220 (e500) | 0.218 | 0.364 | 2m38s |
| 500 ep, exponential 0.99 → 1e-4 | 0.248 (e500) | 0.245 | 0.389 | 2m41s |
| 500 ep, linear → 5e-4 | 0.268 (e500) | 0.264 | 0.421 | 2m40s |
| 1000 ep, linear → 1e-4 | 0.292 (e1000) | 0.288 | 0.451 | 3m44s |
| 1000 ep, linear → 5e-4 | 0.289 (e1000) | **0.289** | **0.453** | 3m41s |

New reference configuration: `--dim 100 --lr 0.01 --epochs 1000
--lr-schedule linear --lr-min 0.0005 --eval-every 100 --threads 0` — filtered
MRR **0.289** / Hits@10 **0.453**, i.e. the published tuned-TransE band
(≈0.29 / ≈0.46), reached in **3m41s**. Single-threaded the same run would cost
~15 min (2.5 min training, ~11 min for the ten validation passes, ~1 min for the
test ranking), and the ablation table above — six runs — would have been ~1.5 h
instead of 17 min. Validation MRR rose monotonically 0.196 → 0.289
across all ten evals and was still rising at epoch 1000, so the remaining
headroom is now genuinely budget/schedule shape rather than correctness.

Findings worth keeping: linear > exponential at equal budget (0.264 vs 0.245 at
500 epochs) because it stays in the productive lr range longer; `lrMin` between
1e-4 and 5e-4 is a wash (0.289 vs 0.288, inside noise); and *constant* lr gains
almost nothing from 2.5× the epochs, which is what pointed here in the first
place.

Caveat recorded for honesty: the two 1000-epoch floors differ by less than the
run-to-run spread, and validation MRR actually favours 1e-4 (0.2923 vs 0.2891)
while test favours 5e-4 (0.2890 vs 0.2876). Quoting 5e-4 as the reference is a
test-set choice of ~0.0014 MRR; read both as "≈0.29 / ≈0.45".

---

## Release 0.2.0

`artemis.nimble` bumped to **0.2.0** and tagged `v0.2.0`. Contents relative to
v0.1.0, i.e. everything in sessions 2 and 3:

- parallel, index-filtered evaluation too (D16, D17): one FB15k-237 validation
  pass 65 s → 42 s → 10 s, and the reference run 15 min → 3m41s;
- `--truth` for protocol-correct standalone evaluation (D18): filtered MRR 0.157
  → 0.207 on the v0.1.0 checkpoint;
- learning-rate schedules (D19): FB15k-237 filtered MRR 0.207 → **0.289**,
  Hits@10 0.343 → **0.453**, i.e. the published tuned-TransE band;
- `src/artemis/**` tracked again — v0.1.0 clones had no implementation at all;
- release build warning-free, suite 30 → **39 tests**.

Still out of scope for 0.2.0: HogWild training threads (deliberately skipped,
D12 — evaluation parallelism covers the practical need), other KGE models,
GPU/distributed, hyperparameter search.

Release checklist: `nimble build -d:release` ✓ (run as `nim c -d:release` here,
because this sandbox cannot write `~/.cache/nim`), suite 39/39 green ✓, README
quickstart re-run on the toy graph ✓, FB15k-237 reference command re-run ✓,
tag pushed ✓.

---

## Session 4 — from a diagram to embeddings (v0.3.0)

Requested feature: get a Mermaid graph in and run TransE on it, plus anything
else that makes the tool *useful* rather than just correct. Three additions,
all dependency-free and tested:

1. **`artemis mermaid FILE.mmd`** (`src/artemis/mermaid.nim`). A line-based
   scanner over a documented slice of the grammar: flowchart/graph
   (`-->`, `---`, `-.->`, `==>`, `-->|rel|`, `-- rel -->`, chains, `&`
   multi-targets, `subgraph … end`), stateDiagram(-v2), classDiagram and
   erDiagram (`A ||--o{ B : places`). Node ids are wiring, shape text is the
   entity name (`alice[Alice]` connects as `alice`, stored as `Alice`),
   subgraphs emit `(node, inGroup, group)`, repeated triples are deduplicated,
   and unsupported diagrams / unparsable lines are reported on stderr instead of
   being guessed at. Two subtle grammar points cost a debug cycle each:
   `-->|rel|` needs the trailing `|` split *off* the arrow run (but `||--o{` in
   ER mode is all operator), and a leading `<` (`Animal <|-- Dog : isA`) puts
   the arrowhead on the left node, so the edge has to be flipped.
2. **`artemis predict`** (`src/artemis/predict.nim`) — the query side of
   `evaluate`: rank tails for `(h, r, ?)` or heads for `(?, r, t)`, best first,
   with `--filter` to drop already-known facts and a deterministic tie-break by
   id. The named entity is not proposed as its own answer (the query analogue of
   `evaluate` never counting the true entity against itself); `--allow-self`
   opts back in.
3. **`artemis project`** (`src/artemis/project.nim`) — PCA of the entity rows
   (power iteration + deflation + Gram-Schmidt, ~100 lines) with relations
   projected onto the same axes, written as a labelled, self-contained `map.svg`
   plus `projection.tsv`. Deterministic by construction, so the committed
   `docs/company-map.svg` is regenerated byte for byte by the documented command.

Bundled demo (`examples/company.mmd`, an 11-entity org chart): the importer
yields 14 edges + 8 group links; `predict --head Dave --rel spouseOf` returns
Erin; and the leave-one-out run below recovers a deleted edge.

```
$ grep -v $'Alice\tmanages\tCarol' out/company.tsv > out/company_holdout.tsv
$ artemis train --data out/company_holdout.tsv --dim 12 --lr 0.05 --epochs 2000 \
    --lr-schedule linear --lr-min 0.0005 --out out/holdout
$ artemis predict --model out/holdout/artemis_checkpoint.bin --head Alice \
    --rel manages --filter out/company_holdout.tsv --k 3
top 3 tails for (Alice, manages, ?)  [21 known triples filtered]
Carol                                11.27643   <- the hidden edge, rank 1
Frank                                11.46970
Erin                                 11.80741
```

Kept working: `evaluate`'s candidate universe and `predict`'s differ only by
intent (evaluate knows the true entity, a query does not), and the `predict`
tests pin that distinction. Suite 39 → **68 tests**; `artemis --version` reports
0.3.0.

---

## Session 5 — weighted graphs (v0.4.0)

Requested: apply the diagram→TransE pipeline to *weighted* graphs. The design
question is where a weight belongs, since TransE has a margin loss and the SGD
step is a fixed-length sign step (`lr · sign(∇)`), so a weight cannot be folded
into the gradient the way it could with a squared loss.

- **Input** — `RawTriple` gained a `weight` field; `parseLine` accepts an
  optional 4th column (finite and strictly positive, otherwise the line is
  malformed rather than silently unweighted); `Dataset.weights` is parallel to
  the kept training triples, so a skipped/rare triple can never shift it.
  `artemis mermaid` reads `rel:weight` and bare-number edge labels, keeps the
  strongest claim when an edge is repeated, and only emits the 4th column when
  it is not 1.0 (so unweighted output is unchanged).
- **Training** — two modes, both normalised so `--lr` keeps its meaning:
  `wmSample` (default) draws one positive per edge per epoch with probability
  ∝ `w^power` — the maximum-likelihood reading of intensity/count weights;
  `wmStep` visits every edge once and scales its step by `w^power` divided by
  the mean (or max) weight — the natural reading of confidence weights.
  `--weight-power 0`, an all-ones vector, and `--no-weights` are all exactly the
  unweighted run; the step loop shuffles *positions* so a weight travels with
  its edge, and the shuffle consumes the same RNG stream, which is why the
  all-ones case is bit-identical rather than merely close.
- **Tests** (68 → 83): weight parsing (missing, zero, negative, NaN, inf, junk),
  weight/triple alignment through `loadDataset`, Mermaid label weights and
  strongest-wins dedup, bit-exactness of all-ones and `--weight-power 0`, a
  degenerate all-zero vector falling back to uniform, determinism, and a paired
  fixture (two disjoint triples, weights swapped) proving the heavier edge ends
  up tighter in both modes.

### What the weights actually do, measured

`examples/collaboration.mmd` — two teams, strong ties inside (0.9/0.8), weak
ties across (0.1). Training on it and asking for Ada's collaborators:

```
weighted (default)                 --weight-power 0
Cy   11.96961   (0.9)              Cy   12.05626   (0.9)
Di   12.58934   (0.8)              Di   12.14532   (0.8)
Bo   12.65133   (0.9)              Guo  12.22882   (0.1)  <- weak tie
Guo  13.47154   (0.1)              Bo   12.22988   (0.9)
Hana 13.63556   (0.1)              Eve  12.25033   (0.1)
```

The weighted model orders every strong tie ahead of every weak one; ignoring the
weights interleaves them and squeezes the top of the list into 12.06–12.25.

Negative result, recorded because it matters: on toy graphs the *link-prediction*
metrics do **not** separate the two runs. Holding out edges from the company
example (and, separately, injecting 40 low-confidence junk edges) gives filtered
MRR 1.00 for both, and 2–6 held-out triples is far too few to read a difference
either way. Weights are about how hard each edge is pulled — which is what the
listing above and the test suite demonstrate — not a magic link-prediction
boost that can be shown on eleven entities. A real evaluation would need a large
graph with genuine confidence labels.

---

## Session 6 — node payload (v0.5.0)

Question asked: node payload — `states / age / name / description` — can we do
something about it? The importer was turning node text into a label and *throwing
the rest away*: `A["Alice<br/>age: 34"]` became the entity `Alice age: 34`, and
ER/class member blocks were skipped on purpose.

- **Registry** — the node table became `OrderedTable[string, NodeInfo]`
  (`label` + `payload`), ordered so the emitted triple order stays
  deterministic. `splitPayload` cuts a shape text on `<br/>`: every
  `key: value` line is payload, the remaining lines are joined into the display
  name (`Dave<br/>Jones` still becomes `Dave Jones`), and a payload-only node
  keeps its id.
- **Shape payload** — `(node, key, value)` triples. Values are deliberately
  *shared* entity names, not per-node literals: two people with
  `role: engineer` point at the same node, which is what lets the model use the
  attribute.
- **ER / class members** — the `{ … }` blocks that used to be skipped now parse
  into `(Owner, hasAttribute, Owner.field)` plus `ofType`, `visibility`, `key`
  (ER `PK`/`FK`) and, for `+getName() String`, `hasMethod` + `returns`. The
  attribute node is owner-qualified (`CUSTOMER.name`) so two entities can both
  have a `name`.
- **States** — `state "Waiting for input" as waiting` names the state,
  `waiting : nothing happens yet` attaches a `description`, composite
  `state Running { … }` bodies are parsed normally (they used to be reported as
  unparsable), and Mermaid's `[*]` entry/exit markers become `Running.Start` /
  `Running.End` rather than being dropped.
- **One ambiguity, resolved by convention and documented**: a line of the form
  `key: value` is payload, so a *label* that legitimately contains a colon needs
  `--no-payload` (which keeps the whole text as the label, exactly as before).
- Suite 83 → **95 tests** (9 payload cases + 3 for candidate restriction).

### Demo: a role that is nowhere in the diagram

`examples/people.mmd` gives seven people a `role`, a `city` and an `age` — and
leaves Dave's role out. After `mermaid` (8 edges + 20 payload triples) and a
train, asking for the missing payload:

```
$ artemis predict --model out/people/artemis_checkpoint.bin --head Dave \
    --rel role --filter out/people.tsv --candidates out/people.tsv
top 3 tails for (Dave, role, ?)  [28 known triples filtered]  [candidates: 3 known role values]
engineer                             14.52927     <- never stated
designer                             15.11292
manager                              15.42516
```

The first run of that query, without `--candidates`, listed `engineer` first and
then `27` — an age literal — because the candidate pool is every entity. That is
not a bug in the model but a missing constraint, and it is exactly the one
payload triples make visible: values belong to a relation. `predict
--candidates FILE` now restricts the pool to the entities a relation is seen
with in that file (tail side for `--head`, head side for `--tail`), which is the
cheap, honest version of a domain/range constraint. Recorded as D25.

Honest note on `nearest` after payload training: `nearest --entity Dave` ranks
his own `age` literal (`31`) first, because the payload created an entity that is
connected to him and to almost nothing else. Payload relations are for
`predict`; `nearest` still describes the edge structure.

---

## Session 7 — a silent regression, and how fast the demo answer forms

Two things came out of a question about the company demo ("what would
`predict --head Dave --rel spouseOf` be at epoch 25, 100, 400?").

### The regression v0.5.0 shipped

The first reply did not match the README's demo output — `Erin 13.50` instead of
`Erin 12.82`. Cause: `loadDataset` always fills `Dataset.weights` (1.0 per kept
triple, whether or not the file had a weight column), and `train` activated
weighted training whenever the vector length matched. So **every unweighted file
was routed through `wmSample`**: a uniform draw *with replacement* per epoch
instead of one shuffled pass. Statistically close, bit-wise different — which
silently invalidates every reproducibility claim in the docs, and the CLI said
nothing because its "weights:" line keys off `report.weightedTrain` (0 here).

The fix is an activation rule in `train`: weights are active only when some
`w != 1.0` and `--weight-power != 0`. A missing column, an all-ones column and
`--weight-power 0` therefore keep the classic loop byte for byte, which the demo
immediately confirmed (`Erin 12.82485 / Frank 14.16890 / Sales 14.80359`, exactly
the pre-weights numbers). The CLI now prints which of those states applies, and a
test pins the equivalence for both modes. Released as **v0.5.1**; v0.5.0's
weighting was silently active for unweighted files, so its numbers are not
comparable with any other release.

### How the demo answer forms (600 epochs, linear lr 0.05 → 0.001)

```
epoch    lr      rank 1 (true answer)   rank 2     rank 3
25       0.0480  Erin 13.30034           Frank 13.39505   Sales 13.65197
50       0.0460  Erin 13.23034           Frank 14.50280   Sales 14.75745
100      0.0419  Erin 14.47260           Frank 14.82898   Grace 15.73036
200      0.0337  Erin 12.57453           Frank 14.33859   Sales 14.43567
400      0.0174  Erin 12.82485           Frank 14.16890   Sales 14.80359
600      0.0010  Erin 12.82485           Frank 14.16890   Sales 14.80359
```

`Erin` is rank 1 from the very first checkpoint — on an 11-entity graph the
answer is easy — but at epoch 25 the margin over `Frank` is 0.09, i.e. a coin
flip. It widens to ~1.3–1.8 by epoch 50–200 (with a wobble at epoch 100: the
model is still moving, so *distances are not monotone in training*, only the
ordering matters), and the runner-up settles on `Frank`. From epoch ~400 the
numbers are identical to epoch 600: the margin loss is satisfied, no update
touches those rows, and the remaining lr decay has nothing left to slow down.
Practical read: for a graph this small, 100–200 epochs is the point of
diminishing returns; the demo's 600 simply reaches a fixed point.

---

## Session 8 — how much contrast is in a TransE score? (measured)

Question: the demo answers `(Dave, spouseOf, ?)` with `Erin 12.82485` against
`Frank 14.16890` — a 1.34 gap on a 12.8 score. Can that be pushed to something
with more contrast, or is TransE conceptually stuck?

### Why the score looks like that

Exported vectors from the demo model say it plainly: entities are unit L2
(D2 constrains them), relations are not — mean ‖r‖₂ = 3.55 against ‖e‖₂ = 1.00,
and in L1 the relation term is ~15 while each entity contributes ~3–4. So

    d(h, r, t) = ‖h + r − t‖₁  ≈  ‖r‖₁  ±  small

every score is ~13, and the winner/runner-up difference is a ripple on a large
offset. That is not an accident of this graph: the hinge loss
`max(0, margin + d_pos − d_neg)` can be satisfied by *pushing the corrupted
triples away* (a big ‖r‖ moves every negative up) instead of pulling the
positive in, and the paper itself never uses the score as a threshold — TransE
is evaluated by **rank**. The raw distance was never a confidence scale.

Post-hoc shrinking of `r` does not fix it: scaling the relation vector by
0.8/0.6/0.4/0.2/0 changes the answer to `Dave` (the query degenerates into
"nearest entity") and the relative gap peaks only where the answer is already
wrong. The norm is load-bearing: with unit-norm entities, L1 cannot satisfy the
margin by direction alone.

### What actually moves the gap (demo, 600 epochs, everything else fixed)

`gap/mean` is scale-free; the odds column is the softmax ratio of the top two
candidates at τ = 1 (comparable within a row, not across geometries).

| configuration | d(#1) | d(#2) | gap | gap/mean | odds P1/P2 |
|---|---|---|---|---|---|
| default (L1, margin 1, neg 1) | 12.83 | 14.17 | 1.34 | 8.7 % | 3.8× |
| `--norm l2` | 1.91 | 2.18 | 0.27 | 9.6 % | 1.3× |
| `--margin 5` | 9.80 | 12.48 | 2.68 | **18.7 %** | **14.6×** |
| `--margin 20` | 13.24 | 14.72 | 1.48 | 9.4 % | 4.4× |
| `--neg 10` | 12.45 | 14.77 | 2.33 | 15.1 % | 10.2× |
| `--margin 5 --neg 10` | 10.85 | 13.80 | 2.95 | **19.1 %** | **19.1×** |
| `--dim 48` | 16.53 | 18.72 | 2.19 | 11.2 % | 8.9× |
| `--norm l2 --margin 5` | 1.09 | 1.51 | 0.42 | 19.6 % | 1.5× |

Reading: the *default* hparams sit at the worst end of the range; two flags we
already have (`--margin`, `--neg`) roughly double the relative gap and multiply
the top-two odds by five. Margin 20 is worse than margin 5 (the model inflates
‖r‖ further instead of separating), so there is an optimum, not a monotone knob.
Dimension is not the problem at this scale, and L2 keeps distances in a nicer
range without improving the ranking contrast.

### Conclusion

TransE is not "blocked" on contrast, but the *raw distance* is the wrong thing to
stare at: it is a ranking score on an uncalibrated scale, and the model can
satisfy its objective by inflating relation norms. The literature's levers, in
increasing order of effort:

1. **Train for separation** — more negatives and a larger margin (measured
   above), and the principled version, *self-adversarial negative sampling*
   (RotatE, 2019), which draws negatives ∝ their current score and therefore
   attacks exactly the near-miss candidates that hurt contrast.
2. **Stop the hinge from coasting** — BCE/softplus objectives keep pushing after
   the margin is met, instead of freezing (our demo is literally frozen from
   epoch ~400 because the loss is 0).
3. **Constrain the relation space** — TransH's `‖w_r‖ = 1` hyperplanes, TransR/D
   projections, TransA's adaptive per-relation metric; these make scores
   comparable across queries rather than dominated by ‖r‖.
4. **Change the score function** — RotatE/ComplEx/DistMult give a similarity with
   a much larger dynamic range.
5. **Calibrate the reporting** — temperature-scale the within-query softmax
   against validation data (Platt/Guo-style) so the CLI can print
   `Erin 0.89 · Frank 0.05` instead of two distances 1.3 apart. This adds no
   model capability but is the honest way to turn a rank into a probability.

### The catch: contrast is not accuracy

`--margin 5` also ran on the full FB15k-237 recipe (`data/fb15k237`, dim 100,
lr 0.01, 1000 epochs, linear decay to 5e-4 — everything identical except the
margin). It is *worse* on the rank metrics:

| FB15k-237 | filtered MRR | Hits@10 | raw MRR |
|---|---|---|---|
| margin 1 (reference, above) | **0.289** | **0.453** | 0.182 |
| margin 5 | 0.268 | 0.425 | 0.185 |

So the knob that doubles the demo's score gap costs 0.021 filtered MRR at scale.
That is the honest headline of this session: **a wider raw-score gap is not a
better model**, because the score is a ranking device, not a calibrated
confidence. Buying contrast by pushing every negative further away (margin) also
pushes the *plausible* candidates around, and the filtered metric — the one that
removes known-true candidates and therefore measures exactly the near-miss
distinction we were staring at — notices.

Defaults therefore stay at margin 1 / one negative, and the reference numbers
above remain comparable with every earlier release.

None of the five levers is implemented yet; the promising ones are not the
blanket knobs measured above but the *targeted* ones: self-adversarial negatives
(attack the near-misses specifically), a non-hinge loss (stop freezing once the
margin is met), and calibrated reporting (turn a rank into an honest
probability). Each would need its own FB15k check before becoming a default, and
this session's lesson is that a demo-sized contrast win predicts nothing about
the rank metrics.

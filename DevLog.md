# Artemis — Dev Log

Chronicle of the v0.1.0 build session (2026-10-16, branch `toy`).
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

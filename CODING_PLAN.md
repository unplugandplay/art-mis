# Artemis — Coding Plan

Companion to `PRD.md`. Ordered phases with concrete tasks, target files, and
acceptance criteria. Each phase ends in a compilable, testable state.

**Environment:** Nim ≥ 2.2, nimble, macOS/Linux. Compile flags: `--mm:orc`,
`-d:release` for benchmarks, default debug otherwise.

---

## Phase 0 — Project scaffolding (M1 start)

**Tasks**

- [x] `artemis.nimble`: package `artemis`, `srcDir = "src"`, `bin = @["artemis"]`,
  `requires "nim >= 2.0.0"`, `requires "arraymancer >= 0.7.32"`,
  tasks `test` (runs `tests/all`) and `bench`.
- [x] Directory tree:

```
Artémis/
  artemis.nimble
  config.nims            # --mm:orc, hint[Conf]=off
  src/artemis.nim        # library re-exports
  src/artemis/           # modules per PRD §7.1
  src/artemispkg/…       # (only if bin/lib split needed)
  src/main.nim?          # no — CLI entry is src/artemis_main.nim? → use bin = @["artemis"] with src/artemis.nim as CLI dispatch guarded by isMainModule
  tests/t_types.nim, t_io.nim, t_scoring.nim, t_gradients.nim, t_sampling.nim, all.nim
  data/toy_family.tsv    # small hand-made graph for smoke tests
  ReadMe.md
```

  Decision: single entry — `src/artemis.nim` contains `when isMainModule: cli.main()`
  and re-exports library modules, so `nimble build` yields the `artemis` binary
  and `import artemis` works as a library.
- [x] `nimble install arraymancer` (or `nimble deps`); verify `import arraymancer`
  compiles with `--mm:orc` on this machine.
- [x] Seed `data/toy_family.tsv`: ~20 triples, e.g.
  `(alice, parentOf, bob)`, `(bob, spouseOf, carol)`, …

**Acceptance:** `nimble build` succeeds; `artemis --help` prints stub usage.

---

## Phase 1 — Types & data pipeline (`types.nim`, `io.nim`) (M1)

**Tasks**

- [x] Define `Triple` (int ids), `RawTriple`, `Vocab`, `Dataset`
  (`train, valid, test: seq[Triple]`), `HyperParams`, `TransEModel` per PRD §7.2.
- [x] `io.nim`:
  - `proc parseLine(line, delim): Option[RawTriple]`
  - `proc buildVocab(raws: seq[RawTriple], minCount: int): (Vocab, Vocab)`
  - `proc loadDataset(path, delim, minCount): tuple[data: Dataset, model-vocab]`
  - Malformed-line counter; sorted-then-shuffled-free id assignment order
    (first-appearance order keeps determinism trivially).
- [x] Truth-set index for filtered metrics later: `HashSet[Triple]` built from
  all splits (`proc buildTruthSet`).
- [x] Tests `t_io.nim`: parse happy path, skip malformed, min-count filtering,
  round-trip id→name.

**Acceptance:** `nimble test` green on io/types; toy file loads with expected
entity/relation counts.

---

## Phase 2 — Embeddings (`embeddings.nim`) (M2)

**Tasks**

- [x] `proc initModel(vocabs, hp, rng): TransEModel` — uniform `[-6/√d, 6/√d]`
  via Arraymancer `randomTensor` or manual fill through `seq[float32]` →
  `toTensor`.
- [x] `proc normalizeEntities(model)` — L2-normalize each entity row in place
  (`_.clamp` not needed; divide by norm, guard ‖v‖=0).
- [x] `proc normalizeRow(model, idx)` — post-update single-row normalization.
- [x] Row accessors returning **copies** for math (`model.entityEmb[i, _]`) and a
  mutable-view helper for updates; benchmark both, keep the fast one
  (Arraymancer slicing copies by default — likely use
  `model.entityEmb.data.toOpenArray(i*dim, (i+1)*dim)` style raw access).
- [x] Tests: after init, all entity rows have ‖v‖₂ ≈ 1; determinism under seed.

**Acceptance:** model initializes deterministically; normalization invariant holds.

---

## Phase 3 — Scoring (`scoring.nim`) (M2)

**Tasks**

- [x] `proc distance(model, t: Triple, norm: NormKind): float32` —
  `‖h + r − t‖ₚ`, p ∈ {1,2}. Implement over raw `float32` pointers/openArrays
  (avoid Arraymancer temporaries in hot loop: one fused pass, no allocs).
- [x] `proc scoreAllTails(model, h, r, outBuf)` and `scoreAllHeads(model, r, t, outBuf)`
  for evaluation: vectorized with Arraymancer broadcasting
  (`cand = E + (h + r)` then row-norms) — measure; fall back to fused loop if
  broadcasting allocates too much.
- [x] Tests `t_scoring.nim`: hand-computed distances on a 2-dim fixture; L1 vs L2.

**Acceptance:** distances match hand computation to 1e-6; benchmark note recorded
in README (scoring all tails on toy graph < 1 ms).

---

## Phase 4 — Negative sampling (`sampling.nim`) (M3)

**Tasks**

- [x] `proc corruptUniform(rng, t: Triple, nEnt, corruptHead: bool): Triple`.
- [x] `proc bernoulliTable(data: Dataset, nRel): seq[float32]` — per-relation
  `tph / (tph + hpt)` = P(corrupt head).
- [x] `proc sampleNegative(rng, data, model-ish params, t): Triple` — chooses
  side by Bernoulli (or 0.5 if uniform mode), resamples up to K=10 times if the
  corrupted triple ∈ truth set, then accepts.
- [x] Tests `t_sampling.nim`: distribution sanity (corrupts only one side,
  correct side ratio on skewed relation), never returns the input triple,
  bounded retries terminate.

**Acceptance:** 100k samples on toy data: no positive returned; head/tail ratio
matches Bernoulli table within ±2%.

---

## Phase 5 — Gradients & SGD step (`gradients.nim`) (M3)

**Tasks**

- [x] `proc transEGrad(x: openArray[float32], norm): gradient in-place buffer` —
  L2: `x/‖x‖`; L1: `sign(x)`.
- [x] `proc trainStep(model, pos, neg, hp): float32` —
  1. `xPos = h + r − t`, `xNeg = h + r − t_neg` (fused loops)
  2. `dPos, dNeg`; `loss = max(0, margin + dPos − dNeg)`
  3. if `loss > 0`: apply updates of PRD §7.3 with `hp.lr`
     (`h -= η∇pos - η∇neg`… careful: h and r appear in both terms;
     t only positive side, t_neg only negative side)
  4. `normalizeRow` on h, t, t_neg
  5. return loss
- [x] Shared scratch buffers (preallocated per epoch, reused per step).
- [x] **Finite-difference test** `t_gradients.nim`: numeric gradient of the
  full margin loss w.r.t. a few coordinates vs. analytic — agree to 1e-3.
- [x] Optional: `--autograd-check` dev flag reproducing the blueprint verbatim
  with Arraymancer autograd to cross-validate one step. Low priority; cut if
  autograd on slices fights the API.

**Acceptance:** finite-difference test passes; one step on toy graph reduces
`dPos − dNeg` as expected; loss returned is exact.

---

## Phase 6 — Training loop (`train.nim`) (M3)

**Tasks**

- [x] `proc train(model, data, hp): TrainLog` —
  per epoch: shuffle train (seeded `Random`), form batches of `hp.batchSize`,
  for each positive draw `hp.negPerPos` negatives, accumulate mean loss.
- [x] Checkpoint: `export.saveCheckpoint(model, hp, path)` every `ckptEvery`
  epochs (0 = only final).
- [x] Early stopping: if `evalEvery > 0` and valid provided, run filtered MRR on
  valid every N epochs, keep best checkpoint, patience = 5 evals.
- [x] Progress: single-line carriage-return status (epoch, mean loss, elapsed,
  optional MRR); `--quiet` flag.
- [x] Smoke: toy graph, dim=10, 200 epochs ⇒ training-triple mean distance
  ≪ random-init mean distance; assert in test with loose bound.

**Acceptance:** toy-graph convergence test passes deterministically; checkpoint
file round-trips (load == save, bit-exact).

---

## Phase 7 — Evaluation (`evaluate.nim`) (M4)

**Tasks**

- [x] `proc ranks(model, data, truthSet): EvalResult` — for each triple:
  - score true triple once;
  - head-rank: count candidates with strictly-better score (+ tie-break: add
    half the ties — standard "optimistic/pessimistic average" is fine, document);
  - filtered variant: skip candidates ∈ truthSet (other than the true one);
  - batched candidate scoring via Phase-3 vectorized path.
- [x] Aggregate: MeanRank, MRR, Hits@1/3/10, raw + filtered → `EvalResult` object.
- [x] Tests: tiny fixture with known expected ranks (3 entities, hand-computed).
- [x] Benchmark on FB15k-237 (download script `scripts/get_data.sh`); record
  numbers in README; sanity-band check vs. published TransE baselines.

**Acceptance:** fixture ranks exact; FB15k-237 filtered Hits@10 within a sane
band of literature TransE (~0.4–0.5 with tuned hparams — document what we get).

---

## Phase 8 — Export & persistence (`export.nim`) (M5)

**Tasks**

- [x] `saveCheckpoint` / `loadCheckpoint`: binary header (magic `ARTE`, version,
  nEnt, nRel, dim, hyperparams) + raw float32 payload. Bit-exact round-trip test.
- [x] `exportEmbeddingsTsv(model, path)`: entities + relations sections.
- [x] `exportVocabJson(model, hp, metrics, path)`.
- [x] Tests: round-trips; TSV parseable by `awk`/`numpy.loadtxt` sanity.

**Acceptance:** checkpoint round-trip test green; exported TSV loads in Python
with `numpy` without errors (script in `scripts/check_export.py`).

---

## Phase 9 — CLI (`cli.nim` + `src/artemis.nim` main block) (M5)

**Tasks**

- [x] `parseopt`-based dispatch: `train`, `eval`, `export`, `nearest` per PRD §FR-8.
- [x] Flag → `HyperParams` mapping table; `--help` per subcommand.
- [x] `eval --truth FILE[,FILE]` and `--threads N`: filtered ranking against the
  full known-true set (FR-6) and parallel ranking (D16, D18).
- [x] `nearest`: cosine / Euclidean over entity rows, top-k print.
- [x] Man-style README: install, quickstart on toy data, FB15k-237 repro
  commands, flag reference, results table.

**Acceptance:** end-to-end from README works verbatim:
`artemis train --data data/toy_family.tsv --dim 10 --epochs 200` →
`artemis eval` → `artemis nearest --entity alice` prints sensible neighbours.

---

## Phase 10 — Hardening & polish

**Tasks**

- [x] `-d:release` benchmark pass: profile hot loop (`distance`/`trainStep`);
  measured ~850k triples/s single-threaded (synthetic 15k entities × 500k
  triples, dim 50); hot loop is allocation-free by construction (preallocated
  scratch, raw row-major access) — verified indirectly via linear scaling.
- [x] Memory audit at synthetic FB15k-scale (273 MB peak RSS; `/usr/bin/time -l`).
- [x] Stretch: parallel **evaluation** (PRD NFR-5 applied where it is safe):
  truth-set candidate index + chunked workers behind `--threads N`
  (0 = one per core, default 1). SGD stays serial, so D7 determinism holds;
  FB15k-237 validation pass 65 s → 42 s (index) → 10 s (8 workers).
- [x] Learning-rate schedules (`--lr-schedule constant|exp|linear`, `--lr-decay`,
  `--lr-min`; D19): the measured lever for the FB15k-237 gap — 1000 epochs with
  linear decay reaches filtered MRR 0.289 / Hits@10 0.453 in 3m41s, versus
  0.207 / 0.343 for the constant-lr 200-epoch v0.1.0 reference.
- [ ] Stretch: `--threads:on` HogWild *training* flag (PRD NFR-5) — still
  skipped (v0.1.0, v0.2.0) to protect determinism; single-thread ships as the
  default, with evaluation parallelism covering the practical need (D16).
- [x] CI-ready test command; README badges optional.
- [x] Tag `v0.1.0`.
- [x] Tag `v0.2.0` (parallel evaluation, `--truth`, lr schedules; 39 tests).

---

## Implementation order summary

| Phase | Delivers | Gate to proceed |
|-------|----------|-----------------|
| 0 | Buildable skeleton | `nimble build` ok |
| 1 | Data pipeline | io tests green |
| 2 | Model init + norm | norm invariant test |
| 3 | Scoring | hand-check test |
| 4 | Sampling | distribution tests |
| 5 | Gradients | finite-difference test |
| 6 | Training | toy convergence |
| 7 | Evaluation | fixture ranks + FB15k numbers |
| 8 | Export | round-trip tests |
| 9 | CLI + docs | README repro |
| 10 | Perf + release | v0.1.0 tag |

## Key technical decisions (locked)

1. **Manual closed-form gradients** in the hot loop, not Arraymancer autograd —
  blueprint's `discard` is replaced by PRD §7.3 math. Autograd at most a test oracle.
2. **Raw `openArray[float32]` inner loops** over the tensor storage for
  per-triple ops; Arraymancer broadcasting only for batched eval scoring.
3. **Unit-norm constraint on entities after every update** (paper requirement,
  absent from blueprint).
4. **First-appearance id order + seeded RNG everywhere** ⇒ bit-exact
  determinism (acceptance criterion #5).
5. **`Tensor[float32]` storage layout is row-major `[n, dim]`**; row `i` =
  `storage[i*dim ..< (i+1)*dim]` — all hot-path code relies on this.

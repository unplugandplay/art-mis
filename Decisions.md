# Artemis — Decision Records

Each entry: **decision → context → consequence**. The five "locked" decisions
from `CODING_PLAN.md` are marked (L).

---

## D1 — Manual closed-form gradients, not Arraymancer autograd (L)

**Decision.** The SGD step applies the analytic TransE gradients directly to the
affected embedding rows; autograd is not used in the hot path.

**Context.** The original blueprint left the update as a `discard` placeholder
with a comment that autograd would handle it. For `x = h + r − t` the gradient
of `‖x‖₂` is `x/‖x‖₂` and of `‖x‖₁` is `sign(x)` — trivial closed forms. Only
3–4 embedding rows change per step, and Arraymancer autograd on row slices
would allocate temporaries and fight the in-place update model.

**Consequence.** Zero-allocation hot loop; gradient correctness is pinned by a
finite-difference test (`tests/t_gradients.nim`, tolerance 1e-2 on float32).
If a non-norm model is ever added, this module is where autograd would return.

## D2 — Unit-norm constraint on entities after every update (L)

**Decision.** Entity rows are re-projected onto the unit L2 sphere after
initialization and after every update that touches them; relation rows are
unconstrained.

**Context.** The TransE paper constrains ‖e‖₂ ≤ 1; the blueprint omitted it.

**Consequence.** An invariant test enforces ‖e‖ = 1 ± 1e-4 after training.
Rows shared between the positive and negative update are normalized twice,
which is idempotent and therefore harmless.

## D3 — Fused loops over raw storage instead of Arraymancer broadcasting (L)

**Decision.** Per-triple scoring, gradients, and updates run over
`storage.raw_buffer` (`ptr UncheckedArray[float32]`) in one fused pass per
distance; Arraymancer is used for storage, `newTensor`, and `toRawSeq`.

**Context.** The plan allowed broadcasting for batched eval scoring, with a
fallback to fused loops. Fused loops were chosen outright after the probe:
same O(n·dim) complexity, zero temporaries, no allocation.

**Consequence.** ~851k triples/s single-threaded at dim 50; 273 MB RSS at
15k × 500k scale. The code depends on only a small, stable slice of
Arraymancer's API (see D5).

## D4 — Sequential update application for shared rows (L)

**Decision.** The positive-side update (`h ← h − ηgPos`, `r ← r − ηgPos`,
`t ← t + ηgPos`) and the negative-side update (`h' ← h' + ηgNeg`,
`r' ← r' + ηgNeg`, `t' ← t' − ηgNeg`) are applied in sequence to whichever rows
they name, rather than special-casing head- vs tail-corruption.

**Context.** `r` is always shared between the pair; one of `h`/`t` is shared
depending on the corrupted side. Floating-point addition commutes well enough
at float32 here, and the pattern stays correct for both corruption modes.

**Consequence.** One uniform code path; both corruption modes verified
numerically by the finite-difference test (`dL/dh = gPos − gNeg` caught an
initial test bug, not an implementation bug).

## D5 — Raw-storage API verified by probe before implementation (L)

**Decision.** `Tensor[float32]` with row-major layout, row `i` at
`i*dim ..< (i+1)*dim`, accessed via `storage.raw_buffer`.

**Context.** `unsafe_raw_data` (used in some Arraymancer versions) does not
exist in 0.7.33; a probe program confirmed `storage.raw_buffer` is
`ptr UncheckedArray[float32]` there, and that `unsafe_raw_buf`/`toRawSeq`/
`newTensor` all work with `--mm:orc`.

**Consequence.** If Arraymancer ever breaks, re-implementing the storage over
`seq[float32]` touches only `types.nim` + the accessor lines — the PRD's
stated risk mitigation.

## D6 — `export.nim` renamed to `persist.nim`

**Decision.** The persistence module is `persist.nim`.

**Context.** `export` is a Nim keyword; a module named `export` cannot appear
in an `export` list.

**Consequence.** No keyword gymnastics; `export ... persist` reads fine.

## D7 — Determinism by construction (L)

**Decision.** Ids are assigned in first-appearance order; all randomness flows
through `std/random` `Rand` instances seeded from `HyperParams.seed` (model
init) and a fresh `initRand(hp.seed)` inside `train` (shuffling + sampling).

**Context.** PRD acceptance criterion: same seed + data ⇒ identical
embeddings.

**Consequence.** Bit-exact reproducibility is tested (toy training twice with
seed 42 compares equal via `rawCopy`). Multi-threaded HogWild training would
break this invariant — hence D12.

## D8 — Negative sampling: uniform + Bernoulli, bounded retries

**Decision.** Corruption side is chosen per relation with
P(corrupt head) = tph/(tph+hpt) in Bernoulli mode (one-to-many relations
corrupt the head more often), else 0.5. Candidates are resampled up to 10
times to avoid known-true triples, then accepted as-is.

**Context.** The TransH-lineage "bern" trick; exact truth-avoidance is
impossible on saturated vocabularies (toy test proves termination).

**Consequence.** Distribution sanity is tested (0.9 ± 0.02 on a skewed
relation; uniform 0.5 ± 0.03); the best-effort nature is documented in the
PRD risk table.

## D9 — Rank formula: average tie handling

**Decision.** `rank = 1 + #better + 0.5·#ties` against all candidates; both
head and tail ranks are pooled into one metric set; filtered ranks skip
candidates forming any train∪valid∪test triple.

**Context.** Standard link-prediction protocol; ties are rare with float32
scores but must not crash the math.

**Consequence.** Hand-computed fixture test (3 entities) pins exact raw and
filtered ranks, MRR, and Hits@k.

## D10 — CLI option parsing: pending-key state machine

**Decision.** `collectOpts` treats space-separated values (`--data x`) by
parking the key and consuming the next `cmdArgument`; a fixed list of boolean
flags (`quiet`, `bern`, `no-bern`, `help`) is exempt.

**Context.** `std/parseopt` yields `--key value` values as separate arguments
(probe-verified) rather than attaching them to `p.val`.

**Consequence.** Both `--key value` and `--key=value` work; misuse produces
clear errors ("option --data expects a value").

## D11 — Convergence measured by ranking, not raw distance

**Decision.** Toy-graph convergence tests assert loss ratio and ranking
quality (MRR, Hits@3), not mean training-triple distance.

**Context.** On tiny vocabularies the margin objective is satisfiable with
large relation vectors: mean loss ~0.1 coexisted with mean distance ~7.7
(initial 9.85). Distance-only assertions would either fail or be meaningless.

**Consequence.** The test now captures what actually matters (the model ranks
true triples first); observed seed-42 trajectory: MRR 0.29 → 0.68.

## D12 — Threads stretch skipped in v0.1.0

**Decision.** No HogWild `--threads:on` mode ships yet.

**Context.** PRD NFR-5 lists parallelism as a stretch goal; it conflicts with
the determinism acceptance criterion (D7).

**Consequence.** Single-thread default is deterministic and already 12× inside
the performance budget; threads remain a flagged, opt-in future feature.

## D13 — Checkpoint format: magic + version + embedded vocab

**Decision.** Binary layout `ARTECKPT` magic, version u32, dims, hyperparams
(norm, bernoulli, margin, lr, seed), name tables, raw float32 payload,
native little-endian.

**Context.** `nearest`/`eval` need the vocabulary to resolve names; TSV alone
cannot round-trip bit-exactly.

**Consequence.** Bit-exact round-trip is tested; unknown files raise a clear
`ValueError`; version field guards future format changes.

## D14 — L1 is the default norm; L2 is supported

**Decision.** `HyperParams.norm` defaults to `nkL1` (paper's best setting);
`--norm l2` matches the original blueprint's formulation.

**Context.** The blueprint used L2; the paper reports L1 as superior.

**Consequence.** Both norms share one code path via `NormKind`; L2 is what the
finite-difference test exercises (smooth gradient), L1 via `sign` is exercised
by training tests.

## D15 — Tests import stdlib modules explicitly

**Decision.** Test files import `std/tables`, `std/math`, etc. directly even
when library modules already import them.

**Context.** Nim's `import` is not transitive: a `Table` field is visible
through a type, but its procs are not.

**Consequence.** Slightly noisier imports; zero ambiguity about where procs
come from.

## D16 — Evaluation runs on worker threads; training stays single-threaded

**Decision.** `evaluate` takes `threads` (CLI `--threads N`, `0` = one per core,
default **1**). It splits the pooled triples into that many contiguous chunks;
each worker owns a `ScoreBoard` (score + hoisted-term scratch) and accumulates
its own `EvalResult`, and the chunk partials are summed in chunk order. Workers
share only a GC-free `EmbeddingView` (raw `ptr UncheckedArray[float32]` + dims),
never a `Tensor` or a `Table`.

**Context.** On FB15k-237 (14,505 entities, dim 100) one validation pass cost
**65 s** against **0.15 s/epoch** of training: ~95 % of a reference run's wall
clock was ranking, and the SGD loop is inherently serial. Nim 2.x enables
threads by default, so no build flag is required.

**Consequence.** 8 workers cut the pass to ~10 s (≈6.5×) with metrics equal to
~1e-12; the default stays 1 so the documented reference command remains
bit-reproducible, and D7 is untouched because the training path never leaves one
thread. Re-ranks are reproducible for a fixed thread count — only the order of
float64 additions changes, never the per-triple ranks. HogWild *training*
remains skipped (D12); this is NFR-5 applied where it cannot hurt determinism.

## D17 — Filtered ranks come from a precomputed candidate index

**Decision.** Known-true tails of each `(h, r)` and heads of each `(r, t)` are
indexed once, flattened into per-evaluation-triple candidate lists (CSR), and a
filtered rank is computed as `rawRank − #{truth candidates that score better}`
(a tie subtracts 0.5).

**Context.** The straightforward protocol probes a `HashSet[Triple]` for every
candidate — ≈5·10⁸ hashes per FB15k-237 validation pass — and drags hashing into
the middle of the scoring loop.

**Consequence.** Bit-identical metrics, because both counts are exact multiples
of 0.5 in float64; the rank loop is now pure arithmetic. `tests/t_evaluate.nim`
keeps the original per-candidate algorithm as an oracle and checks the new path
against it for L1 and L2, at 1, 3, 4, 8 and 16 threads. The tail-side `h + r` is
hoisted out of the candidate loop (identical rounding order — one addition less
per candidate element); the head side is deliberately left as `(e + r) − t`,
since hoisting `r − t` would associate the sum differently and change the last
bits.

## D18 — Standalone `eval` takes `--truth` files for filtered ranking

**Decision.** `artemis eval` always filters against the file it evaluates, and
`--truth train.tsv,valid.tsv` adds further known-true splits to that set (the
union is what `--valid/--test` given to `train` already use). The command prints
`truth set: N triples` so the filtering basis is never implicit.

**Context.** FR-6 defines filtered metrics against *all* known-true triples, but
`eval --data test.tsv` only knew the test split. Measured on the FB15k-237
reference checkpoint: filtered MRR 0.157 (test-only truth) vs 0.207 (protocol) —
a silent 25 % understatement of the headline metric, and the trap of reporting
it as "filtered".

**Consequence.** The documented protocol is reproducible from a checkpoint
alone; the README quickstart and the FB15k-237 section spell out the flag, and
the reference run's own test metrics are reproduced exactly
(`--truth train.txt,valid.txt` → filtered MRR 0.2069 / Hits@10 0.3429).

## D19 — Learning-rate schedule: pure function, constant by default, linear wins

**Decision.** `LrSchedule = {lsConstant, lsExponential, lsLinear}` plus
`lrDecay` and `lrMin` on `HyperParams`; `scheduledLr(hp, epoch)` returns the
epoch's rate (constant `lr`; `lr·decay^(e−1)` floored at `lrMin`; linear from
`lr` to `lrMin` across `hp.epochs`). CLI `--lr-schedule constant|exp|linear`,
`--lr-decay F`, `--lr-min F`. `lsConstant` is the default, so every v0.1.0
command reproduces bit-identically.

**Context.** The 500-epoch probe (D16 follow-up) showed constant lr 0.01 losing
to nothing but its own noise after ~epoch 300: 2.5× the epochs bought +0.011
filtered MRR, while the curve oscillated ±0.005. The effective learning rate —
not the epoch count — was the lever, and the README's "gap is training budget"
claim needed to be split into "budget" vs "schedule".

**Consequence.** On FB15k-237 the ablation (same data, dim 100, seed 42) runs
0.207 (200 ep constant) → 0.218 (500 ep constant) → 0.245 (500 ep exponential
0.99 → 1e-4) → 0.264 (500 ep linear → 5e-4) → **0.289** filtered MRR / 0.453
Hits@10 (1000 ep linear → 5e-4), i.e. the published tuned-TransE band, in 3m41s
with `--threads 0`. Linear beats exponential at equal budget because it stays in
the productive lr range longer. The schedule is a pure function tested directly
(`scheduledLr`) and applied per epoch in the loop; the active lr is printed when
a schedule is on and recorded in `vocab.json`. Checkpoint layout unchanged (the
schedule is a training-time knob, not model state).

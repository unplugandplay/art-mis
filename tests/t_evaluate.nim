import std/[unittest, sets, random, math]
import artemis/types
import artemis/evaluate
import artemis/embeddings
import artemis/scoring
import helpers

# Fixture: e0=(0,0), e1=(1,0), e2=(0,2); r0=(1,0).
# Eval triple (e0, r0, e2); sTrue = L1 |(1,-2)| = 3.
#   head side: e1 -> 4 (no), e2 -> 1 (yes, (2,0,2) not in truth): rank 2 raw & filtered
#   tail side: e0 -> 1 (yes, (0,0,0) in truth): raw +1, filtered skip
#              e1 -> 0 (yes, (0,0,1) in truth): raw +1, filtered skip
#   => raw ranks {2, 3}, filtered ranks {2, 1}
let m = modelFromArrays(
  @["e0", "e1", "e2"], @["r0"], 2,
  @[0.0'f32, 0,  1, 0,  0, 2],
  @[1.0'f32, 0])

proc referenceEvaluate(m: TransEModel, data: seq[Triple], truth: HashSet[Triple],
                       norm: NormKind): EvalResult =
  ## The pre-index implementation, verbatim: a hash-set probe per candidate.
  ## Kept as the oracle for the flattened-candidate / threaded code.
  proc add(e: var EvalResult, rank, fRank: float64) =
    e.meanRank += rank
    e.fMeanRank += fRank
    e.mrr += 1.0 / rank
    e.fMrr += 1.0 / fRank
    e.hits1 += (if rank <= 1.0: 1.0 else: 0.0)
    e.hits3 += (if rank <= 3.0: 1.0 else: 0.0)
    e.hits10 += (if rank <= 10.0: 1.0 else: 0.0)
    e.fHits1 += (if fRank <= 1.0: 1.0 else: 0.0)
    e.fHits3 += (if fRank <= 3.0: 1.0 else: 0.0)
    e.fHits10 += (if fRank <= 10.0: 1.0 else: 0.0)
    inc e.n

  let nEnt = m.nEntities
  var scores = newSeq[float32](nEnt)
  for t in data:
    let sTrue = distance(m, t, norm).float64
    scoreAllHeads(m, t.rel, t.tail, norm, scores)
    var rank, fRank = 1.0
    for e in 0 ..< nEnt:
      if e == t.head: continue
      let s = scores[e].float64
      let inTruth = Triple(head: e, rel: t.rel, tail: t.tail) in truth
      if s < sTrue:
        rank += 1
        if not inTruth: fRank += 1
      elif s == sTrue:
        rank += 0.5
        if not inTruth: fRank += 0.5
    result.add(rank, fRank)
    scoreAllTails(m, t.head, t.rel, norm, scores)
    rank = 1.0
    fRank = 1.0
    for e in 0 ..< nEnt:
      if e == t.tail: continue
      let s = scores[e].float64
      let inTruth = Triple(head: t.head, rel: t.rel, tail: e) in truth
      if s < sTrue:
        rank += 1
        if not inTruth: fRank += 1
      elif s == sTrue:
        rank += 0.5
        if not inTruth: fRank += 0.5
    result.add(rank, fRank)

proc syntheticFixture(seed: int64):
    tuple[m: TransEModel, data: seq[Triple], truth: HashSet[Triple]] =
  ## 64 entities / 5 relations, 200 evaluated triples, and a truth set that
  ## also holds 200 *other* true triples so that filtering actually bites.
  let nEnt = 64
  let nRel = 5
  var entNames, relNames: seq[string]
  for i in 0 ..< nEnt: entNames.add "e" & $i
  for i in 0 ..< nRel: relNames.add "r" & $i
  var rng = initRand(seed)
  let hp = HyperParams(dim: 8, seed: seed)
  result.m = initModel(makeVocab(entNames), makeVocab(relNames), hp, rng)
  for _ in 0 ..< 200:
    let t = Triple(head: rng.rand(nEnt - 1), rel: rng.rand(nRel - 1),
                   tail: rng.rand(nEnt - 1))
    result.data.add t
    result.truth.incl t
  for _ in 0 ..< 200:
    result.truth.incl Triple(head: rng.rand(nEnt - 1), rel: rng.rand(nRel - 1),
                             tail: rng.rand(nEnt - 1))

proc close(a, b: EvalResult, tol = 1e-9): bool =
  a.n == b.n and
    abs(a.meanRank - b.meanRank) < tol and abs(a.mrr - b.mrr) < tol and
    abs(a.hits1 - b.hits1) < tol and abs(a.hits3 - b.hits3) < tol and
    abs(a.hits10 - b.hits10) < tol and
    abs(a.fMeanRank - b.fMeanRank) < tol and abs(a.fMrr - b.fMrr) < tol and
    abs(a.fHits1 - b.fHits1) < tol and abs(a.fHits3 - b.fHits3) < tol and
    abs(a.fHits10 - b.fHits10) < tol

suite "evaluate: fixture with hand-computed ranks":
  let data = @[Triple(head: 0, rel: 0, tail: 2)]
  let truth = [Triple(head: 0, rel: 0, tail: 0),
               Triple(head: 0, rel: 0, tail: 1),
               Triple(head: 0, rel: 0, tail: 2)].toHashSet

  test "raw and filtered ranks match hand computation":
    let r = evaluate(m, data, truth, nkL1)
    check r.n == 2
    check r.meanRank == 5.0        # 2 + 3
    check r.fMeanRank == 3.0       # 2 + 1
    check abs(r.mrr - (0.5 + 1.0 / 3.0)) < 1e-9
    check abs(r.fMrr - (0.5 + 1.0)) < 1e-9
    check r.hits1 == 0.0
    check r.fHits1 == 1.0          # filtered tail rank is 1
    check r.hits3 == 2.0
    check r.fHits10 == 2.0

  test "empty data yields zero result":
    let r0 = evaluate(m, @[], truth, nkL1)
    check r0.n == 0
    check r0.mrrMean == 0.0

suite "evaluate: flattened filter index and worker chunks":
  let fx = syntheticFixture(7)

  test "filtered ranks match the per-candidate hash-set algorithm (L1)":
    for threads in [1, 3, 8]:
      check close(evaluate(fx.m, fx.data, fx.truth, nkL1, threads),
                  referenceEvaluate(fx.m, fx.data, fx.truth, nkL1))

  test "filtered ranks match the per-candidate hash-set algorithm (L2)":
    for threads in [1, 4]:
      check close(evaluate(fx.m, fx.data, fx.truth, nkL2, threads),
                  referenceEvaluate(fx.m, fx.data, fx.truth, nkL2))

  test "thread count does not change the pooled metrics":
    let seqR = evaluate(fx.m, fx.data, fx.truth, nkL1, 1)
    for threads in [2, 5, 16]:
      check close(evaluate(fx.m, fx.data, fx.truth, nkL1, threads), seqR)

  test "more workers than triples is safe":
    let tiny = @[Triple(head: 0, rel: 0, tail: 1)]
    check evaluate(fx.m, tiny, fx.truth, nkL1, 64).n == 2

  test "empty truth set degrades to raw ranks":
    let r = evaluate(fx.m, fx.data, initHashSet[Triple](), nkL1, 4)
    check abs(r.fMrr - r.mrr) < 1e-9
    check abs(r.fMeanRank - r.meanRank) < 1e-9

import std/[unittest, sets]
import artemis/types
import artemis/evaluate
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

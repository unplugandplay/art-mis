import std/[unittest, sets]
import artemis/types
import artemis/scoring
import artemis/predict
import helpers

# Fixture: e0=(0,0), e1=(1,0), e2=(2,0); r0=(1,0).
#   d(e0, r0, e) = |1 - x_e|  -> e1:0, e0:1, e2:1   (ties broken by id)
#   d(e, r0, e2) = |x_e - 1|  -> e1:0, e0:1, e2:1
let m = modelFromArrays(
  @["e0", "e1", "e2"], @["r0"], 2,
  @[0.0'f32, 0,  1, 0,  2, 0],
  @[1.0'f32, 0])

suite "predict: hand-computed rankings":
  let v = view(m)

  test "tails are ranked best-first with deterministic tie-breaks":
    let preds = predictTails(v, head = 0, rel = 0, k = 3, allowSelf = true)
    check preds.len == 3
    check preds[0].entity == 1 and preds[0].score == 0.0'f32
    check preds[1].entity == 0 and preds[1].score == 1.0'f32
    check preds[2].entity == 2 and preds[2].score == 1.0'f32

  test "heads are ranked best-first":
    let preds = predictHeads(v, rel = 0, tail = 2, k = 3, allowSelf = true)
    check preds.len == 3
    check preds[0].entity == 1 and preds[0].score == 0.0'f32
    check preds[1].entity == 0
    check preds[2].entity == 2

  test "k truncates; the named entity is not proposed as its own answer":
    let preds = predictTails(v, head = 0, rel = 0, k = 1)
    check preds.len == 1
    check preds[0].entity == 1
    for p in predictTails(v, head = 0, rel = 0, k = 3):
      check p.entity != 0 # `e0` is the head we already named
    for p in predictHeads(v, rel = 0, tail = 2, k = 3):
      check p.entity != 2

  test "allowSelf puts the named entity back in the pool":
    let preds = predictTails(v, head = 0, rel = 0, k = 3, allowSelf = true)
    check preds.len == 3
    var found = false
    for p in preds:
      if p.entity == 0: found = true
    check found

  test "known-true candidates can be filtered out":
    let truth = [Triple(head: 0, rel: 0, tail: 1)].toHashSet
    let preds = predictTails(v, head = 0, rel = 0, k = 3, truth = truth,
                             allowSelf = true)
    check preds.len == 2
    check preds[0].entity == 0
    check preds[1].entity == 2

  test "an empty truth set behaves like no filtering":
    let preds = predictTails(v, head = 0, rel = 0, k = 3,
                             truth = initHashSet[Triple](), allowSelf = true)
    check preds.len == 3

  test "k larger than the candidate pool is harmless":
    check predictTails(v, head = 0, rel = 0, k = 99, allowSelf = true).len == 3

suite "predict: candidate restriction":
  let v = view(m)

  test "--candidates keeps a relation to its own value set":
    # only e2 is "seen" as a tail of r0, so a tail question may only answer e2
    var restrict = initHashSet[int]()
    restrict.incl 2
    let preds = predictTails(v, head = 0, rel = 0, k = 5, restrict = restrict)
    check preds.len == 1
    check preds[0].entity == 2

  test "an empty restriction behaves like no restriction":
    # e1 and e2: the named entity (e0) is excluded by default
    check predictTails(v, head = 0, rel = 0, k = 5).len == 2

  test "restriction and filtering compose":
    var restrict = initHashSet[int]()
    restrict.incl 1
    restrict.incl 2
    let truth = [Triple(head: 0, rel: 0, tail: 1)].toHashSet
    let preds = predictTails(v, head = 0, rel = 0, k = 5, truth = truth,
                             restrict = restrict)
    check preds.len == 1
    check preds[0].entity == 2

suite "predict: calibrated probabilities":
  let v = view(m)

  test "no temperature means no probabilities":
    for p in predictTails(v, head = 0, rel = 0, k = 3):
      check p.prob == 0.0

  test "probabilities follow the scores and sum to one":
    let preds = predictTails(v, head = 0, rel = 0, k = 1, allowSelf = true,
                             temperature = 1.0'f32)
    check preds.len == 1
    # the denominator is every eligible candidate (e0, e1, e2), not just the
    # printed one, which shares the mass with e0 and e2
    check preds[0].prob > 0.0 and preds[0].prob < 1.0
    let all3 = predictTails(v, head = 0, rel = 0, k = 3, allowSelf = true,
                            temperature = 1.0'f32)
    var total = 0.0
    for p in all3: total += p.prob
    check abs(total - 1.0) < 1e-9
    check all3[0].prob > all3[1].prob
    # e0 and e2 score identically for this query, so they must tie exactly
    check abs(all3[1].prob - all3[2].prob) < 1e-15

  test "restriction and filtering shrink the denominator":
    var restrict = initHashSet[int]()
    restrict.incl 1
    restrict.incl 2
    let preds = predictTails(v, head = 0, rel = 0, k = 2, restrict = restrict,
                             temperature = 1.0'f32, allowSelf = true)
    var total = 0.0
    for p in preds: total += p.prob
    check abs(total - 1.0) < 1e-9   # only e1 and e2 are eligible

  test "a sharper temperature concentrates the mass":
    let soft = predictTails(v, head = 0, rel = 0, k = 1, allowSelf = true,
                            temperature = 100.0'f32)
    let sharp = predictTails(v, head = 0, rel = 0, k = 1, allowSelf = true,
                             temperature = 0.01'f32)
    check sharp[0].prob > soft[0].prob

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

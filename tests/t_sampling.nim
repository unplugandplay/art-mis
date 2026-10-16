import std/[unittest, random, sets, math]
import artemis/types
import artemis/sampling
import helpers

suite "sampling: negative corruption":
  test "never returns the input triple, corrupts exactly one side":
    # 20 entities, sparse truth set -> resampling can always succeed
    let nEnt = 20
    var triples: seq[Triple]
    for i in 0 ..< 5:
      triples.add Triple(head: i, rel: 0, tail: i + 10)
    let truth = triples.toHashSet
    let s = initSampler(triples, 1, nEnt, false, truth)
    var rng = initRand(1)
    for _ in 0 ..< 10000:
      let t = triples[rng.rand(triples.len - 1)]
      let n = s.sample(rng, t)
      check n != t
      check n notin truth
      check (n.head != t.head) xor (n.tail != t.tail) # exactly one side
      check n.rel == t.rel

  test "Bernoulli ratio matches tph/(tph+hpt) on skewed relation":
    # relation 0: h0 -> t1..t9  =>  tph=9, hpt=1  =>  P(corrupt head) = 0.9
    let nEnt = 10
    var triples: seq[Triple]
    for i in 1 ..< 10:
      triples.add Triple(head: 0, rel: 0, tail: i)
    let truth = triples.toHashSet
    let probs = bernoulliHeadProb(triples, 1)
    check abs(probs[0] - 0.9'f32) < 1e-6'f32
    let s = initSampler(triples, 1, nEnt, true, truth)
    var rng = initRand(2)
    var headCorruptions = 0
    const N = 10000
    let t = Triple(head: 0, rel: 0, tail: 1)
    for _ in 0 ..< N:
      let n = s.sample(rng, t)
      if n.head != t.head:
        inc headCorruptions
    let ratio = headCorruptions.float32 / N.float32
    check abs(ratio - 0.9'f32) < 0.02'f32

  test "uniform mode corrupts head ~50%":
    var triples: seq[Triple]
    for i in 0 ..< 5:
      triples.add Triple(head: i, rel: 0, tail: i + 10)
    let s = initSampler(triples, 1, 20, false, triples.toHashSet)
    var rng = initRand(3)
    var headCorruptions = 0
    const N = 10000
    for _ in 0 ..< N:
      let t = triples[rng.rand(triples.len - 1)]
      if s.sample(rng, t).head != t.head:
        inc headCorruptions
    check abs(headCorruptions.float32 / N.float32 - 0.5'f32) < 0.03'f32

  test "bounded retries terminate on saturated vocab":
    # every possible corruption of the only triple is a known truth:
    # retries must give up and return *something* without hanging
    let triples = @[Triple(head: 0, rel: 0, tail: 1)]
    let s = initSampler(triples, 1, 2, true, triples.toHashSet)
    var rng = initRand(4)
    let n = s.sample(rng, triples[0])
    check n.rel == triples[0].rel

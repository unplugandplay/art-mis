## Negative sampling: uniform and Bernoulli corruption.
##
## Bernoulli (TransH-style): per relation, tph = avg distinct tails per head,
## hpt = avg distinct heads per tail; P(corrupt head) = tph / (tph + hpt).
## One-to-many relations corrupt the head more often, and vice versa.
import std/[random, sets, tables]
import ./types

const maxResampleAttempts = 10

type
  Sampler* = object
    bernoulli*: bool
    headProb*: seq[float32] ## per relation; only used when bernoulli is true
    truth*: HashSet[Triple]
    nEnt*: int

proc bernoulliHeadProb*(train: seq[Triple], nRel: int): seq[float32] =
  ## Per-relation P(corrupt head) = tph / (tph + hpt).
  var tailsPerHead = newSeq[Table[int, HashSet[int]]](nRel)
  var headsPerTail = newSeq[Table[int, HashSet[int]]](nRel)
  for r in 0 ..< nRel:
    tailsPerHead[r] = initTable[int, HashSet[int]]()
    headsPerTail[r] = initTable[int, HashSet[int]]()
  for t in train:
    tailsPerHead[t.rel].mgetOrPut(t.head, initHashSet[int]()).incl(t.tail)
    headsPerTail[t.rel].mgetOrPut(t.tail, initHashSet[int]()).incl(t.head)
  result = newSeq[float32](nRel)
  for r in 0 ..< nRel:
    var tph, hpt = 0.0'f32
    if tailsPerHead[r].len > 0:
      for _, tails in tailsPerHead[r]:
        tph += tails.len.float32
      tph /= tailsPerHead[r].len.float32
    if headsPerTail[r].len > 0:
      for _, heads in headsPerTail[r]:
        hpt += heads.len.float32
      hpt /= headsPerTail[r].len.float32
    if tph + hpt > 0'f32:
      result[r] = tph / (tph + hpt)
    else:
      result[r] = 0.5'f32

proc initSampler*(train: seq[Triple], nRel, nEnt: int, bernoulli: bool,
                  truth: HashSet[Triple]): Sampler =
  result.bernoulli = bernoulli
  result.nEnt = nEnt
  result.truth = truth
  if bernoulli:
    result.headProb = bernoulliHeadProb(train, nRel)

proc sample*(s: Sampler, rng: var Rand, t: Triple): Triple =
  ## Corrupt one side of `t`. P(corrupt head) follows the Bernoulli table
  ## when enabled, else 0.5. Resamples (bounded) to avoid producing a
  ## known-true triple; accepts the last candidate after the bound.
  result = t
  let pHead =
    if s.bernoulli: s.headProb[t.rel]
    else: 0.5'f32
  let corruptHead = rng.rand(1.0) < pHead
  for _ in 0 ..< maxResampleAttempts:
    let e = rng.rand(s.nEnt - 1)
    if corruptHead:
      result.head = e
    else:
      result.tail = e
    if result != t and result notin s.truth:
      return

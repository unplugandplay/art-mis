import std/[unittest, random, math]
import artemis/types
import artemis/io
import artemis/embeddings
import artemis/scoring
import artemis/sampling
import artemis/train
import artemis/evaluate
import helpers

proc trainToy(seed: int64): tuple[m: TransEModel, res: TrainResult,
                                  mrrBefore, hits3Before, mrrAfter,
                                  hits3After: float64] =
  var hp = HyperParams(dim: 10, epochs: 300, lr: 0.05'f32, norm: nkL1,
                       quiet: true, seed: seed)
  let (data, ents, rels, _) = loadDataset("data/toy_family.tsv", hp)
  var rng = initRand(seed)
  result.m = initModel(ents, rels, hp, rng)
  let truth = buildTruthSet(data.train)
  let before = evaluate(result.m, data.train, truth, nkL1)
  result.mrrBefore = before.mrr / before.n.float64
  result.hits3Before = before.hits3 / before.n.float64
  let sampler = initSampler(data.train, rels.len, ents.len, true, truth)
  result.res = train(result.m, data.train, sampler, hp)
  let after = evaluate(result.m, data.train, truth, nkL1)
  result.mrrAfter = after.mrr / after.n.float64
  result.hits3After = after.hits3 / after.n.float64

suite "train: toy graph convergence":
  test "loss decreases over training":
    let (_, res, _, _, _, _) = trainToy(42)
    check res.log.len == 300
    check res.log[^1].meanLoss < 0.5 * res.log[0].meanLoss

  test "ranking quality improves strongly (margin-loss objective)":
    # Note: on a tiny 8-entity graph the margin objective is satisfiable with
    # large relation vectors, so raw train distance is NOT the right signal;
    # ranking quality is. Observed seed-42 values: MRR 0.29 -> ~0.7,
    # raw Hits@3 0.31 -> ~0.9.
    let (_, _, mrrB, hits3B, mrrA, hits3A) = trainToy(42)
    check mrrA > 2.0 * mrrB
    check hits3A >= 0.7
    check hits3A > hits3B

  test "unit-norm invariant holds after training":
    let (m, _, _, _, _, _) = trainToy(42)
    let ep = m.entityEmb.storage.raw_buffer
    for e in 0 ..< m.nEntities:
      var sq = 0.0'f32
      for i in 0 ..< m.dim:
        sq += ep[e * m.dim + i] * ep[e * m.dim + i]
      check abs(sqrt(sq) - 1.0'f32) < 1e-4'f32

  test "training is deterministic under a fixed seed":
    let a = trainToy(42)
    let b = trainToy(42)
    check rawCopy(a.m.entityEmb) == rawCopy(b.m.entityEmb)
    check rawCopy(a.m.relEmb) == rawCopy(b.m.relEmb)

  test "different seeds differ":
    let a = trainToy(42)
    let b = trainToy(43)
    check rawCopy(a.m.entityEmb) != rawCopy(b.m.entityEmb)

suite "train: learning-rate schedule":
  test "constant schedule returns lr for every epoch":
    let hp = HyperParams(lr: 0.01'f32, lrSchedule: lsConstant, epochs: 10)
    for e in [1, 5, 10, 999]:
      check scheduledLr(hp, e) == hp.lr

  test "exponential schedule decays from lr and stops at lrMin":
    let hp = HyperParams(lr: 0.01'f32, lrSchedule: lsExponential,
                         lrDecay: 0.5'f32, lrMin: 0.002'f32, epochs: 10)
    check scheduledLr(hp, 1) == 0.01'f32
    check abs(scheduledLr(hp, 2) - 0.005'f32) < 1e-9'f32
    check abs(scheduledLr(hp, 3) - 0.0025'f32) < 1e-9'f32
    check scheduledLr(hp, 4) == 0.002'f32 # floor, not 0.00125
    check scheduledLr(hp, 50) == 0.002'f32

  test "linear schedule runs from lr to lrMin over the budget":
    let hp = HyperParams(lr: 0.01'f32, lrSchedule: lsLinear,
                         lrMin: 0.0'f32, epochs: 11)
    check scheduledLr(hp, 1) == 0.01'f32
    check abs(scheduledLr(hp, 6) - 0.005'f32) < 1e-9'f32 # midpoint
    check scheduledLr(hp, 11) == 0.0'f32
    let one = HyperParams(lr: 0.01'f32, lrSchedule: lsLinear, epochs: 1)
    check scheduledLr(one, 1) == 0.01'f32

  test "a decaying schedule still trains (loss falls, unit norm holds)":
    var hp = HyperParams(dim: 10, epochs: 200, lr: 0.05'f32, norm: nkL1,
                         quiet: true, seed: 42, lrSchedule: lsLinear,
                         lrMin: 0.0'f32)
    let (data, ents, rels, _) = loadDataset("data/toy_family.tsv", hp)
    var rng = initRand(42)
    var m = initModel(ents, rels, hp, rng)
    let truth = buildTruthSet(data.train)
    let before = evaluate(m, data.train, truth, nkL1).mrrMean
    let sampler = initSampler(data.train, rels.len, ents.len, true, truth)
    let res = train(m, data.train, sampler, hp)
    check res.log.len == 200
    check res.log[^1].meanLoss < 0.7 * res.log[0].meanLoss
    check evaluate(m, data.train, truth, nkL1).mrrMean > before
    let ep = m.entityEmb.storage.raw_buffer
    for e in 0 ..< m.nEntities:
      var sq = 0.0'f32
      for i in 0 ..< m.dim:
        sq += ep[e * m.dim + i] * ep[e * m.dim + i]
      check abs(sqrt(sq) - 1.0'f32) < 1e-4'f32

suite "train: weighted graphs":
  proc trainToyWeighted(seed: int64, weights: seq[float32],
                        mode: WeightMode, power = 1.0'f32): seq[float32] =
    var hp = HyperParams(dim: 10, epochs: 120, lr: 0.05'f32, norm: nkL1,
                         quiet: true, seed: seed, weightMode: mode,
                         weightPower: power)
    let (data, ents, rels, _) = loadDataset("data/toy_family.tsv", hp)
    var rng = initRand(seed)
    var m = initModel(ents, rels, hp, rng)
    let truth = buildTruthSet(data.train)
    let sampler = initSampler(data.train, rels.len, ents.len, true, truth)
    discard train(m, data.train, sampler, hp, weights = weights)
    rawCopy(m.entityEmb)

  test "all-ones weights are inactive in both modes (no silent behaviour change)":
    # Regression: filling `weights` with 1.0 used to route every file through
    # weighted *sampling*, which changed unweighted runs (and every documented
    # number) without saying so. Uniform weights must leave the classic
    # shuffled loop untouched, in sample mode as well as step mode.
    var unit = newSeq[float32](16)
    for i in 0 ..< unit.len: unit[i] = 1.0'f32
    for mode in [wmSample, wmStep]:
      check trainToyWeighted(42, @[], mode) == trainToyWeighted(42, unit, mode)
      # ... and so must `--weight-power 0`, whatever the weights are
      var mixed = newSeq[float32](16)
      for i in 0 ..< mixed.len: mixed[i] = (i mod 4).float32 + 0.25'f32
      check trainToyWeighted(42, @[], mode) ==
            trainToyWeighted(42, mixed, mode, power = 0.0'f32)

  test "step mode with all-ones weights is bit-identical to no weights":
    var unit = newSeq[float32](16)
    for i in 0 ..< unit.len: unit[i] = 1.0'f32
    check trainToyWeighted(42, @[], wmStep) ==
          trainToyWeighted(42, unit, wmStep)
    # a degenerate all-zero vector falls back to uniform instead of dividing by
    # zero (the parser rejects non-positive weights long before this)
    check trainToyWeighted(42, @[], wmStep) ==
          trainToyWeighted(42, newSeq[float32](16), wmStep)

  test "weight-power 0 neutralises any weight vector":
    var weights = newSeq[float32](16)
    for i in 0 ..< weights.len: weights[i] = (i mod 5).float32 + 0.5'f32
    check trainToyWeighted(42, @[], wmStep) ==
          trainToyWeighted(42, weights, wmStep, power = 0.0'f32)

  test "weights actually change the fit":
    var weights = newSeq[float32](16)
    for i in 0 ..< weights.len: weights[i] = if i mod 2 == 0: 5.0'f32 else: 0.2'f32
    check trainToyWeighted(42, @[], wmStep) !=
          trainToyWeighted(42, weights, wmStep)
    check trainToyWeighted(42, @[], wmSample) !=
          trainToyWeighted(42, weights, wmSample)

  test "weighted training is deterministic under a fixed seed":
    var weights = newSeq[float32](16)
    for i in 0 ..< weights.len: weights[i] = (i mod 3).float32 + 0.5'f32
    for mode in [wmStep, wmSample]:
      check trainToyWeighted(42, weights, mode) ==
            trainToyWeighted(42, weights, mode)

  proc pairDistances(w1, w2: float32, mode: WeightMode,
                     epochs: int): (float32, float32) =
    ## Two *disjoint* triples (no shared entity or relation, so they do not
    ## compete for the same vectors): the heavier edge should end up tighter.
    var hp = HyperParams(dim: 8, epochs: epochs, lr: 0.05'f32, seed: 7,
                         quiet: true, weightMode: mode)
    let (ents, rels) = (makeVocab(@["a", "b", "c", "d"]),
                        makeVocab(@["r1", "r2"]))
    var rng = initRand(7)
    var m = initModel(ents, rels, hp, rng)
    let data = @[Triple(head: 0, rel: 0, tail: 1),
                 Triple(head: 2, rel: 1, tail: 3)]
    let truth = buildTruthSet(data)
    let sampler = initSampler(data, rels.len, ents.len, false, truth)
    discard train(m, data, sampler, hp, weights = @[w1, w2])
    (distance(m, data[0], nkL1), distance(m, data[1], nkL1))

  test "the heavier edge ends up tighter (step mode)":
    let (d0Heavy, d1Light) = pairDistances(6.0'f32, 0.2'f32, wmStep, 150)
    let (d0Light, d1Heavy) = pairDistances(0.2'f32, 6.0'f32, wmStep, 150)
    check d0Heavy < d1Light # within the first run, the heavy edge is tighter
    check d1Heavy < d0Light # within the swapped run, likewise
    check d0Heavy < d0Light # and per edge: heavy beats light
    check d1Heavy < d1Light

  test "the heavier edge ends up tighter (sample mode)":
    let (d0Heavy, d1Light) = pairDistances(6.0'f32, 0.2'f32, wmSample, 150)
    let (d0Light, d1Heavy) = pairDistances(0.2'f32, 6.0'f32, wmSample, 150)
    check d0Heavy < d1Light
    check d1Heavy < d0Light
    check d0Heavy < d0Light
    check d1Heavy < d1Light

suite "train: self-adversarial negatives":
  test "alpha 0 is exactly uniform, whatever the distances":
    let w = adversarialWeights(@[1.0'f32, 5.0, 9.0], 0.0'f32)
    check w == @[1.0'f32, 1.0'f32, 1.0'f32]

  test "weights average one, so the total step size is unchanged":
    for alpha in [0.5'f32, 1.0'f32, 4.0'f32]:
      let w = adversarialWeights(@[1.0'f32, 5.0, 9.0], alpha)
      var total = 0.0'f32
      for x in w: total += x
      check abs(total - 3.0'f32) < 1e-4'f32

  test "harder negatives (smaller distance) get more weight":
    let w = adversarialWeights(@[1.0'f32, 5.0, 9.0], 1.0'f32)
    check w[0] > w[1] and w[1] > w[2]

  test "a larger alpha concentrates on the hardest negative":
    let w = adversarialWeights(@[1.0'f32, 5.0, 9.0], 1.0'f32)
    let sharp = adversarialWeights(@[1.0'f32, 5.0, 9.0], 6.0'f32)
    check sharp[0] > w[0]
    check sharp[2] < w[2]

  test "identical distances stay uniform (nothing to prefer)":
    check adversarialWeights(@[3.0'f32, 3.0, 3.0], 2.0'f32) ==
          @[1.0'f32, 1.0'f32, 1.0'f32]

  test "an empty group is empty":
    check adversarialWeights(@[], 1.0'f32).len == 0

  proc trainToyAdv(alpha: float32, negs: int): seq[float32] =
    var hp = HyperParams(dim: 10, epochs: 120, lr: 0.05'f32, norm: nkL1,
                         quiet: true, seed: 42, negPerPos: negs,
                         advAlpha: alpha)
    let (data, ents, rels, _) = loadDataset("data/toy_family.tsv", hp)
    var rng = initRand(42)
    var m = initModel(ents, rels, hp, rng)
    let truth = buildTruthSet(data.train)
    let sampler = initSampler(data.train, rels.len, ents.len, true, truth)
    discard train(m, data.train, sampler, hp)
    rawCopy(m.entityEmb)

  test "advAlpha 0 reproduces the plain loop bit for bit (even with 3 negatives)":
    check trainToyAdv(0.0'f32, 3) == trainToyAdv(0.0'f32, 3)
    var plain = HyperParams(dim: 10, epochs: 120, lr: 0.05'f32, norm: nkL1,
                            quiet: true, seed: 42, negPerPos: 3)
    let (data, ents, rels, _) = loadDataset("data/toy_family.tsv", plain)
    var rng = initRand(42)
    var m = initModel(ents, rels, plain, rng)
    let truth = buildTruthSet(data.train)
    let sampler = initSampler(data.train, rels.len, ents.len, true, truth)
    discard train(m, data.train, sampler, plain)
    check rawCopy(m.entityEmb) == trainToyAdv(0.0'f32, 3)

  test "a positive alpha changes the fit, deterministically":
    check trainToyAdv(1.0'f32, 3) != trainToyAdv(0.0'f32, 3)
    check trainToyAdv(1.0'f32, 3) == trainToyAdv(1.0'f32, 3)

  test "one negative leaves the weighting with nothing to do":
    check trainToyAdv(2.0'f32, 1) == trainToyAdv(0.0'f32, 1)

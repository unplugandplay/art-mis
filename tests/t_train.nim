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

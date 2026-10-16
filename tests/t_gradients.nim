import std/[unittest, random, math]
import artemis/types
import artemis/embeddings
import artemis/scoring
import artemis/gradients
import helpers

suite "gradients: closed-form vs finite differences":
  var rng = initRand(99)
  let hp = HyperParams(dim: 8, margin: 10.0'f32, norm: nkL2) # margin 10: loss > 0
  let (ents, rels) = (makeVocab(@["a", "b", "c", "d", "e"]),
                      makeVocab(@["r0", "r1"]))
  var m = initModel(ents, rels, hp, rng)
  let pos = Triple(head: 0, rel: 0, tail: 1)
  let neg = Triple(head: 0, rel: 0, tail: 3) # tail-corrupted
  var scratch = initScratch(hp.dim)

  test "computePair returns exact margin loss":
    let loss = computePair(m, pos, neg, hp.margin, hp.norm, scratch)
    let expected = hp.margin + distance(m, pos, nkL2) - distance(m, neg, nkL2)
    check loss > 0'f32
    check abs(loss - expected) < 1e-6'f32

  test "analytic gradients match finite differences":
    discard computePair(m, pos, neg, hp.margin, hp.norm, scratch)
    # seq assignment copies (value semantics) -> safe snapshots
    let gPos = scratch.gPos
    let gNeg = scratch.gNeg
    let ep = m.entityEmb.storage.raw_buffer
    const eps = 1e-3'f32
    for j in [0, 3, 7]:
      # neg shares pos's head (tail corruption): dL/dh = gPos - gNeg
      let orig = ep[pos.head * hp.dim + j]
      ep[pos.head * hp.dim + j] = orig + eps
      let lPlus = computePair(m, pos, neg, hp.margin, hp.norm, scratch)
      ep[pos.head * hp.dim + j] = orig - eps
      let lMinus = computePair(m, pos, neg, hp.margin, hp.norm, scratch)
      ep[pos.head * hp.dim + j] = orig
      let fd = (lPlus - lMinus) / (2 * eps)
      check abs(fd - (gPos[j] - gNeg[j])) < 1e-2'f32
      # dL/dt[j] = -gPos[j] (t appears only in the positive term)
      let origT = ep[pos.tail * hp.dim + j]
      ep[pos.tail * hp.dim + j] = origT + eps
      let ltPlus = computePair(m, pos, neg, hp.margin, hp.norm, scratch)
      ep[pos.tail * hp.dim + j] = origT - eps
      let ltMinus = computePair(m, pos, neg, hp.margin, hp.norm, scratch)
      ep[pos.tail * hp.dim + j] = origT
      let fdT = (ltPlus - ltMinus) / (2 * eps)
      check abs(fdT + gPos[j]) < 1e-2'f32
      # dL/dt_neg[j] = +gNeg[j] (t_neg appears only in the negative term)
      let origN = ep[neg.tail * hp.dim + j]
      ep[neg.tail * hp.dim + j] = origN + eps
      let lnPlus = computePair(m, pos, neg, hp.margin, hp.norm, scratch)
      ep[neg.tail * hp.dim + j] = origN - eps
      let lnMinus = computePair(m, pos, neg, hp.margin, hp.norm, scratch)
      ep[neg.tail * hp.dim + j] = origN
      let fdN = (lnPlus - lnMinus) / (2 * eps)
      check abs(fdN - gNeg[j]) < 1e-2'f32

  test "trainStep decreases the margin loss and keeps unit norm":
    var hp2 = HyperParams(dim: 8, margin: 1.0'f32, lr: 0.001'f32, norm: nkL2)
    let before = computePair(m, pos, neg, hp2.margin, hp2.norm, scratch)
    if before > 0:
      discard trainStep(m, pos, neg, hp2, scratch)
      let after = computePair(m, pos, neg, hp2.margin, hp2.norm, scratch)
      check after < before
    let ep2 = m.entityEmb.storage.raw_buffer
    for idx in [pos.head, pos.tail, neg.head, neg.tail]:
      var sq = 0.0'f32
      for i in 0 ..< hp.dim:
        sq += ep2[idx * hp.dim + i] * ep2[idx * hp.dim + i]
      check abs(sqrt(sq) - 1.0'f32) < 1e-5'f32

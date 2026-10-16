import std/[unittest, random, math]
import artemis/types
import artemis/embeddings
import helpers

suite "embeddings: init & normalization":
  test "deterministic init under fixed seed":
    var rng1 = initRand(42)
    var rng2 = initRand(42)
    let hp = HyperParams(dim: 8)
    let (ents, rels) = (makeVocab(@["a", "b", "c"]), makeVocab(@["r"]))
    let m1 = initModel(ents, rels, hp, rng1)
    let m2 = initModel(ents, rels, hp, rng2)
    check rawCopy(m1.entityEmb) == rawCopy(m2.entityEmb)
    check rawCopy(m1.relEmb) == rawCopy(m2.relEmb)

  test "entity rows are unit-norm after init":
    var rng = initRand(7)
    let hp = HyperParams(dim: 16)
    let (ents, rels) = (makeVocab(@["a", "b", "c", "d"]), makeVocab(@["r"]))
    let m = initModel(ents, rels, hp, rng)
    let ep = m.entityEmb.storage.raw_buffer
    for e in 0 ..< m.nEntities:
      var sq = 0.0'f32
      for i in 0 ..< hp.dim:
        sq += ep[e * hp.dim + i] * ep[e * hp.dim + i]
      check abs(sqrt(sq) - 1.0'f32) < 1e-5'f32

  test "normalizeEntityRow re-projects one row":
    var rng = initRand(7)
    let hp = HyperParams(dim: 4)
    let (ents, rels) = (makeVocab(@["a", "b"]), makeVocab(@["r"]))
    var m = initModel(ents, rels, hp, rng)
    let ep = m.entityEmb.storage.raw_buffer
    ep[0] = 3.0'f32 # blow up row 0
    m.normalizeEntityRow(0)
    var sq = 0.0'f32
    for i in 0 ..< hp.dim:
      sq += ep[i] * ep[i]
    check abs(sqrt(sq) - 1.0'f32) < 1e-6'f32

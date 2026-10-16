import std/[unittest, random, os, strutils, json, tables]
import artemis/types
import artemis/embeddings
import artemis/persist
import helpers

suite "persist: checkpoint round-trip":
  test "save/load is bit-exact and preserves vocab + hyperparams":
    var rng = initRand(1234)
    let hp = HyperParams(dim: 12, margin: 2.5'f32, lr: 0.007'f32,
                         norm: nkL2, bernoulli: false, seed: 1234)
    let (ents, rels) = (makeVocab(@["paris", "france", "tokyo", "japan"]),
                        makeVocab(@["capitalOf", "locatedIn"]))
    let m = initModel(ents, rels, hp, rng)
    let path = getTempDir() / "artemis_test_ckpt.bin"
    saveCheckpoint(m, hp, path)
    let (m2, hp2) = loadCheckpoint(path)
    check m2.entities.names == m.entities.names
    check m2.relations.names == m.relations.names
    check m2.entities.toId == m.entities.toId
    check m2.dim == hp.dim
    check hp2.margin == hp.margin
    check hp2.lr == hp.lr
    check hp2.norm == hp.norm
    check hp2.bernoulli == hp.bernoulli
    check hp2.seed == hp.seed
    check rawCopy(m2.entityEmb) == rawCopy(m.entityEmb)
    check rawCopy(m2.relEmb) == rawCopy(m.relEmb)

  test "rejects non-checkpoint files":
    let path = getTempDir() / "artemis_test_bad.bin"
    writeFile(path, "garbage garbage garbage")
    expect ValueError:
      discard loadCheckpoint(path)

suite "persist: TSV / JSON export":
  test "embeddings TSV has one line per entity and relation":
    var rng = initRand(5)
    let hp = HyperParams(dim: 3)
    let (ents, rels) = (makeVocab(@["a", "b"]), makeVocab(@["r"]))
    let m = initModel(ents, rels, hp, rng)
    let path = getTempDir() / "artemis_test_emb.tsv"
    exportEmbeddingsTsv(m, path)
    let lines = readFile(path).strip.splitLines
    check lines.len == 3 # 2 entities + 1 relation
    let f0 = lines[0].split('\t')
    check f0[0] == "a"
    check f0[1].split(' ').len == 3
    for tok in f0[1].split(' '):
      discard parseFloat(tok) # raises if not a float
    check lines[2].startsWith("REL/r\t")

  test "metadata JSON parses and contains vocab":
    var rng = initRand(5)
    let hp = HyperParams(dim: 3, seed: 5)
    let (ents, rels) = (makeVocab(@["a", "b"]), makeVocab(@["r"]))
    let m = initModel(ents, rels, hp, rng)
    let path = getTempDir() / "artemis_test_meta.json"
    exportMetaJson(m, hp, nil, path)
    let j = parseJson(readFile(path))
    check j["dim"].getInt == 3
    check j["entities"].getElems.len == 2
    check j["hyperparams"]["seed"].getInt == 5

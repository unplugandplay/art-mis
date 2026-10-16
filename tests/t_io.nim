import std/[unittest, options, os, tables]
import artemis/types
import artemis/io
import helpers

suite "io: parsing":
  test "parseLine happy path":
    let t = parseLine("alice\tparentOf\tbob")
    check t.isSome
    check t.get.head == "alice"
    check t.get.relation == "parentOf"
    check t.get.tail == "bob"

  test "parseLine rejects malformed input":
    check parseLine("").isNone
    check parseLine("# comment").isNone
    check parseLine("only\ttwo").isNone
    check parseLine("a\t\tb").isNone        # empty relation field
    check parseLine("a\tb\tc\td").isNone   # 4th field is not a number

  test "loadRawTriples counts malformed lines":
    let path = getTempDir() / "artemis_t_io_triples.tsv"
    writeFile(path, "a\tr\tb\nb\tr\tc\nbroken\n\nx\tr\ty\n")
    var malformed = 0
    let raws = loadRawTriples(path, '\t', malformed)
    check raws.len == 3
    check malformed == 2 # "broken" + empty line

suite "io: vocabulary":
  test "first-appearance id order + round trip":
    let raws = @[rt("b", "r1", "a"), rt("a", "r2", "c"), rt("b", "r1", "c")]
    var filtered = 0
    let (ents, rels) = buildVocab(raws, 1, filtered)
    check filtered == 0
    check ents.names == @["b", "a", "c"] # first appearance
    check rels.names == @["r1", "r2"]
    for n in ents.names:
      check ents.names[ents.toId[n]] == n

  test "min-count filter drops rare names":
    let raws = @[rt("a", "r", "b"), rt("a", "r", "c"), rt("b", "r", "c"),
                 rt("rare", "r", "a")]
    var filtered = 0
    let (ents, _) = buildVocab(raws, 2, filtered)
    check filtered == 1
    check "rare" notin ents.names

  test "toTriples skips unknown names":
    let (ents, rels) = (makeVocab(@["a", "b"]), makeVocab(@["r"]))
    var skipped = 0
    let ts = toTriples(@[rt("a", "r", "b"), rt("a", "r", "zzz")], ents, rels,
                       skipped)
    check ts == @[Triple(head: 0, rel: 0, tail: 1)]
    check skipped == 1

  test "toTriples collects one weight per kept triple":
    let (ents, rels) = (makeVocab(@["a", "b", "c"]), makeVocab(@["r"]))
    var skipped = 0
    var weights: seq[float32]
    let ts = toTriples(@[rt("a", "r", "b", 3.0'f32), rt("a", "r", "zzz"),
                         rt("b", "r", "c", 0.25'f32)],
                       ents, rels, skipped, weights)
    check ts.len == 2
    check weights == @[3.0'f32, 0.25'f32] # the skipped triple contributes none

suite "io: weights":
  test "parseLine reads an optional weight column":
    let t = parseLine("a\tr\tb\t2.5")
    check t.isSome
    check t.get.weight == 2.5'f32
    check parseLine("a\tr\tb").get.weight == 1.0'f32 # default when absent
    check parseLine("a\tr\tb\t1e-3").get.weight == 0.001'f32

  test "a bad weight makes the line malformed":
    check parseLine("a\tr\tb\t0").isNone      # weights must be positive
    check parseLine("a\tr\tb\t-1").isNone
    check parseLine("a\tr\tb\tNaN").isNone
    check parseLine("a\tr\tb\tinf").isNone
    check parseLine("a\tr\tb\t1.0\textra").isNone
    check parseLine("a\tr\tb\theavy").isNone

  test "loadDataset aligns weights with the training triples it keeps":
    let path = getTempDir() / "artemis_t_io_weighted.tsv"
    writeFile(path, "a\tr\tb\t4\nb\tr\tc\nrare\tr\ta\t9\n")
    let (data, ents, rels, report) = loadDataset(path, HyperParams())
    check data.train.len == 3
    check data.weights == @[4.0'f32, 1.0'f32, 9.0'f32] # missing weight -> 1.0
    check report.weightedTrain == 2
    check ents.len == 4 and rels.len == 1 # a, b, c, rare
    check report.malformedTrain == 0

  test "an unweighted file yields an all-ones weight vector":
    let path = getTempDir() / "artemis_t_io_plain.tsv"
    writeFile(path, "a\tr\tb\nb\tr\tc\n")
    let (data, _, _, report) = loadDataset(path, HyperParams())
    check data.weights == @[1.0'f32, 1.0'f32]
    check report.weightedTrain == 0 # defaulted 1.0 is not an explicit weight

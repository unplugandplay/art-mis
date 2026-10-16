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
    check parseLine("a\tb\tc\td").isNone   # too many fields

  test "loadRawTriples counts malformed lines":
    let path = getTempDir() / "artemis_t_io_triples.tsv"
    writeFile(path, "a\tr\tb\nb\tr\tc\nbroken\n\nx\tr\ty\n")
    var malformed = 0
    let raws = loadRawTriples(path, '\t', malformed)
    check raws.len == 3
    check malformed == 2 # "broken" + empty line

suite "io: vocabulary":
  test "first-appearance id order + round trip":
    let raws = @[("b", "r1", "a"), ("a", "r2", "c"), ("b", "r1", "c")]
    var filtered = 0
    let (ents, rels) = buildVocab(raws, 1, filtered)
    check filtered == 0
    check ents.names == @["b", "a", "c"] # first appearance
    check rels.names == @["r1", "r2"]
    for n in ents.names:
      check ents.names[ents.toId[n]] == n

  test "min-count filter drops rare names":
    let raws = @[("a", "r", "b"), ("a", "r", "c"), ("b", "r", "c"),
                 ("rare", "r", "a")]
    var filtered = 0
    let (ents, _) = buildVocab(raws, 2, filtered)
    check filtered == 1
    check "rare" notin ents.names

  test "toTriples skips unknown names":
    let (ents, rels) = (makeVocab(@["a", "b"]), makeVocab(@["r"]))
    var skipped = 0
    let ts = toTriples(@[("a", "r", "b"), ("a", "r", "zzz")], ents, rels,
                       skipped)
    check ts == @[Triple(head: 0, rel: 0, tail: 1)]
    check skipped == 1

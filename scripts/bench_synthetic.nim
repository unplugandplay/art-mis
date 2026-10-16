## Synthetic-scale benchmark: trains on random triples to measure raw
## throughput (sec/epoch, triples/sec) and peak memory.
##
## Usage: nim r -d:release scripts/bench_synthetic.nim [nEnt nRel nTriples dim epochs]
import std/[random, times, strutils, strformat, sets, os, tables]
import artemis/types
import artemis/embeddings
import artemis/sampling
import artemis/train

let nEnt = if paramCount() > 0: parseInt(paramStr(1)) else: 15000
let nRel = if paramCount() > 1: parseInt(paramStr(2)) else: 200
let nTriples = if paramCount() > 2: parseInt(paramStr(3)) else: 500000
let dim = if paramCount() > 3: parseInt(paramStr(4)) else: 50
let epochs = if paramCount() > 4: parseInt(paramStr(5)) else: 3

var rng = initRand(1)
var entNames = newSeq[string](nEnt)
var relNames = newSeq[string](nRel)
var ents, rels: Vocab
ents.toId = initTable[string, int](nEnt)
rels.toId = initTable[string, int](nRel)
for i in 0 ..< nEnt:
  entNames[i] = "e" & $i
  ents.toId[entNames[i]] = i
for i in 0 ..< nRel:
  relNames[i] = "r" & $i
  rels.toId[relNames[i]] = i
ents.names = entNames
rels.names = relNames

var triples = newSeq[Triple](nTriples)
for i in 0 ..< nTriples:
  triples[i] = Triple(head: rng.rand(nEnt - 1), rel: rng.rand(nRel - 1),
                      tail: rng.rand(nEnt - 1))

let hp = HyperParams(dim: dim, epochs: epochs, batchSize: 100, quiet: true,
                     seed: 42)
var m = initModel(ents, rels, hp, rng)
let truth = triples.toHashSet
let sampler = initSampler(triples, nRel, nEnt, true, truth)

echo &"benchmark: {nEnt} entities, {nRel} relations, {nTriples} triples, dim={dim}, {epochs} epochs"
let t0 = epochTime()
discard train(m, triples, sampler, hp)
let dt = epochTime() - t0
echo &"total: {dt:.2f}s  ->  {dt / epochs.float64:.3f}s/epoch, {epochs.float64 * nTriples.float64 / dt:.0f} triples/s"

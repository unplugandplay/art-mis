## Persistence: binary checkpoints, embedding TSV, metadata JSON.
##
## Checkpoint layout (native little-endian, versioned):
##   magic      8 bytes   "ARTECKPT"
##   version    uint32    = 1
##   dim        uint32
##   nEnt       uint32
##   nRel       uint32
##   norm       uint8     0 = L1, 1 = L2
##   bernoulli  uint8
##   margin     float32
##   lr         float32
##   seed       int64
##   entity names: nEnt x (uint32 len, bytes)
##   relation names: nRel x (uint32 len, bytes)
##   entity embeddings: nEnt*dim float32, row-major
##   relation embeddings: nRel*dim float32, row-major
import std/[streams, json, strutils, tables]
import arraymancer
import ./types

const
  CheckpointMagic* = "ARTECKPT"
  CheckpointVersion* = 1'u32

proc saveCheckpoint*(m: TransEModel, hp: HyperParams, path: string) =
  let fs = openFileStream(path, fmWrite)
  defer: fs.close()
  fs.write(CheckpointMagic)
  var v = CheckpointVersion
  fs.writeData(addr v, sizeof v)
  var dimU = m.dim.uint32
  var nEntU = m.nEntities.uint32
  var nRelU = m.nRelations.uint32
  fs.writeData(addr dimU, sizeof dimU)
  fs.writeData(addr nEntU, sizeof nEntU)
  fs.writeData(addr nRelU, sizeof nRelU)
  var norm = hp.norm.uint8
  var bern = hp.bernoulli.uint8
  var margin = hp.margin
  var lr = hp.lr
  var seed = hp.seed
  fs.writeData(addr norm, sizeof norm)
  fs.writeData(addr bern, sizeof bern)
  fs.writeData(addr margin, sizeof margin)
  fs.writeData(addr lr, sizeof lr)
  fs.writeData(addr seed, sizeof seed)
  for name in m.entities.names:
    var n = name.len.uint32
    fs.writeData(addr n, sizeof n)
    if n > 0: fs.writeData(unsafeAddr name[0], n.int)
  for name in m.relations.names:
    var n = name.len.uint32
    fs.writeData(addr n, sizeof n)
    if n > 0: fs.writeData(unsafeAddr name[0], n.int)
  let ep = m.entityEmb.storage.raw_buffer
  let rp = m.relEmb.storage.raw_buffer
  fs.writeData(ep, m.nEntities * m.dim * sizeof(float32))
  fs.writeData(rp, m.nRelations * m.dim * sizeof(float32))

proc readName(fs: FileStream): string =
  var n: uint32
  discard fs.readData(addr n, sizeof n)
  result = newString(n.int)
  if n > 0:
    discard fs.readData(addr result[0], n.int)

proc loadCheckpoint*(path: string): tuple[m: TransEModel, hp: HyperParams] =
  let fs = openFileStream(path, fmRead)
  defer: fs.close()
  var magic = newString(8)
  discard fs.readData(addr magic[0], 8)
  if magic != CheckpointMagic:
    raise newException(ValueError, "not an Artemis checkpoint: " & path)
  var v, dimU, nEntU, nRelU: uint32
  discard fs.readData(addr v, sizeof v)
  if v != CheckpointVersion:
    raise newException(ValueError, "unsupported checkpoint version: " & $v)
  discard fs.readData(addr dimU, sizeof dimU)
  discard fs.readData(addr nEntU, sizeof nEntU)
  discard fs.readData(addr nRelU, sizeof nRelU)
  var norm, bern: uint8
  var margin, lr: float32
  var seed: int64
  discard fs.readData(addr norm, sizeof norm)
  discard fs.readData(addr bern, sizeof bern)
  discard fs.readData(addr margin, sizeof margin)
  discard fs.readData(addr lr, sizeof lr)
  discard fs.readData(addr seed, sizeof seed)
  var entities, relations: Vocab
  entities.names = newSeq[string](nEntU.int)
  entities.toId = initTable[string, int](nEntU.int)
  for i in 0 ..< nEntU.int:
    entities.names[i] = fs.readName()
    entities.toId[entities.names[i]] = i
  relations.names = newSeq[string](nRelU.int)
  relations.toId = initTable[string, int](nRelU.int)
  for i in 0 ..< nRelU.int:
    relations.names[i] = fs.readName()
    relations.toId[relations.names[i]] = i
  result.m.entities = entities
  result.m.relations = relations
  result.m.entityEmb = newTensor[float32](nEntU.int, dimU.int)
  result.m.relEmb = newTensor[float32](nRelU.int, dimU.int)
  discard fs.readData(result.m.entityEmb.storage.raw_buffer,
                      nEntU.int * dimU.int * sizeof(float32))
  discard fs.readData(result.m.relEmb.storage.raw_buffer,
                      nRelU.int * dimU.int * sizeof(float32))
  result.hp = HyperParams(dim: dimU.int, margin: margin, lr: lr,
                          norm: NormKind(norm.int), bernoulli: bern != 0,
                          seed: seed)

proc exportEmbeddingsTsv*(m: TransEModel, path: string) =
  ## One line per vector: `<name>\t<dim floats, space-separated>`.
  ## Entities first, then relations (relation names prefixed with "REL/").
  let dim = m.dim
  let ep = m.entityEmb.storage.raw_buffer
  let rp = m.relEmb.storage.raw_buffer
  var f = open(path, fmWrite)
  defer: f.close()
  for e in 0 ..< m.nEntities:
    var parts = newSeqOfCap[string](dim)
    for i in 0 ..< dim:
      parts.add $ep[e * dim + i]
    f.writeLine(m.entities.names[e] & "\t" & parts.join(" "))
  for r in 0 ..< m.nRelations:
    var parts = newSeqOfCap[string](dim)
    for i in 0 ..< dim:
      parts.add $rp[r * dim + i]
    f.writeLine("REL/" & m.relations.names[r] & "\t" & parts.join(" "))

proc exportMetaJson*(m: TransEModel, hp: HyperParams, metrics: JsonNode,
                     path: string) =
  var root = %*{
    "format": "artemis-meta",
    "version": 1,
    "dim": m.dim,
    "nEntities": m.nEntities,
    "nRelations": m.nRelations,
    "hyperparams": {
      "margin": hp.margin, "lr": hp.lr, "epochs": hp.epochs,
      "lrSchedule": $hp.lrSchedule, "lrDecay": hp.lrDecay, "lrMin": hp.lrMin,
      "batchSize": hp.batchSize, "negPerPos": hp.negPerPos,
      "norm": $hp.norm, "bernoulli": hp.bernoulli, "seed": hp.seed,
      "weightMode": $hp.weightMode, "weightPower": hp.weightPower,
      "weightNorm": $hp.weightNorm, "useWeights": hp.useWeights,
    },
    "entities": m.entities.names,
    "relations": m.relations.names,
  }
  if metrics != nil:
    root["metrics"] = metrics
  writeFile(path, root.pretty() & "\n")

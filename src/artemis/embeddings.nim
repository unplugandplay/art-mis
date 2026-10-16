## Embedding storage: initialization and the unit-norm constraint.
##
## Hot-path code accesses rows through `storage.raw_buffer`
## (a `ptr UncheckedArray[float32]`), row-major, row `i` at `i*dim`.
import std/[random, math]
import arraymancer
import ./types

proc initModel*(entities, relations: Vocab, hp: HyperParams,
                rng: var Rand): TransEModel =
  ## Uniform init in [-6/sqrt(dim), 6/sqrt(dim)] (paper default), then
  ## project entity rows onto the unit sphere.
  result.entities = entities
  result.relations = relations
  result.entityEmb = newTensor[float32](entities.len, hp.dim)
  result.relEmb = newTensor[float32](relations.len, hp.dim)
  let bound = 6.0'f32 / sqrt(hp.dim.float32)
  let ep = result.entityEmb.storage.raw_buffer
  for i in 0 ..< entities.len * hp.dim:
    ep[i] = (rng.rand(1.0) * 2.0 - 1.0).float32 * bound
  let rp = result.relEmb.storage.raw_buffer
  for i in 0 ..< relations.len * hp.dim:
    rp[i] = (rng.rand(1.0) * 2.0 - 1.0).float32 * bound
  # Unit-norm constraint applies to entities from step 0.
  let dim = hp.dim
  for e in 0 ..< entities.len:
    var sq = 0.0'f32
    for i in 0 ..< dim:
      sq += ep[e * dim + i] * ep[e * dim + i]
    let n = sqrt(sq)
    if n > 1e-12'f32:
      for i in 0 ..< dim:
        ep[e * dim + i] /= n

proc normalizeEntities*(m: var TransEModel) =
  ## Project every entity row onto the unit L2 sphere (paper constraint).
  let dim = m.dim
  let p = m.entityEmb.storage.raw_buffer
  for e in 0 ..< m.nEntities:
    var sq = 0.0'f32
    for i in 0 ..< dim:
      sq += p[e * dim + i] * p[e * dim + i]
    let n = sqrt(sq)
    if n > 1e-12'f32:
      for i in 0 ..< dim:
        p[e * dim + i] /= n

proc normalizeEntityRow*(m: var TransEModel, idx: int) {.inline.} =
  ## Post-update single-row unit-norm projection.
  let dim = m.dim
  let p = m.entityEmb.storage.raw_buffer
  let base = idx * dim
  var sq = 0.0'f32
  for i in 0 ..< dim:
    sq += p[base + i] * p[base + i]
  let n = sqrt(sq)
  if n > 1e-12'f32:
    for i in 0 ..< dim:
      p[base + i] /= n

proc entityRow*(m: TransEModel, idx: int): seq[float32] =
  ## Copy of entity row `idx` (tests / export / inspection).
  let dim = m.dim
  let p = m.entityEmb.storage.raw_buffer
  result = newSeq[float32](dim)
  for i in 0 ..< dim:
    result[i] = p[idx * dim + i]

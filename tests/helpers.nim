## Shared test helpers.
import std/tables
import arraymancer
import artemis/types

proc rt*(head, rel, tail: string, weight = 1.0'f32): RawTriple =
  ## Terse constructor for a parsed triple, with an optional weight.
  (head: head, relation: rel, tail: tail, weight: weight)

proc makeVocab*(names: seq[string]): Vocab =
  result.names = names
  for i, n in names:
    result.toId[n] = i

proc modelFromArrays*(entNames, relNames: seq[string], dim: int,
                      ent, rel: seq[float32]): TransEModel =
  ## Build a model with explicit embedding values (row-major flat arrays).
  doAssert ent.len == entNames.len * dim
  doAssert rel.len == relNames.len * dim
  result.entities = makeVocab(entNames)
  result.relations = makeVocab(relNames)
  result.entityEmb = newTensor[float32](entNames.len, dim)
  result.relEmb = newTensor[float32](relNames.len, dim)
  let ep = result.entityEmb.storage.raw_buffer
  for i in 0 ..< ent.len:
    ep[i] = ent[i]
  let rp = result.relEmb.storage.raw_buffer
  for i in 0 ..< rel.len:
    rp[i] = rel[i]

proc rawCopy*(t: Tensor[float32]): seq[float32] =
  let n = t.size
  let p = t.storage.raw_buffer
  result = newSeq[float32](n)
  for i in 0 ..< n:
    result[i] = p[i]

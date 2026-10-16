## TransE dissimilarity scoring: d(h, r, t) = ||h + r - t||_p.
##
## Fused, allocation-free loops over raw row-major storage — the hot path.
## (The coding plan allowed Arraymancer broadcasting for batch scoring; fused
## loops won: same complexity, zero temporaries. See README design notes.)
##
## `EmbeddingView` is a GC-free view of the two matrices plus the cost norm;
## it is cheap to copy and safe to share with worker threads. `ScoreBoard`
## carries the per-worker scratch buffers, including the loop-invariant
## `h + r` term hoisted out of the candidate loop (exact same rounding order,
## one addition less per candidate element).
import std/math
import ./types

type
  EmbeddingView* = object
    ## Raw, GC-free view of the embedding matrices + the dissimilarity norm.
    ## Contains no managed memory: safe to hand to a worker thread.
    ep*, rp*: ptr UncheckedArray[float32]
    dim*, nEnt*, nRel*: int
    norm*: NormKind

  ScoreBoard* = object
    ## Per-worker scratch; allocate once per thread, reuse for every triple.
    scores*: seq[float32] ## d(., ., e) for every entity e
    left*: seq[float32]   ## candidate-independent term (h + r for tail scoring)

proc view*(m: TransEModel): EmbeddingView {.inline.} =
  ## Build a raw view of `m` (norm defaults to L1; set `v.norm` explicitly).
  EmbeddingView(ep: m.entityEmb.storage.raw_buffer,
                rp: m.relEmb.storage.raw_buffer,
                dim: m.dim, nEnt: m.nEntities, nRel: m.nRelations)

proc initScoreBoard*(v: EmbeddingView): ScoreBoard =
  ScoreBoard(scores: newSeq[float32](v.nEnt), left: newSeq[float32](v.dim))

proc reserve(sb: var ScoreBoard, v: EmbeddingView) {.inline.} =
  if sb.scores.len != v.nEnt: sb.scores.setLen(v.nEnt)
  if sb.left.len != v.dim: sb.left.setLen(v.dim)

proc distance*(v: EmbeddingView, h, r, t: int): float32 {.inline.} =
  ## ||v_h + v_r - v_t||_p for explicit row ids.
  let dim = v.dim
  let ep = v.ep
  let rp = v.rp
  let hb = h * dim
  let rb = r * dim
  let tb = t * dim
  case v.norm
  of nkL1:
    var s = 0.0'f32
    for i in 0 ..< dim:
      s += abs(ep[hb + i] + rp[rb + i] - ep[tb + i])
    s
  of nkL2:
    var s = 0.0'f32
    for i in 0 ..< dim:
      let x = ep[hb + i] + rp[rb + i] - ep[tb + i]
      s += x * x
    sqrt(s)

proc distance*(v: EmbeddingView, tr: Triple): float32 {.inline.} =
  distance(v, tr.head, tr.rel, tr.tail)

proc distance*(m: TransEModel, h, r, t: int, norm: NormKind): float32 {.inline.} =
  var v = view(m)
  v.norm = norm
  distance(v, h, r, t)

proc distance*(m: TransEModel, tr: Triple, norm: NormKind): float32 {.inline.} =
  distance(m, tr.head, tr.rel, tr.tail, norm)

proc scoreAllTails*(v: EmbeddingView, sb: var ScoreBoard, h, r: int) =
  ## sb.scores[e] = d(h, r, e) for every entity e. Used by evaluation.
  ## `sb.left` holds h + r, so the candidate loop is one subtract + abs per
  ## element — and the rounding order matches `distance` exactly.
  reserve(sb, v)
  let dim = v.dim
  let ep = v.ep
  let rp = v.rp
  let hb = h * dim
  let rb = r * dim
  case v.norm
  of nkL1:
    for i in 0 ..< dim:
      sb.left[i] = ep[hb + i] + rp[rb + i]
    for e in 0 ..< v.nEnt:
      let eb = e * dim
      var s = 0.0'f32
      for i in 0 ..< dim:
        s += abs(sb.left[i] - ep[eb + i])
      sb.scores[e] = s
  of nkL2:
    for i in 0 ..< dim:
      sb.left[i] = ep[hb + i] + rp[rb + i]
    for e in 0 ..< v.nEnt:
      let eb = e * dim
      var s = 0.0'f32
      for i in 0 ..< dim:
        let x = sb.left[i] - ep[eb + i]
        s += x * x
      sb.scores[e] = sqrt(s)

proc scoreAllHeads*(v: EmbeddingView, sb: var ScoreBoard, r, t: int) =
  ## sb.scores[e] = d(e, r, t) for every entity e. Used by evaluation.
  ## The head side is deliberately *not* hoisted: `(e + r) - t` must keep the
  ## original association, since `e + (r - t)` rounds differently.
  reserve(sb, v)
  let dim = v.dim
  let ep = v.ep
  let rp = v.rp
  let rb = r * dim
  let tb = t * dim
  case v.norm
  of nkL1:
    for e in 0 ..< v.nEnt:
      let eb = e * dim
      var s = 0.0'f32
      for i in 0 ..< dim:
        s += abs(ep[eb + i] + rp[rb + i] - ep[tb + i])
      sb.scores[e] = s
  of nkL2:
    for e in 0 ..< v.nEnt:
      let eb = e * dim
      var s = 0.0'f32
      for i in 0 ..< dim:
        let x = ep[eb + i] + rp[rb + i] - ep[tb + i]
        s += x * x
      sb.scores[e] = sqrt(s)

proc scoreAllTails*(m: TransEModel, h, r: int, norm: NormKind,
                    outScores: var seq[float32]) =
  ## Convenience wrapper: outScores[e] = d(h, r, e). Reuses `outScores` as
  ## scratch; the hoisted `h + r` term is allocated per call, so hot loops
  ## should hold a `ScoreBoard` instead.
  var v = view(m)
  v.norm = norm
  var sb = ScoreBoard(scores: outScores, left: newSeq[float32](v.dim))
  scoreAllTails(v, sb, h, r)
  outScores = sb.scores

proc scoreAllHeads*(m: TransEModel, r, t: int, norm: NormKind,
                    outScores: var seq[float32]) =
  ## Convenience wrapper: outScores[e] = d(e, r, t).
  var v = view(m)
  v.norm = norm
  var sb = ScoreBoard(scores: outScores, left: newSeq[float32](v.dim))
  scoreAllHeads(v, sb, r, t)
  outScores = sb.scores

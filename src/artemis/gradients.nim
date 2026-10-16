## Closed-form TransE gradients and the SGD update step.
##
## For x = h + r - t, d = ||x||_p:
##   L2: grad = x / ||x||   (0 when ||x|| == 0)
##   L1: grad = sign(x)
## Margin loss L = max(0, margin + d_pos - d_neg), when L > 0:
##   h  <- h  - lr * g_pos      r  <- r  - lr * g_pos      t  <- t  + lr * g_pos
##   h' <- h' + lr * g_neg      r' <- r' + lr * g_neg      t' <- t' - lr * g_neg
## (r == r' always; one of h == h' or t == t' depending on corrupted side.)
## Updated entity rows are re-projected onto the unit sphere.
import std/math
import ./types
import ./embeddings

type
  StepScratch* = object
    ## Preallocated per-thread/per-epoch buffers; reused across steps so the
    ## hot loop performs zero allocations.
    xPos*, xNeg*, gPos*, gNeg*: seq[float32]

proc initScratch*(dim: int): StepScratch =
  StepScratch(
    xPos: newSeq[float32](dim),
    xNeg: newSeq[float32](dim),
    gPos: newSeq[float32](dim),
    gNeg: newSeq[float32](dim),
  )

proc gradOf(x: openArray[float32], d: float32, norm: NormKind,
            g: var openArray[float32]) {.inline.} =
  case norm
  of nkL1:
    for i in 0 ..< x.len:
      g[i] =
        if x[i] > 0'f32: 1.0'f32
        elif x[i] < 0'f32: -1.0'f32
        else: 0.0'f32
  of nkL2:
    if d > 1e-12'f32:
      for i in 0 ..< x.len:
        g[i] = x[i] / d
    else:
      for i in 0 ..< x.len:
        g[i] = 0.0'f32

proc computePair*(m: TransEModel, pos, neg: Triple, margin: float32,
                  norm: NormKind, s: var StepScratch): float32 =
  ## Fill scratch buffers with x_pos/x_neg and their gradients; return the
  ## margin loss (0 means "no update needed").
  let dim = m.dim
  let ep = m.entityEmb.storage.raw_buffer
  let rp = m.relEmb.storage.raw_buffer
  let hb = pos.head * dim
  let rb = pos.rel * dim
  let tb = pos.tail * dim
  let hbn = neg.head * dim
  let tbn = neg.tail * dim
  var dPos, dNeg = 0.0'f32
  case norm
  of nkL1:
    for i in 0 ..< dim:
      s.xPos[i] = ep[hb + i] + rp[rb + i] - ep[tb + i]
      s.xNeg[i] = ep[hbn + i] + rp[rb + i] - ep[tbn + i]
      dPos += abs(s.xPos[i])
      dNeg += abs(s.xNeg[i])
  of nkL2:
    var sqPos, sqNeg = 0.0'f32
    for i in 0 ..< dim:
      s.xPos[i] = ep[hb + i] + rp[rb + i] - ep[tb + i]
      s.xNeg[i] = ep[hbn + i] + rp[rb + i] - ep[tbn + i]
      sqPos += s.xPos[i] * s.xPos[i]
      sqNeg += s.xNeg[i] * s.xNeg[i]
    dPos = sqrt(sqPos)
    dNeg = sqrt(sqNeg)
  result = margin + dPos - dNeg
  if result > 0'f32:
    gradOf(s.xPos, dPos, norm, s.gPos)
    gradOf(s.xNeg, dNeg, norm, s.gNeg)
  else:
    result = 0.0'f32

proc trainStep*(m: var TransEModel, pos, neg: Triple, hp: HyperParams,
                s: var StepScratch): float32 =
  ## One SGD step on one (positive, negative) pair. Returns the loss.
  result = computePair(m, pos, neg, hp.margin, hp.norm, s)
  if result <= 0'f32:
    return
  let dim = m.dim
  let lr = hp.lr
  let ep = m.entityEmb.storage.raw_buffer
  let rp = m.relEmb.storage.raw_buffer
  let hb = pos.head * dim
  let rb = pos.rel * dim
  let tb = pos.tail * dim
  let hbn = neg.head * dim
  let tbn = neg.tail * dim
  # Positive side: pull h, r, t together.
  # Negative side: push them apart. Relation row gets both updates.
  for i in 0 ..< dim:
    let gp = lr * s.gPos[i]
    let gn = lr * s.gNeg[i]
    rp[rb + i] += gn - gp
    ep[hb + i] -= gp
    ep[tb + i] += gp
    ep[hbn + i] += gn
    ep[tbn + i] -= gn
  # Unit-norm constraint on every touched entity row (idempotent on
  # duplicates, e.g. neg.head == pos.head when the tail was corrupted).
  m.normalizeEntityRow(pos.head)
  m.normalizeEntityRow(pos.tail)
  m.normalizeEntityRow(neg.head)
  m.normalizeEntityRow(neg.tail)

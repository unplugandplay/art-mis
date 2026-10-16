## Training loop: shuffled epochs, mini-batches, checkpoints, early stopping.
import std/[random, sets, times, strformat, os, math]
import arraymancer
import ./types
import ./gradients
import ./sampling
import ./evaluate
import ./scoring
import ./persist

proc rawCopySeq(t: Tensor[float32]): seq[float32] =
  ## Contiguous copy of a tensor's storage (the deprecated `toRawSeq`).
  let n = t.size
  result = newSeq[float32](n)
  if n > 0:
    copyMem(addr result[0], t.storage.raw_buffer, n * sizeof(float32))

proc weightedDraw(rng: var Rand, cumulative: openArray[float64],
                  total: float64): int =
  ## Index of the edge drawn with probability proportional to its weight; the
  ## table is the running sum, so this is a binary search on a uniform draw.
  var u = rng.rand(total)
  var lo = 0
  var hi = cumulative.len - 1
  while lo < hi:
    let mid = (lo + hi) div 2
    if cumulative[mid] <= u: lo = mid + 1
    else: hi = mid
  result = lo

proc adversarialWeights*(dists: openArray[float32],
                         alpha: float32): seq[float32] =
  ## RotatE-style self-adversarial weights for the negatives drawn for one
  ## positive: harder negatives (smaller distance) get more of the push.
  ##
  ## RotatE exponentiates the raw score, but our dissimilarity is unbounded (a
  ## relation vector can inflate every distance), so the group is standardised
  ## first and `alpha` is dimensionless: 0 is exactly uniform, larger values
  ## concentrate on the hardest of the sampled negatives. The weights average
  ## 1, so the total step size matches the unweighted loop.
  let n = dists.len
  result = newSeq[float32](n)
  if n == 0: return
  for i in 0 ..< n: result[i] = 1.0'f32
  if alpha <= 0.0'f32: return
  var mean = 0.0'f32
  for d in dists: mean += d
  mean /= n.float32
  var varSum = 0.0'f32
  for d in dists: varSum += (d - mean) * (d - mean)
  let sd = sqrt(varSum / n.float32)
  if sd <= 1e-6'f32: return # all negatives equally hard: stay uniform
  var total = 0.0'f32
  for i in 0 ..< n:
    result[i] = exp(-alpha * (dists[i] - mean) / sd)
    total += result[i]
  if total <= 0.0'f32:
    for i in 0 ..< n: result[i] = 1.0'f32
    return
  for i in 0 ..< n: result[i] = result[i] * n.float32 / total

proc weightFactors(hp: HyperParams, weights: seq[float32]): seq[float32] =
  ## Per-edge multiplier for `wmStep`: weights are raised to `hp.weightPower`
  ## and divided by their mean (or max), so an all-equal file behaves exactly
  ## like an unweighted run and `--lr` keeps its meaning.
  if weights.len == 0: return
  result = newSeq[float32](weights.len)
  var total = 0.0'f32
  var peak = 0.0'f32
  for i, w in weights:
    let t = pow(max(w, 0.0'f32), hp.weightPower)
    result[i] = t
    total += t
    peak = max(peak, t)
  if total <= 0.0'f32:
    for i in 0 ..< result.len: result[i] = 1.0'f32
    return
  let denom =
    case hp.weightNorm
    of wnMean: total / weights.len.float32
    of wnMax: peak
  if denom <= 0.0'f32: return
  for i in 0 ..< result.len: result[i] = result[i] / denom

proc scheduledLr*(hp: HyperParams, epoch: int): float32 =
  ## Learning rate for 1-based `epoch`. `lsConstant` returns `hp.lr` for every
  ## epoch, which keeps v0.1.0 runs bit-identical.
  let e = max(1, epoch)
  case hp.lrSchedule
  of lsConstant:
    result = hp.lr
  of lsExponential:
    result = hp.lr * pow(hp.lrDecay, (e - 1).float32)
    if result < hp.lrMin: result = hp.lrMin
  of lsLinear:
    if hp.epochs <= 1:
      result = hp.lr
    else:
      let frac = (e - 1).float32 / (hp.epochs - 1).float32
      result = hp.lr + (hp.lrMin - hp.lr) * frac

type
  TrainLogEntry* = object
    epoch*: int
    meanLoss*: float64
    validMRR*: float64 ## NaN when validation was not run this epoch

  TrainResult* = object
    log*: seq[TrainLogEntry]
    bestMRR*: float64
    bestEpoch*: int
    stoppedEarly*: bool
    epochsRun*: int

proc stepNegatives(m: var TransEModel, pos: Triple, sampler: Sampler,
                   rng: var Rand, hp: HyperParams, scratch: var StepScratch,
                   adversarial: bool): float32 =
  ## One positive's worth of negative updates. With `adversarial` the drawn
  ## negatives are re-weighted by `adversarialWeights` (and the *same* draws
  ## happen either way, so `advAlpha = 0` is bit-identical to the plain loop).
  if hp.negPerPos <= 1 or not adversarial:
    for _ in 0 ..< hp.negPerPos:
      let neg = sampler.sample(rng, pos)
      result += trainStep(m, pos, neg, hp, scratch)
    return
  var negs = newSeqOfCap[Triple](hp.negPerPos)
  var dists = newSeqOfCap[float32](hp.negPerPos)
  for _ in 0 ..< hp.negPerPos:
    let neg = sampler.sample(rng, pos)
    negs.add neg
    dists.add distance(m, neg, hp.norm)
  let weights = adversarialWeights(dists, hp.advAlpha)
  for j in 0 ..< negs.len:
    var whp = hp
    whp.lr = hp.lr * weights[j]
    result += trainStep(m, pos, negs[j], whp, scratch)

proc train*(m: var TransEModel, trainData: seq[Triple], sampler: Sampler,
            hp: HyperParams, valid: seq[Triple] = @[],
            truth: HashSet[Triple] = initHashSet[Triple](),
            ckptDir: string = "",
            weights: seq[float32] = @[]): TrainResult =
  ## SGD over shuffled positives; `hp.negPerPos` negatives each.
  ##
  ## Weighted graphs: with `hp.weightMode == wmSample` each epoch draws
  ## `trainData.len` positives with probability proportional to `w^power`
  ## (so an edge with twice the weight is seen twice as often, which is the
  ## maximum-likelihood reading of intensity/count weights). With `wmStep`
  ## every edge is still visited once per epoch and instead the SGD step is
  ## scaled by the normalised weight (the natural reading of confidence
  ## weights). `hp.useWeights = false`, or an empty `weights`, ignores them.
  ## When `hp.evalEvery > 0` and `valid` is non-empty, filtered MRR on the
  ## validation split drives early stopping (patience = hp.patience) and the
  ## best-scoring weights are restored at the end.
  var rng = initRand(hp.seed)
  var scratch = initScratch(hp.dim)
  var order = trainData
  var orderIdx: seq[int]
  let t0 = epochTime()
  # Weights are only *active* when they actually say something: a missing
  # column, an all-ones column, or `--weight-power 0` all leave the classic
  # shuffled loop in place, bit for bit as before weights existed. (Sample mode
  # with uniform weights would otherwise become a uniform draw *with
  # replacement*, which is statistically similar but not the same run.)
  var nonUnit = false
  if hp.weightPower != 0.0'f32:
    for w in weights:
      if w != 1.0'f32:
        nonUnit = true
        break
  let weighted = hp.useWeights and weights.len == trainData.len and
                 weights.len > 0 and nonUnit
  let sampling = weighted and hp.weightMode == wmSample
  let adversarial = hp.advAlpha > 0.0'f32 and hp.negPerPos > 1
  var cumulative: seq[float64]
  var totalWeight = 0.0
  var factors: seq[float32]
  if weighted:
    if sampling:
      cumulative = newSeq[float64](weights.len)
      for i, w in weights:
        totalWeight += pow(max(w, 0.0'f32), hp.weightPower)
        cumulative[i] = totalWeight
    else:
      factors = weightFactors(hp, weights)
  let doValid = hp.evalEvery > 0 and valid.len > 0
  var bestMRR = -1.0
  var bestEnt, bestRel: seq[float32]
  var sinceImprove = 0
  result.bestEpoch = -1
  let logEvery = max(1, hp.epochs div 100)

  for epoch in 1 .. hp.epochs:
    var epochLoss = 0.0
    # Effective learning rate for this epoch (constant unless a schedule is on).
    var ehp = hp
    ehp.lr = scheduledLr(hp, epoch)
    if sampling:
      # One epoch = one weighted draw per triple, so the work per epoch (and
      # therefore the wall clock) matches the unweighted loop.
      for _ in 0 ..< trainData.len:
        let pos = trainData[weightedDraw(rng, cumulative, totalWeight)]
        epochLoss += stepNegatives(m, pos, sampler, rng, ehp, scratch, adversarial)
    elif factors.len > 0:
      # Step mode: shuffle *positions* so each edge keeps its own weight, and
      # scale that edge's step. With every weight equal to 1 the factors are
      # exactly 1.0 and the shuffle draws the same RNG stream as the plain
      # triple shuffle, so an unweighted file is bit-identical to v0.3.0.
      if orderIdx.len == 0:
        orderIdx = newSeq[int](order.len)
        for i in 0 ..< orderIdx.len: orderIdx[i] = i
      rng.shuffle(orderIdx)
      for bStart in countup(0, orderIdx.len - 1, hp.batchSize):
        let bEnd = min(bStart + hp.batchSize, orderIdx.len)
        for i in bStart ..< bEnd:
          let oi = orderIdx[i]
          var whp = ehp
          whp.lr = ehp.lr * factors[oi]
          epochLoss += stepNegatives(m, order[oi], sampler, rng, whp, scratch,
                                     adversarial)
    else:
      rng.shuffle(order)
      for bStart in countup(0, order.len - 1, hp.batchSize):
        let bEnd = min(bStart + hp.batchSize, order.len)
        for i in bStart ..< bEnd:
          epochLoss += stepNegatives(m, order[i], sampler, rng, ehp, scratch,
                                     adversarial)
    let meanLoss = epochLoss / (order.len * hp.negPerPos).float64

    var mrr = NaN
    if doValid and (epoch mod hp.evalEvery == 0 or epoch == hp.epochs):
      mrr = evaluate(m, valid, truth, hp.norm, hp.threads).mrrMean
      if mrr > bestMRR:
        bestMRR = mrr
        result.bestEpoch = epoch
        sinceImprove = 0
        bestEnt = rawCopySeq(m.entityEmb)
        bestRel = rawCopySeq(m.relEmb)
      else:
        inc sinceImprove

    result.log.add TrainLogEntry(epoch: epoch, meanLoss: meanLoss, validMRR: mrr)
    result.epochsRun = epoch

    if not hp.quiet:
      let freshMrr = mrr == mrr # not NaN
      let boundary = epoch mod logEvery == 0 or epoch == 1 or epoch == hp.epochs
      let lrTag = if hp.lrSchedule == lsConstant: "" else: &"  lr={ehp.lr:.5f}"
      if freshMrr:
        stderr.writeLine &"  epoch {epoch:>5}/{hp.epochs}  loss={meanLoss:.5f}  validMRR={mrr:.4f}{lrTag}  [{epochTime() - t0:.1f}s]"
      elif boundary:
        stderr.writeLine &"  epoch {epoch:>5}/{hp.epochs}  loss={meanLoss:.5f}{lrTag}  [{epochTime() - t0:.1f}s]"

    if ckptDir.len > 0 and hp.ckptEvery > 0 and epoch mod hp.ckptEvery == 0:
      saveCheckpoint(m, hp, ckptDir / &"ckpt_e{epoch}.bin")

    if doValid and sinceImprove >= hp.patience:
      result.stoppedEarly = true
      break

  result.bestMRR = bestMRR
  # Restore best-validation weights when early stopping was active.
  if doValid and bestMRR >= 0 and result.bestEpoch != result.epochsRun:
    let dim = m.dim
    let ep = m.entityEmb.storage.raw_buffer
    let rp = m.relEmb.storage.raw_buffer
    copyMem(ep, addr bestEnt[0], m.nEntities * dim * sizeof(float32))
    copyMem(rp, addr bestRel[0], m.nRelations * dim * sizeof(float32))

## Training loop: shuffled epochs, mini-batches, checkpoints, early stopping.
import std/[random, sets, times, strformat, os]
import arraymancer
import ./types
import ./gradients
import ./sampling
import ./evaluate
import ./persist

proc rawCopySeq(t: Tensor[float32]): seq[float32] =
  ## Contiguous copy of a tensor's storage (the deprecated `toRawSeq`).
  let n = t.size
  result = newSeq[float32](n)
  if n > 0:
    copyMem(addr result[0], t.storage.raw_buffer, n * sizeof(float32))

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

proc train*(m: var TransEModel, trainData: seq[Triple], sampler: Sampler,
            hp: HyperParams, valid: seq[Triple] = @[],
            truth: HashSet[Triple] = initHashSet[Triple](),
            ckptDir: string = ""): TrainResult =
  ## SGD over shuffled positives; `hp.negPerPos` negatives each.
  ## When `hp.evalEvery > 0` and `valid` is non-empty, filtered MRR on the
  ## validation split drives early stopping (patience = hp.patience) and the
  ## best-scoring weights are restored at the end.
  var rng = initRand(hp.seed)
  var scratch = initScratch(hp.dim)
  var order = trainData
  let t0 = epochTime()
  let doValid = hp.evalEvery > 0 and valid.len > 0
  var bestMRR = -1.0
  var bestEnt, bestRel: seq[float32]
  var sinceImprove = 0
  result.bestEpoch = -1
  let logEvery = max(1, hp.epochs div 100)

  for epoch in 1 .. hp.epochs:
    rng.shuffle(order)
    var epochLoss = 0.0
    for bStart in countup(0, order.len - 1, hp.batchSize):
      let bEnd = min(bStart + hp.batchSize, order.len)
      for i in bStart ..< bEnd:
        let pos = order[i]
        for _ in 0 ..< hp.negPerPos:
          let neg = sampler.sample(rng, pos)
          epochLoss += trainStep(m, pos, neg, hp, scratch)
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
      if freshMrr:
        stderr.writeLine &"  epoch {epoch:>5}/{hp.epochs}  loss={meanLoss:.5f}  validMRR={mrr:.4f}  [{epochTime() - t0:.1f}s]"
      elif boundary:
        stderr.writeLine &"  epoch {epoch:>5}/{hp.epochs}  loss={meanLoss:.5f}  [{epochTime() - t0:.1f}s]"

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

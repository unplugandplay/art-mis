## Score calibration: turn a TransE ranking into a probability.
##
## A TransE score is a dissimilarity on an uncalibrated scale — the model can
## satisfy its margin by inflating relation norms, so raw distances are only
## comparable *within* one query (see `DevLog.md`, session 8). The standard fix
## is temperature scaling: read each candidate's score as
##
##     p(e) = exp(-d(e) / τ) / Σ_e' exp(-d(e') / τ)
##
## and fit the single parameter τ on held-out triples by minimising the negative
## log-likelihood of the true entity. Nothing about the model changes; τ is
## metadata stored in the checkpoint, and `predict --probabilities` uses it.
import std/math
import ./types
import ./scoring

type
  Calibration* = object
    temperature*: float32 ## 0 means "not calibrated"
    nll*: float64         ## negative log-likelihood at `temperature`
    nllUnit*: float64     ## the same at τ = 1, for comparison
    meanProb*: float64    ## mean probability assigned to the true entity
    n*: int               ## triples the fit used

proc softmaxProbability*(scores: openArray[float32], target: int,
                         temperature: float32): float64 =
  ## `p(target)` under the temperature-scaled softmax of `scores`.
  if scores.len == 0 or temperature <= 0.0'f32: return 0.0
  var lo = Inf.float64
  for s in scores:
    lo = min(lo, s.float64)
  var z = 0.0
  for s in scores:
    z += exp(-(s.float64 - lo) / temperature.float64)
  if z <= 0.0: return 0.0
  exp(-(scores[target].float64 - lo) / temperature.float64) / z

proc sampleTriples*(data: seq[Triple], limit: int): seq[Triple] =
  ## Deterministic, spread-out subsample: with `limit < data.len` this takes
  ## every `data.len div limit`-th triple, so the fit does not depend on which
  ## end of the file happens to be interesting.
  if limit <= 0 or data.len <= limit: return data
  let stride = data.len div limit
  for i in countup(0, data.len - 1, max(1, stride)):
    result.add data[i]
    if result.len >= limit: break

proc fitTemperature*(v: EmbeddingView, data: seq[Triple], limit = 500,
                     samples = 24, lo = 0.01'f32, hi = 200.0'f32): Calibration =
  ## Score `limit` triples once (entity side), then grid-search τ over a
  ## logarithmic range. Cheap: the scoring dominates, the search is a pass over
  ## the cached score matrix.
  let triples = sampleTriples(data, limit)
  result.n = triples.len
  if triples.len == 0 or v.nEnt == 0:
    result.temperature = 1.0'f32
    return

  # Cache the tail-side scores for every sampled triple: n x nEnt floats.
  var scores = newSeq[float32](triples.len * v.nEnt)
  var sb = initScoreBoard(v)
  for i, t in triples:
    scoreAllTails(v, sb, t.head, t.rel)
    let row = i * v.nEnt
    for e in 0 ..< v.nEnt:
      scores[row + e] = sb.scores[e]

  proc nllAt(tau: float32): tuple[nll, meanProb: float64] =
    var total = 0.0
    var totalP = 0.0
    for i, t in triples:
      let row = i * v.nEnt
      # numerically stable log-softmax of -score/tau
      var hi2 = NegInf.float64
      for e in 0 ..< v.nEnt:
        hi2 = max(hi2, -scores[row + e].float64 / tau.float64)
      var z = 0.0
      for e in 0 ..< v.nEnt:
        z += exp(-scores[row + e].float64 / tau.float64 - hi2)
      let logP = -scores[row + t.tail].float64 / tau.float64 - hi2 - ln(z)
      total += -logP
      totalP += exp(logP)
    (total / triples.len.float64, totalP / triples.len.float64)

  let unit = nllAt(1.0'f32)
  result.nllUnit = unit.nll
  # log-spaced grid, then a local refinement around the winner
  var bestTau = 1.0'f32
  var best = unit
  for k in 0 ..< samples:
    let frac = k.float64 / (samples - 1).float64
    let tau = (ln(lo.float64) + frac * (ln(hi.float64) - ln(lo.float64))).exp.float32
    let got = nllAt(tau)
    if got.nll < best.nll:
      best = got
      bestTau = tau
  for k in -8 .. 8:
    let tau = bestTau * pow(1.25'f32, k.float32)
    if tau <= 0.0'f32: continue
    let got = nllAt(tau)
    if got.nll < best.nll:
      best = got
      bestTau = tau
  result.temperature = bestTau
  result.nll = best.nll
  result.meanProb = best.meanProb

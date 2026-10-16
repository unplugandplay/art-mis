## Link-prediction queries: complete `(h, r, ?)` or `(?, r, t)` from a trained
## model, optionally filtering out triples that are already known.
##
## This is the query side of `evaluate`: same scoring, same "lower is better"
## dissimilarity, but it returns the ranking instead of aggregate metrics.
import std/[algorithm, sets]
import ./types
import ./scoring

type
  Prediction* = object
    entity*: int     ## predicted entity id (index into the vocab)
    score*: float32  ## dissimilarity d(h, r, ?) / d(?, r, t); lower is better

proc rankCandidates(scores: seq[float32], k: int, rel, fixed: int,
                    tailSide: bool, truth: HashSet[Triple],
                    skipSelf: bool): seq[Prediction] =
  ## Best-first candidates. When `skipSelf` is on, the entity the caller already
  ## named is not proposed as its own answer (`Alice manages Alice`); this is
  ## the query-side analogue of `evaluate`, which never counts the true entity
  ## as a competitor to itself. When `truth` is non-empty, candidates that
  ## already form a true triple are skipped (filtered ranking).
  var candidates: seq[int]
  for e in 0 ..< scores.len:
    if skipSelf and e == fixed: continue
    if truth.len > 0:
      let t =
        if tailSide: Triple(head: fixed, rel: rel, tail: e)
        else: Triple(head: e, rel: rel, tail: fixed)
      if t in truth: continue
    candidates.add e
  candidates.sort(proc(a, b: int): int =
    if scores[a] < scores[b]: -1
    elif scores[a] > scores[b]: 1
    else: a - b) # stable, deterministic tie-break by id
  let n = min(k, candidates.len)
  result = newSeqOfCap[Prediction](n)
  for i in 0 ..< n:
    result.add Prediction(entity: candidates[i], score: scores[candidates[i]])

proc predictTails*(v: EmbeddingView, head, rel, k: int,
                   truth: HashSet[Triple] = initHashSet[Triple](),
                   allowSelf = false): seq[Prediction] =
  ## Top-`k` tails for `(head, rel, ?)`, best first.
  var sb = initScoreBoard(v)
  scoreAllTails(v, sb, head, rel)
  rankCandidates(sb.scores, k, rel, head, tailSide = true, truth,
                 skipSelf = not allowSelf)

proc predictHeads*(v: EmbeddingView, rel, tail, k: int,
                   truth: HashSet[Triple] = initHashSet[Triple](),
                   allowSelf = false): seq[Prediction] =
  ## Top-`k` heads for `(?, rel, tail)`, best first.
  var sb = initScoreBoard(v)
  scoreAllHeads(v, sb, rel, tail)
  rankCandidates(sb.scores, k, rel, tail, tailSide = false, truth,
                 skipSelf = not allowSelf)

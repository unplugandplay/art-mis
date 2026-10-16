## Link-prediction evaluation: MeanRank, MRR, Hits@1/3/10, raw and filtered.
##
## For each held-out triple we rank the true head (all tails fixed) and the
## true tail (all heads fixed) against every entity. Rank = 1 + #better +
## 0.5*#ties ("average" tie handling; ties are rare with float32 scores).
##
## Filtered ranks exclude every candidate that forms a triple present in
## `truth`. Instead of a hash-set probe per candidate (14.5k per side on
## FB15k-237), the known-true tails of each `(h, r)` and the known-true heads
## of each `(r, t)` are pre-indexed once, and flattened into per-triple
## candidate lists: the rank loop stays pure arithmetic, and the filtered rank
## is `rawRank - #truth candidates that score better` — bit-identical to the
## per-candidate counting, since both are exact multiples of 0.5.
##
## Work is split into contiguous chunks, each running on its own thread with
## its own `ScoreBoard`; chunk partials are summed in chunk order, so for a
## fixed thread count results are reproducible run to run.
import std/[sets, strformat, tables, typedthreads]
import ./types
import ./scoring

type
  EvalResult* = object
    n*: int ## number of ranks pooled (2 per evaluated triple)
    meanRank*, mrr*: float64
    hits1*, hits3*, hits10*: float64
    fMeanRank*, fMrr*: float64
    fHits1*, fHits3*, fHits10*: float64

  FilterIndex = object
    ## known-true tails/heads keyed by pair id: `(h, r)` and `(r, t)`.
    tailsByHR: Table[int, seq[int32]]
    headsByRT: Table[int, seq[int32]]
    nEnt*, nRel*: int

  FilterCandidates = object
    ## Per-evaluation-triple CSR view of `FilterIndex`, indexed by triple
    ## position so workers never touch a hash table.
    tailIdx, tailEnt: seq[int32]
    headIdx, headEnt: seq[int32]

  EvalJob = object
    ## GC-free work unit: safe to copy into a worker thread.
    v: EmbeddingView
    triples: ptr UncheckedArray[Triple]
    tailIdx, tailEnt, headIdx, headEnt: ptr UncheckedArray[int32]
    lo, hi: int
    res: ptr EvalResult

proc `$`*(e: EvalResult): string =
  let n = if e.n > 0: e.n.float64 else: 1.0
  result = &"n={e.n}\n"
  result &= &"  raw:      MR={e.meanRank / n:9.2f}  MRR={e.mrr / n:7.4f}  " &
            &"Hits@1={e.hits1 / n:6.4f}  Hits@3={e.hits3 / n:6.4f}  Hits@10={e.hits10 / n:6.4f}\n"
  result &= &"  filtered: MR={e.fMeanRank / n:9.2f}  MRR={e.fMrr / n:7.4f}  " &
            &"Hits@1={e.fHits1 / n:6.4f}  Hits@3={e.fHits3 / n:6.4f}  Hits@10={e.fHits10 / n:6.4f}"

proc mrrMean*(e: EvalResult): float64 =
  ## Mean filtered MRR (used for early stopping).
  if e.n > 0: e.fMrr / e.n.float64 else: 0.0

proc `+=`*(a: var EvalResult, b: EvalResult) =
  a.n += b.n
  a.meanRank += b.meanRank
  a.mrr += b.mrr
  a.hits1 += b.hits1
  a.hits3 += b.hits3
  a.hits10 += b.hits10
  a.fMeanRank += b.fMeanRank
  a.fMrr += b.fMrr
  a.fHits1 += b.fHits1
  a.fHits3 += b.fHits3
  a.fHits10 += b.fHits10

proc hitsAt(rank: float64, k: int): float64 {.inline.} =
  if rank <= k.float64: 1.0 else: 0.0

proc accumulate(e: var EvalResult, rank, fRank: float64) =
  e.meanRank += rank
  e.fMeanRank += fRank
  e.mrr += 1.0 / rank
  e.fMrr += 1.0 / fRank
  e.hits1 += hitsAt(rank, 1)
  e.hits3 += hitsAt(rank, 3)
  e.hits10 += hitsAt(rank, 10)
  e.fHits1 += hitsAt(fRank, 1)
  e.fHits3 += hitsAt(fRank, 3)
  e.fHits10 += hitsAt(fRank, 10)
  inc e.n

proc buildFilterIndex(truth: HashSet[Triple], nEnt, nRel: int): FilterIndex =
  result.nEnt = nEnt
  result.nRel = nRel
  for t in truth:
    result.tailsByHR.mgetOrPut(t.head * nRel + t.rel, @[]).add t.tail.int32
    result.headsByRT.mgetOrPut(t.rel * nEnt + t.tail, @[]).add t.head.int32

proc flatten(data: seq[Triple], idx: FilterIndex): FilterCandidates =
  ## Turn the sparse index into per-triple candidate lists (two passes:
  ## measure, prefix-sum, fill).
  let n = data.len
  result.tailIdx = newSeq[int32](n + 1)
  result.headIdx = newSeq[int32](n + 1)
  var tailTotal, headTotal = 0
  for i, t in data:
    let nt = idx.tailsByHR.getOrDefault(t.head * idx.nRel + t.rel).len
    let nh = idx.headsByRT.getOrDefault(t.rel * idx.nEnt + t.tail).len
    result.tailIdx[i] = nt.int32
    result.headIdx[i] = nh.int32
    tailTotal += nt
    headTotal += nh
  result.tailEnt = newSeq[int32](tailTotal)
  result.headEnt = newSeq[int32](headTotal)
  var accT, accH = 0
  for i in 0 ..< n:
    let ct = result.tailIdx[i].int
    let ch = result.headIdx[i].int
    result.tailIdx[i] = accT.int32
    result.headIdx[i] = accH.int32
    accT += ct
    accH += ch
  result.tailIdx[n] = accT.int32
  result.headIdx[n] = accH.int32
  for i, t in data:
    var w = result.tailIdx[i].int
    for c in idx.tailsByHR.getOrDefault(t.head * idx.nRel + t.rel):
      result.tailEnt[w] = c
      inc w
    w = result.headIdx[i].int
    for c in idx.headsByRT.getOrDefault(t.rel * idx.nEnt + t.tail):
      result.headEnt[w] = c
      inc w

proc ptrOf[T](s: var seq[T]): ptr UncheckedArray[T] {.inline.} =
  if s.len == 0: nil
  else: cast[ptr UncheckedArray[T]](addr s[0])

proc rankOf(scores: openArray[float32], self: int, sTrue: float32): float64 =
  ## Raw rank: 1 + #better + 0.5*#ties, skipping the true entity itself.
  result = 1.0
  for e in 0 ..< scores.len:
    if e == self: continue
    let s = scores[e]
    if s < sTrue: result += 1.0
    elif s == sTrue: result += 0.5

proc truthAdjust(scores: openArray[float32], sTrue: float32, self: int,
                 cand: ptr UncheckedArray[int32], lo, hi: int): float64 =
  ## How much the raw rank must drop once known-true candidates are filtered.
  result = 0.0
  for j in lo ..< hi:
    let c = cand[j].int
    if c == self: continue
    let s = scores[c]
    if s < sTrue: result += 1.0
    elif s == sTrue: result += 0.5

proc evalRange(job: EvalJob) {.thread.} =
  ## Rank every triple in [lo, hi) and store the chunk's aggregate.
  var sb = initScoreBoard(job.v)
  var acc: EvalResult
  for i in job.lo ..< job.hi:
    let t = job.triples[i]
    # --- head side: rank d(e, r, t) for all e
    scoreAllHeads(job.v, sb, t.rel, t.tail)
    let sHead = sb.scores[t.head]
    let rankHead = rankOf(sb.scores, t.head, sHead)
    let fRankHead = rankHead - truthAdjust(sb.scores, sHead, t.head,
                                           job.headEnt, job.headIdx[i].int,
                                           job.headIdx[i + 1].int)
    acc.accumulate(rankHead, fRankHead)
    # --- tail side: rank d(h, r, e) for all e
    scoreAllTails(job.v, sb, t.head, t.rel)
    let sTail = sb.scores[t.tail]
    let rankTail = rankOf(sb.scores, t.tail, sTail)
    let fRankTail = rankTail - truthAdjust(sb.scores, sTail, t.tail,
                                           job.tailEnt, job.tailIdx[i].int,
                                           job.tailIdx[i + 1].int)
    acc.accumulate(rankTail, fRankTail)
  job.res[] = acc

proc jobFor(v: EmbeddingView, data: seq[Triple], cand: var FilterCandidates,
            lo, hi: int, res: ptr EvalResult): EvalJob =
  EvalJob(v: v,
          triples: cast[ptr UncheckedArray[Triple]](unsafeAddr data[0]),
          tailIdx: ptrOf(cand.tailIdx), tailEnt: ptrOf(cand.tailEnt),
          headIdx: ptrOf(cand.headIdx), headEnt: ptrOf(cand.headEnt),
          lo: lo, hi: hi, res: res)

proc evaluate*(m: TransEModel, data: seq[Triple], truth: HashSet[Triple],
               norm: NormKind, threads: int = 1): EvalResult =
  ## Pooled raw + filtered link-prediction metrics.
  ##
  ## `threads > 1` splits the triples into that many contiguous chunks and
  ## evaluates them concurrently (as many workers as there is work, capped at
  ## `threads`). Results are identical to the single-threaded run up to the
  ## last bits of the float64 aggregation; a fixed thread count is fully
  ## reproducible.
  if data.len == 0:
    return
  var v = view(m)
  v.norm = norm
  let idx = buildFilterIndex(truth, v.nEnt, v.nRel)
  var cand = flatten(data, idx)
  let nWorkers = max(1, min(threads, data.len))

  if nWorkers == 1:
    evalRange(jobFor(v, data, cand, 0, data.len, addr result))
    return

  var partials = newSeq[EvalResult](nWorkers)
  var workers = newSeq[Thread[EvalJob]](nWorkers)
  let chunk = (data.len + nWorkers - 1) div nWorkers
  var spawned = 0
  for k in 0 ..< nWorkers:
    let lo = k * chunk
    if lo >= data.len: break
    let hi = min(lo + chunk, data.len)
    createThread(workers[k], evalRange,
                 jobFor(v, data, cand, lo, hi, addr partials[k]))
    inc spawned
  for k in 0 ..< spawned:
    joinThread(workers[k])
  for k in 0 ..< spawned:
    result += partials[k]

## Command-line interface: train / eval / export / nearest.
import std/[parseopt, os, strutils, strformat, tables, sets, random, json,
            math, algorithm, cpuinfo]
import ./types
import ./io
import ./embeddings
import ./sampling
import ./train
import ./evaluate
import ./persist

const GeneralHelp = """
artemis — TransE knowledge-graph embeddings in Nim

Usage:
  artemis train   --data train.tsv [--valid valid.tsv --test test.tsv] [options]
  artemis eval    --model checkpoint.bin --data test.tsv [--truth train.tsv,valid.tsv]
  artemis export  --model checkpoint.bin [--out dir]
  artemis nearest --model checkpoint.bin --entity NAME [--k 10] [--metric cosine|l2]

Train options:
  --dim N          embedding dimension            (default 50)
  --margin F       margin ranking loss margin     (default 1.0)
  --lr F           SGD learning rate              (default 0.01)
  --lr-schedule S  constant | exp | linear        (default constant)
  --lr-decay F     per-epoch factor for exp       (default 1.0 = no decay)
  --lr-min F       floor (exp) / final lr (linear)(default 0.0)
  --epochs N       max epochs                     (default 1000)
  --batch N        batch size                     (default 100)
  --neg N          negatives per positive         (default 1)
  --norm l1|l2     dissimilarity norm             (default l1)
  --bern / --no-bern  Bernoulli negative sampling (default on)
  --seed N         RNG seed                       (default 42)
  --eval-every N   validate every N epochs        (default 0 = off)
  --ckpt-every N   checkpoint every N epochs      (default 0 = final only)
  --min-count N    drop rarer entities/relations  (default 1)
  --threads N      eval worker threads            (default 1; 0 = one per core)
  --delimiter CH   tab | comma | space | other    (default tab)
  --out DIR        output directory               (default .)
  --quiet          suppress progress

Eval options:
  --threads N      eval worker threads            (default 1; 0 = one per core)
  --truth F[,F...] extra known-true files for filtered ranking (e.g. train,valid)
  --delimiter CH   tab | comma | space | other    (default tab)
"""

proc die(msg: string) {.noreturn.} =
  stderr.writeLine "error: " & msg
  quit(1)

proc parseDelim(s: string): char =
  case s
  of "tab": '\t'
  of "comma": ','
  of "space": ' '
  else:
    if s.len == 1: s[0]
    else: die("delimiter must be one char or tab|comma|space")

const flagKeys = ["quiet", "bern", "no-bern", "help"]

proc collectOpts(args: seq[string]): tuple[opts: Table[string, string],
                                           flags: Table[string, bool]] =
  ## parseopt yields `--key value` values as separate cmdArgument tokens;
  ## known boolean flags are kept in `flags`, everything else expects a value.
  var p = initOptParser(args)
  var pending = ""
  while true:
    p.next()
    case p.kind
    of cmdEnd: break
    of cmdArgument:
      if pending.len > 0:
        result.opts[pending] = p.key
        pending = ""
      else:
        die("unexpected argument: " & p.key)
    of cmdShortOption, cmdLongOption:
      if pending.len > 0:
        die("option --" & pending & " expects a value")
      if p.val.len > 0:
        result.opts[p.key] = p.val
      elif p.key in flagKeys:
        result.flags[p.key] = true
      else:
        pending = p.key
  if pending.len > 0:
    die("option --" & pending & " expects a value")

proc optInt(o: Table[string, string], key: string, def: int): int =
  if key in o: parseInt(o[key]) else: def

proc optF32(o: Table[string, string], key: string, def: float32): float32 =
  if key in o: parseFloat(o[key]).float32 else: def

proc optThreads(o: Table[string, string]): int =
  ## 0 (or `--threads 0`) means "one worker per core"; 1 means sequential.
  let n = o.optInt("threads", 1)
  if n < 0: die("--threads must be >= 0")
  if n == 0: max(1, countProcessors()) else: n

proc cmdTrain(args: seq[string]) =
  let (opts, flags) = collectOpts(args)
  if "data" notin opts:
    die("train requires --data PATH\n" & GeneralHelp)
  var hp = HyperParams()
  hp.dim = opts.optInt("dim", hp.dim)
  hp.margin = opts.optF32("margin", hp.margin)
  hp.lr = opts.optF32("lr", hp.lr)
  hp.lrDecay = opts.optF32("lr-decay", hp.lrDecay)
  hp.lrMin = opts.optF32("lr-min", hp.lrMin)
  hp.epochs = opts.optInt("epochs", hp.epochs)
  hp.batchSize = opts.optInt("batch", hp.batchSize)
  hp.negPerPos = opts.optInt("neg", hp.negPerPos)
  hp.seed = opts.optInt("seed", 42).int64
  hp.evalEvery = opts.optInt("eval-every", 0)
  hp.ckptEvery = opts.optInt("ckpt-every", 0)
  hp.minCount = opts.optInt("min-count", 1)
  hp.threads = opts.optThreads()
  hp.quiet = "quiet" in flags
  hp.bernoulli = not ("no-bern" in flags)
  if "norm" in opts:
    case opts["norm"].toLowerAscii
    of "l1": hp.norm = nkL1
    of "l2": hp.norm = nkL2
    else: die("unknown norm: " & opts["norm"])
  if "delimiter" in opts:
    hp.delimiter = parseDelim(opts["delimiter"])
  if "lr-schedule" in opts:
    case opts["lr-schedule"].toLowerAscii
    of "constant", "none", "off": hp.lrSchedule = lsConstant
    of "exp", "exponential": hp.lrSchedule = lsExponential
    of "linear", "lin": hp.lrSchedule = lsLinear
    else: die("unknown lr schedule: " & opts["lr-schedule"] &
              " (expected constant|exp|linear)")
  if hp.dim <= 0 or hp.margin <= 0 or hp.lr <= 0 or hp.epochs < 1 or
     hp.batchSize < 1 or hp.negPerPos < 1:
    die("invalid hyperparameters (dim/margin/lr/epochs/batch/neg)")
  if hp.lrDecay <= 0 or hp.lrDecay > 1:
    die("--lr-decay must be in (0, 1]")
  if hp.lrMin < 0 or hp.lrMin > hp.lr:
    die("--lr-min must be in [0, --lr]")
  if hp.lrSchedule == lsExponential and hp.lrDecay >= 1.0'f32:
    die("--lr-schedule exp needs --lr-decay < 1 (e.g. 0.99)")

  let outDir = opts.getOrDefault("out", ".")
  createDir(outDir)

  let (data, entities, relations, report) = loadDataset(
    opts["data"], hp, opts.getOrDefault("valid", ""), opts.getOrDefault("test", ""))
  stderr.writeLine &"loaded: {data.train.len} train / {data.valid.len} valid / {data.test.len} test triples"
  stderr.writeLine &"vocab: {entities.len} entities, {relations.len} relations"
  if hp.lrSchedule != lsConstant:
    var desc = &"lr schedule: {$hp.lrSchedule}  lr={hp.lr}  min={hp.lrMin}"
    if hp.lrSchedule == lsExponential:
      desc &= &"  decay={hp.lrDecay}"
    stderr.writeLine desc
  if report.malformedTrain + report.malformedValid + report.malformedTest > 0:
    stderr.writeLine &"warning: skipped {report.malformedTrain + report.malformedValid + report.malformedTest} malformed lines"
  if report.skippedValid + report.skippedTest > 0:
    stderr.writeLine &"warning: skipped {report.skippedValid + report.skippedTest} valid/test triples with unknown names"
  if data.train.len == 0:
    die("no training triples")

  var rng = initRand(hp.seed)
  var model = initModel(entities, relations, hp, rng)
  let truth = buildTruthSet(data.train, data.valid, data.test)
  let sampler = initSampler(data.train, relations.len, entities.len,
                            hp.bernoulli, truth)

  let res = train(model, data.train, sampler, hp, data.valid, truth, outDir)
  if res.stoppedEarly:
    stderr.writeLine &"early stop after {res.epochsRun} epochs (best epoch {res.bestEpoch}, valid MRR {res.bestMRR:.4f})"

  let ckptPath = outDir / "artemis_checkpoint.bin"
  saveCheckpoint(model, hp, ckptPath)
  stderr.writeLine "checkpoint: " & ckptPath

  var metrics = newJObject()
  if data.test.len > 0:
    let r = evaluate(model, data.test, truth, hp.norm, hp.threads)
    echo "test metrics:"
    echo $r
    metrics["test"] = %*{"mrr": r.mrr / r.n.float64,
                         "hits1": r.hits1 / r.n.float64,
                         "hits10": r.hits10 / r.n.float64}
  exportEmbeddingsTsv(model, outDir / "embeddings.tsv")
  exportMetaJson(model, hp, metrics, outDir / "vocab.json")
  stderr.writeLine "exports: " & (outDir / "embeddings.tsv") & ", " & (outDir / "vocab.json")

proc cmdEval(args: seq[string]) =
  let (opts, _) = collectOpts(args)
  if "model" notin opts or "data" notin opts:
    die("eval requires --model PATH --data PATH")
  let (model, hp) = loadCheckpoint(opts["model"])
  var hp2 = hp
  if "delimiter" in opts:
    hp2.delimiter = parseDelim(opts["delimiter"])
  var malformed, skipped = 0
  let raws = loadRawTriples(opts["data"], hp2.delimiter, malformed)
  let data = toTriples(raws, model.entities, model.relations, skipped)
  if skipped > 0:
    stderr.writeLine &"warning: skipped {skipped} triples with unknown names"
  # Filtered ranking needs every known-true triple, not just the evaluated
  # split: `--truth` adds the other files (standard protocol: train+valid+test).
  var known = data
  for spec in opts.getOrDefault("truth", "").split(','):
    let path = spec.strip
    if path.len == 0: continue
    var m2, s2 = 0
    let extra = toTriples(loadRawTriples(path, hp2.delimiter, m2),
                          model.entities, model.relations, s2)
    if m2 + s2 > 0:
      stderr.writeLine &"warning: {path}: ignored {m2 + s2} lines (malformed or unknown names)"
    known.add extra
  let truth = buildTruthSet(known)
  stderr.writeLine &"truth set: {truth.len} triples"
  let r = evaluate(model, data, truth, hp2.norm, opts.optThreads())
  echo $r

proc cmdExport(args: seq[string]) =
  let (opts, _) = collectOpts(args)
  if "model" notin opts:
    die("export requires --model PATH")
  let outDir = opts.getOrDefault("out", ".")
  createDir(outDir)
  let (model, hp) = loadCheckpoint(opts["model"])
  exportEmbeddingsTsv(model, outDir / "embeddings.tsv")
  exportMetaJson(model, hp, nil, outDir / "vocab.json")
  stderr.writeLine "exports: " & (outDir / "embeddings.tsv") & ", " & (outDir / "vocab.json")

proc cmdNearest(args: seq[string]) =
  let (opts, _) = collectOpts(args)
  if "model" notin opts or "entity" notin opts:
    die("nearest requires --model PATH --entity NAME")
  let k = opts.optInt("k", 10)
  let metric = opts.getOrDefault("metric", "cosine")
  let (model, _) = loadCheckpoint(opts["model"])
  let name = opts["entity"]
  if name notin model.entities.toId:
    die("unknown entity: " & name)
  let dim = model.dim
  let ep = model.entityEmb.storage.raw_buffer
  let q = model.entities.toId[name]
  let qb = q * dim
  var sims = newSeq[(float32, int)](model.nEntities)
  for e in 0 ..< model.nEntities:
    let eb = e * dim
    var dot, na, nb = 0.0'f32
    for i in 0 ..< dim:
      dot += ep[qb + i] * ep[eb + i]
      na += ep[qb + i] * ep[qb + i]
      nb += ep[eb + i] * ep[eb + i]
    let s =
      if metric == "l2": -sqrt(na + nb - 2 * dot)
      else: dot / max(sqrt(na) * sqrt(nb), 1e-12'f32)
    sims[e] = (s, e)
  sims.sort(SortOrder.Descending)
  var shown = 0
  for (s, e) in sims:
    if e == q: continue
    echo &"{model.entities.names[e]:<30} {s:.5f}"
    inc shown
    if shown >= k: break

proc main*() =
  let args = commandLineParams()
  if args.len == 0 or args[0] in ["-h", "--help", "help"]:
    echo GeneralHelp
    quit(0)
  case args[0]
  of "train": cmdTrain(args[1 ..^ 1])
  of "eval": cmdEval(args[1 ..^ 1])
  of "export": cmdExport(args[1 ..^ 1])
  of "nearest": cmdNearest(args[1 ..^ 1])
  else:
    echo GeneralHelp
    die("unknown command: " & args[0])

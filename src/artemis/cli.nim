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
import ./scoring
import ./mermaid
import ./predict
import ./project
import ./calibrate

const GeneralHelp = """
artemis — TransE knowledge-graph embeddings in Nim

Usage:
  artemis train   --data train.tsv [--valid valid.tsv --test test.tsv] [options]
  artemis eval    --model checkpoint.bin --data test.tsv [--truth train.tsv,valid.tsv]
  artemis export  --model checkpoint.bin [--out dir]
  artemis nearest --model checkpoint.bin --entity NAME [--k 10] [--metric cosine|l2]
  artemis mermaid FILE.mmd [--out triples.tsv] [--default-rel R] [--group-rel R]
  artemis predict --model checkpoint.bin --rel NAME (--head NAME | --tail NAME) [--k 10]
  artemis project --model checkpoint.bin [--out DIR]
  artemis calibrate --model checkpoint.bin --data valid.tsv [--out checkpoint.bin]
  artemis --version

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
  --weight-mode M  sample | step  (weighted graphs; default sample)
  --weight-power F raise weights to this power    (default 1.0; 0 ignores)
  --weight-norm N  mean | max  (step mode scaling)(default mean)
  --no-weights     ignore a weight column entirely
  --adv-alpha F    self-adversarial negative weighting (RotatE-style, 0 = off)
                   needs --neg > 1; scale-free, larger = harder negatives
  --threads N      eval worker threads            (default 1; 0 = one per core)
  --delimiter CH   tab | comma | space | other    (default tab)
  --out DIR        output directory               (default .)
  --quiet          suppress progress

Eval options:
  --threads N      eval worker threads            (default 1; 0 = one per core)
  --truth F[,F...] extra known-true files for filtered ranking (e.g. train,valid)
  --delimiter CH   tab | comma | space | other    (default tab)

Mermaid options:
  --out FILE       write triples here             (default: stdout)
  --no-weight-labels  treat numeric/`rel:w` labels as plain relations
  --no-payload     keep node payload as text instead of triples
  --default-rel R  relation for unlabelled edges  (default relatedTo)
  --group-rel R    relation for subgraph members  (default inGroup)

Calibrate options:
  --data FILE      held-out triples to fit the temperature on
  --sample N       triples to use                    (default 500)
  --out FILE       where to save the calibrated model (default: --model)
  --quiet          no summary

Query options (predict):
  --head NAME      rank tails for (head, rel, ?)
  --tail NAME      rank heads for (?, rel, tail)
  --rel NAME       relation (required)
  --k N            how many candidates to print   (default 10)
  --filter F[,F...] drop candidates already true in these files
  --candidates F[,F...] only propose values this relation is seen with
  --probabilities  print calibrated probabilities (needs `artemis calibrate`)
  --temperature F  override the model's calibration temperature
  --allow-self     also propose the entity you named (off by default)

Project options:
  --out DIR        destination for map.svg + projection.tsv (default .)
  --no-relations   project entities only
  --no-labels      draw points without text labels
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

const flagKeys = ["quiet", "bern", "no-bern", "help", "allow-self",
                  "no-relations", "no-labels", "no-weights",
                  "no-weight-labels", "no-payload", "probabilities"]

proc collectOpts(args: seq[string]): tuple[opts: Table[string, string],
                                           flags: Table[string, bool],
                                           positional: seq[string]] =
  ## parseopt yields `--key value` values as separate cmdArgument tokens;
  ## known boolean flags are kept in `flags`, everything else expects a value.
  ## Arguments that are not attached to an option land in `positional`.
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
        result.positional.add p.key
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
  let (opts, flags, _) = collectOpts(args)
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
  hp.weightPower = opts.optF32("weight-power", hp.weightPower)
  hp.advAlpha = opts.optF32("adv-alpha", hp.advAlpha)
  hp.useWeights = "no-weights" notin flags
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
  if "weight-mode" in opts:
    case opts["weight-mode"].toLowerAscii
    of "sample", "sampling": hp.weightMode = wmSample
    of "step", "lr": hp.weightMode = wmStep
    else: die("unknown weight mode: " & opts["weight-mode"] &
              " (expected sample|step)")
  if "weight-norm" in opts:
    case opts["weight-norm"].toLowerAscii
    of "mean", "avg": hp.weightNorm = wnMean
    of "max": hp.weightNorm = wnMax
    else: die("unknown weight norm: " & opts["weight-norm"] &
              " (expected mean|max)")
  if hp.weightPower < 0:
    die("--weight-power must be >= 0")
  if hp.advAlpha < 0:
    die("--adv-alpha must be >= 0")
  if hp.advAlpha > 0 and hp.negPerPos < 2:
    die("--adv-alpha needs --neg 2 or more (it re-weights the drawn negatives)")
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
  if data.weights.len > 0 and hp.useWeights and report.weightedTrain > 0:
    var total = 0.0'f32
    for w in data.weights: total += w
    stderr.writeLine &"weights: {report.weightedTrain}/{data.weights.len} train triples weighted " &
                     &"(mean {total / data.weights.len.float32:.3f}, mode {$hp.weightMode}, " &
                     &"power {hp.weightPower}, {$hp.weightNorm}-normalised)"
  elif data.weights.len > 0 and not hp.useWeights:
    stderr.writeLine "weights: present but ignored (--no-weights)"
  elif data.weights.len > 0 and hp.weightPower == 0.0'f32:
    stderr.writeLine "weights: ignored (--weight-power 0)"
  elif data.weights.len > 0:
    stderr.writeLine "weights: none in effect (no column, or every value 1.0) - unweighted run"
  if hp.advAlpha > 0:
    stderr.writeLine &"negatives: {hp.negPerPos} per positive, self-adversarial alpha={hp.advAlpha}"
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

  let res = train(model, data.train, sampler, hp, data.valid, truth, outDir,
                  data.weights)
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
  let (opts, _, _) = collectOpts(args)
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
  let (opts, _, _) = collectOpts(args)
  if "model" notin opts:
    die("export requires --model PATH")
  let outDir = opts.getOrDefault("out", ".")
  createDir(outDir)
  let (model, hp) = loadCheckpoint(opts["model"])
  exportEmbeddingsTsv(model, outDir / "embeddings.tsv")
  exportMetaJson(model, hp, nil, outDir / "vocab.json")
  stderr.writeLine "exports: " & (outDir / "embeddings.tsv") & ", " & (outDir / "vocab.json")

proc cmdNearest(args: seq[string]) =
  let (opts, _, _) = collectOpts(args)
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

proc tripleLine(t: RawTriple): string =
  ## `head<TAB>relation<TAB>tail`, plus the weight column when it is not 1.0
  ## (so unweighted output stays exactly as it was before weights existed).
  result = t.head & "\t" & t.relation & "\t" & t.tail
  if t.weight != 1.0'f32:
    result &= "\t" & $t.weight

proc cmdMermaid(args: seq[string]) =
  let (opts, flags, positional) = collectOpts(args)
  var path = opts.getOrDefault("in", "")
  if path.len == 0 and positional.len > 0: path = positional[0]
  if path.len == 0:
    die("mermaid requires a diagram file: artemis mermaid diagram.mmd [--out triples.tsv]")
  if not fileExists(path): die("no such file: " & path)
  let g = parseMermaid(readFile(path),
                       opts.getOrDefault("default-rel", "relatedTo"),
                       opts.getOrDefault("group-rel", "inGroup"),
                       labelWeights = "no-weight-labels" notin flags,
                       payload = "no-payload" notin flags)
  for diagram in g.report.unsupported:
    stderr.writeLine "warning: unsupported diagram, skipped: " & diagram
  for line in g.report.skipped:
    stderr.writeLine "warning: could not parse: " & line
  var summary = &"mermaid: {g.report.nodes} entities, {g.report.edges} edges, " &
                &"{g.report.groups} group links, {g.report.duplicates} duplicates dropped"
  if g.report.weighted > 0: summary &= &", {g.report.weighted} weighted"
  if g.report.attributes > 0: summary &= &", {g.report.attributes} payload triples"
  stderr.writeLine summary
  if g.triples.len == 0:
    stderr.writeLine "warning: no triples produced"
  let outPath = opts.getOrDefault("out", "")
  if outPath.len > 0:
    var f = open(outPath, fmWrite)
    defer: f.close()
    for t in g.triples:
      f.writeLine tripleLine(t)
    stderr.writeLine "triples: " & outPath
  else:
    for t in g.triples:
      echo tripleLine(t)

proc loadTruthFiles(spec: string, delimiter: char, model: TransEModel): HashSet[Triple] =
  ## `--filter a.tsv,b.tsv`: known-true triples to exclude from predictions.
  for path in spec.split(','):
    let p = path.strip
    if p.len == 0: continue
    var malformed, skipped = 0
    let triples = toTriples(loadRawTriples(p, delimiter, malformed),
                            model.entities, model.relations, skipped)
    for t in triples: result.incl t

proc cmdPredict(args: seq[string]) =
  let (opts, flags, _) = collectOpts(args)
  if "model" notin opts or "rel" notin opts:
    die("predict requires --model PATH --rel NAME and --head or --tail")
  if ("head" in opts) == ("tail" in opts):
    die("predict needs exactly one of --head NAME (rank tails) or --tail NAME (rank heads)")
  let k = opts.optInt("k", 10)
  if k < 1: die("--k must be >= 1")
  let (model, hp) = loadCheckpoint(opts["model"])
  let relName = opts["rel"]
  if relName notin model.relations.toId: die("unknown relation: " & relName)
  let rel = model.relations.toId[relName]
  var truth = initHashSet[Triple]()
  if "filter" in opts:
    truth = loadTruthFiles(opts["filter"], hp.delimiter, model)
  let allowSelf = "allow-self" in flags
  var temperature = 0.0'f32
  if "probabilities" in flags:
    temperature = if "temperature" in opts: opts.optF32("temperature", 0.0'f32)
                  else: model.temperature
    if temperature <= 0.0'f32:
      die("this model has no calibration temperature; run `artemis calibrate` " &
          "or pass --temperature F")
  elif "temperature" in opts:
    temperature = opts.optF32("temperature", 0.0'f32)
  elif model.temperature > 0.0'f32:
    stderr.writeLine &"hint: this model is calibrated (tau={model.temperature:.4g}); " &
                     "add --probabilities to print p-values"
  # `--candidates FILE`: keep a relation's answers to the values it is actually
  # seen with, so a `role` question cannot answer with an age literal.
  var restrict = initHashSet[int]()
  if "candidates" in opts:
    let tailSide = "tail" notin opts
    for path in opts["candidates"].split(','):
      let p = path.strip
      if p.len == 0: continue
      var malformed, skipped = 0
      for t in toTriples(loadRawTriples(p, hp.delimiter, malformed),
                         model.entities, model.relations, skipped):
        if t.rel == rel: restrict.incl (if tailSide: t.tail else: t.head)
  var v = view(model)
  v.norm = hp.norm
  if "head" in opts:
    let headName = opts["head"]
    if headName notin model.entities.toId: die("unknown entity: " & headName)
    let preds = predictTails(v, model.entities.toId[headName], rel, k, truth,
                             allowSelf, restrict, temperature)
    echo &"top {preds.len} tails for ({headName}, {relName}, ?)" &
         (if truth.len > 0: &"  [{truth.len} known triples filtered]" else: "") &
         (if restrict.len > 0: &"  [candidates: {restrict.len} known {relName} values]" else: "") &
         (if temperature > 0.0'f32: &"  [calibrated, tau={temperature:.4g}]" else: "")
    for p in preds:
      if temperature > 0.0'f32:
        echo &"{model.entities.names[p.entity]:<34} {p.score:10.5f}   p={p.prob:.4f}"
      else:
        echo &"{model.entities.names[p.entity]:<34} {p.score:10.5f}"
  else:
    let tailName = opts["tail"]
    if tailName notin model.entities.toId: die("unknown entity: " & tailName)
    let preds = predictHeads(v, rel, model.entities.toId[tailName], k, truth,
                             allowSelf, restrict, temperature)
    echo &"top {preds.len} heads for (?, {relName}, {tailName})" &
         (if truth.len > 0: &"  [{truth.len} known triples filtered]" else: "") &
         (if restrict.len > 0: &"  [candidates: {restrict.len} known {relName} values]" else: "") &
         (if temperature > 0.0'f32: &"  [calibrated, tau={temperature:.4g}]" else: "")
    for p in preds:
      if temperature > 0.0'f32:
        echo &"{model.entities.names[p.entity]:<34} {p.score:10.5f}   p={p.prob:.4f}"
      else:
        echo &"{model.entities.names[p.entity]:<34} {p.score:10.5f}"

proc cmdProject(args: seq[string]) =
  let (opts, flags, _) = collectOpts(args)
  if "model" notin opts:
    die("project requires --model PATH")
  let outDir = opts.getOrDefault("out", ".")
  createDir(outDir)
  let (model, _) = loadCheckpoint(opts["model"])
  let p = project(model, includeRelations = not ("no-relations" in flags))
  if p.points.len == 0: die("nothing to project (empty model)")
  writeProjectionTsv(p, outDir / "projection.tsv")
  writeProjectionSvg(p, outDir / "map.svg", labels = not ("no-labels" in flags))
  stderr.writeLine &"projected {p.points.len} vectors to 2D " &
                   &"(variance captured: {p.var1:.4g} / {p.var2:.4g})"
  stderr.writeLine "wrote " & (outDir / "map.svg") & " and " & (outDir / "projection.tsv")

proc cmdCalibrate(args: seq[string]) =
  let (opts, flags, _) = collectOpts(args)
  if "model" notin opts or "data" notin opts:
    die("calibrate requires --model PATH --data valid.tsv")
  let (model, hp) = loadCheckpoint(opts["model"])
  let dataPath = opts["data"]
  var malformed, skipped = 0
  let data = toTriples(loadRawTriples(dataPath, hp.delimiter, malformed),
                       model.entities, model.relations, skipped)
  if data.len == 0: die("no usable triples in " & dataPath)
  if skipped > 0:
    stderr.writeLine &"warning: skipped {skipped} triples with unknown names"
  var v = view(model)
  v.norm = hp.norm
  let sample = opts.optInt("sample", 500)
  let cal = fitTemperature(v, data, limit = sample)
  var m = model
  m.temperature = cal.temperature
  let outPath = opts.getOrDefault("out", opts["model"])
  saveCheckpoint(m, hp, outPath)
  if "quiet" notin flags:
    stderr.writeLine &"calibrated on {cal.n} triples from {dataPath}"
    stderr.writeLine &"temperature: tau={cal.temperature:.4g}  " &
                     &"NLL {cal.nllUnit:.4f} -> {cal.nll:.4f}  " &
                     &"(mean probability of the true tail: {cal.meanProb:.3f})"
    stderr.writeLine "model: " & outPath

proc main*() =
  let args = commandLineParams()
  if args.len == 0 or args[0] in ["-h", "--help", "help"]:
    echo GeneralHelp
    quit(0)
  if args[0] in ["-v", "--version", "version"]:
    echo "artemis " & ArtemisVersion
    quit(0)
  case args[0]
  of "train": cmdTrain(args[1 ..^ 1])
  of "eval": cmdEval(args[1 ..^ 1])
  of "export": cmdExport(args[1 ..^ 1])
  of "nearest": cmdNearest(args[1 ..^ 1])
  of "mermaid": cmdMermaid(args[1 ..^ 1])
  of "predict": cmdPredict(args[1 ..^ 1])
  of "project": cmdProject(args[1 ..^ 1])
  of "calibrate": cmdCalibrate(args[1 ..^ 1])
  else:
    echo GeneralHelp
    die("unknown command: " & args[0])

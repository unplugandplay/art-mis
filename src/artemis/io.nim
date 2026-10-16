## Data ingestion: TSV parsing, vocabulary building, truth-set index.
import std/[options, tables, sets, strutils, syncio]
import ./types

type
  LoadReport* = object
    malformedTrain*: int
    malformedValid*: int
    malformedTest*: int
    skippedValid*: int ## unknown names (not in train vocab)
    skippedTest*: int
    filteredRare*: int ## raw triples dropped by min-count

proc parseLine*(line: string, delim: char = '\t'): Option[RawTriple] =
  ## Parse one `head<delim>relation<delim>tail` line.
  ## Returns none for malformed lines (wrong field count, empty field,
  ## blank line, comment).
  if line.len == 0 or line[0] == '#':
    return none(RawTriple)
  let parts = line.split(delim)
  if parts.len != 3:
    return none(RawTriple)
  let h = parts[0].strip
  let r = parts[1].strip
  let t = parts[2].strip
  if h.len == 0 or r.len == 0 or t.len == 0:
    return none(RawTriple)
  result = some((head: h, relation: r, tail: t))

proc loadRawTriples*(path: string, delim: char = '\t',
                     malformed: var int): seq[RawTriple] =
  ## Read a triple file, skipping (and counting) malformed lines.
  malformed = 0
  for line in lines(path):
    let parsed = parseLine(line, delim)
    if parsed.isSome:
      result.add parsed.get
    else:
      inc malformed

proc internId(v: var Vocab, name: string): int =
  if name in v.toId:
    result = v.toId[name]
  else:
    result = v.names.len
    v.toId[name] = result
    v.names.add name

proc buildVocab*(raws: seq[RawTriple], minCount: int = 1,
                 filtered: var int): tuple[entities, relations: Vocab] =
  ## First-appearance-order vocab over triples surviving the min-count filter.
  var entCount = initCountTable[string]()
  var relCount = initCountTable[string]()
  for t in raws:
    entCount.inc(t.head)
    entCount.inc(t.tail)
    relCount.inc(t.relation)
  filtered = 0
  for t in raws:
    if entCount[t.head] < minCount or entCount[t.tail] < minCount or
       relCount[t.relation] < minCount:
      inc filtered
      continue
    discard result.entities.internId(t.head)
    discard result.relations.internId(t.relation)
    discard result.entities.internId(t.tail)

proc toTriples*(raws: seq[RawTriple], entities, relations: Vocab,
                skipped: var int): seq[Triple] =
  ## Convert raw triples to id triples using an existing vocab.
  ## Triples mentioning unknown names are skipped and counted.
  skipped = 0
  for t in raws:
    let h = entities.toId.getOrDefault(t.head, -1)
    let r = relations.toId.getOrDefault(t.relation, -1)
    let e = entities.toId.getOrDefault(t.tail, -1)
    if h < 0 or r < 0 or e < 0:
      inc skipped
      continue
    result.add Triple(head: h, rel: r, tail: e)

proc loadDataset*(trainPath: string, hp: HyperParams,
                  validPath: string = "", testPath: string = ""
                 ): tuple[data: Dataset, entities, relations: Vocab,
                          report: LoadReport] =
  ## Load train (+ optional valid/test) files. Vocabulary is built from the
  ## training split only, in first-appearance order (deterministic).
  var report: LoadReport
  let rawsTrain = loadRawTriples(trainPath, hp.delimiter, report.malformedTrain)
  let (entities, relations) = buildVocab(rawsTrain, hp.minCount, report.filteredRare)
  var skippedTrain = 0
  result.data.train = toTriples(rawsTrain, entities, relations, skippedTrain)
  result.entities = entities
  result.relations = relations
  if validPath.len > 0:
    let raws = loadRawTriples(validPath, hp.delimiter, report.malformedValid)
    result.data.valid = toTriples(raws, entities, relations, report.skippedValid)
  if testPath.len > 0:
    let raws = loadRawTriples(testPath, hp.delimiter, report.malformedTest)
    result.data.test = toTriples(raws, entities, relations, report.skippedTest)
  result.report = report

proc buildTruthSet*(splits: varargs[seq[Triple]]): HashSet[Triple] =
  ## All known-true triples, for filtered negative sampling and filtered metrics.
  for s in splits:
    for t in s:
      result.incl(t)

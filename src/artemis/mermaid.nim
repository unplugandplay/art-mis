## Mermaid ingestion — turn a Mermaid diagram into `head<TAB>relation<TAB>tail`.
##
## Supported (line-based subsets of the grammar):
##
##   flowchart / graph    `A --> B`, `A -->|rel| B`, `A -- rel --> B`,
##                        chains (`A --> B --> C`), `&` multi-targets,
##                        `subgraph title ... end` (membership triples)
##   stateDiagram(-v2)    `A --> B : rel`
##   classDiagram         `A <|-- B : rel` (member blocks are skipped)
##   erDiagram            `A ||--o{ B : rel` (attribute blocks are skipped)
##
## Node shapes (`id[Label]`, `id(Label)`, `id{Label}`, `id((Label))`, …) become
## entity *names*: the id is what Mermaid connects, the label is what humans
## read, and the output keeps the label (falling back to the id).
##
## Anything else (sequenceDiagram, gantt, pie, …) is reported as unsupported
## rather than guessed at. Individual lines that do not parse are counted and
## skipped, so one odd edge never sinks a diagram.
import std/[strutils, tables, sets]
import ./types
import ./io

type
  MermaidReport* = object
    lines*: int                ## lines considered (after comments/headers)
    nodes*: int                ## distinct entity names emitted
    edges*: int                ## distinct relation triples from edges
    groups*: int               ## distinct membership triples from subgraphs
    duplicates*: int           ## edges dropped because they were already seen
    weighted*: int             ## edges that carry an explicit weight != 1
    unsupported*: seq[string]  ## diagram kinds we do not attempt
    skipped*: seq[string]      ## lines we could not parse

  MermaidGraph* = object
    triples*: seq[RawTriple]
    report*: MermaidReport

  DiagramKind = enum
    dkNone, dkFlow, dkState, dkClass, dkEr, dkUnsupported

const
  ArrowChars = {'-', '.', '=', '<', '>', 'o', 'x', '*', '{', '}', '~', '|'}
  UnsupportedKinds = ["sequencediagram", "gantt", "pie", "journey", "mindmap",
                      "gitgraph", "timeline", "quadrantchart", "xychart-beta",
                      "block-beta", "sankey-beta", "requirementdiagram",
                      "c4context", "c4container", "zenuml", "packet-beta"]
  IgnoredDirectives = ["style", "classdef", "class", "linkstyle", "click",
                       "direction", "acctitle", "accdescr", "note", "rect",
                       "cssclass", "callback", "href", "title"]

proc diagramKind(line: string): DiagramKind =
  if line.len == 0: return dkNone
  let head = line.splitWhitespace()[0].toLowerAscii
  case head
  of "graph", "flowchart": dkFlow
  of "statediagram", "statediagram-v2": dkState
  of "classdiagram": dkClass
  of "erdiagram": dkEr
  else:
    if head in UnsupportedKinds: dkUnsupported else: dkNone

proc isDirective(line: string): bool =
  let head = line.splitWhitespace()[0].toLowerAscii
  result = head in IgnoredDirectives

proc stripDecoration(s: string): string =
  ## `"Alice Smith"`, `Alice<br/>Smith`, `[[Alice]]` -> `Alice Smith`.
  result = s.strip()
  if result.len >= 2 and result[0] == '"' and result[^1] == '"':
    result = result[1 .. ^2]
  for tag in ["<br/>", "<br />", "<br>", "<BR/>"]:
    result = result.replace(tag, " ")
  result = result.splitWhitespace().join(" ")
  const fluff = {'[', ']', '(', ')', '{', '}', '<', '>', '/', '\\', '|', '"'}
  while result.len > 0 and result[0] in fluff:
    result = result[1 .. ^1].strip()
  while result.len > 0 and result[^1] in fluff:
    result = result[0 .. ^2].strip()

proc readId(line: string, i: var int): string =
  ## Mermaid node id. A `-` only continues the id when an id character follows,
  ## so `A-->B` splits into id/arrow/id while `node-1` stays one id.
  let start = i
  while i < line.len:
    let c = line[i]
    if c.isAlphaNumeric or c in {'_', '.'} or ord(c) > 127:
      inc i
    elif c == '-' and i + 1 < line.len and
         (line[i + 1].isAlphaNumeric or line[i + 1] in {'_', '.'}):
      inc i
    else:
      break
  if i > start: line[start ..< i] else: ""

proc readShape(line: string, i: var int): string =
  ## `i` sits on `[`, `(`, `{` or `>`; returns the inner text and advances past
  ## the closer. Returns "" (leaving `i` advanced) when unterminated.
  let open = line[i]
  let closec =
    case open
    of '[': ']'
    of '(': ')'
    of '{': '}'
    of '>': ']'
    else: ']'
  var depth = 0
  inc i
  let start = i
  while i < line.len:
    if line[i] == open and open != '>':
      inc depth
    elif line[i] == closec:
      if depth == 0:
        result = line[start ..< i]
        inc i
        return
      dec depth
    inc i

proc parseNode(line: string, i: var int,
               labels: var Table[string, string]): string =
  ## Parse `id` or `id<shape>`; register the shape text as the node's label.
  while i < line.len and line[i] in {' ', '\t'}: inc i
  let id = readId(line, i)
  if id.len == 0: return ""
  result = id
  if i < line.len and line[i] in {'[', '(', '{', '>'}:
    let save = i
    let raw = readShape(line, i)
    let label = stripDecoration(raw)
    if label.len > 0: labels[id] = label
    else: i = save

proc readArrow(line: string, i: var int, er: bool): string =
  ## Consume a run of arrow/cardinality characters. In ER mode `|` is part of
  ## the operator (`||--o{`); elsewhere a trailing `|` opens an edge label
  ## (`-->|rel|`) and is split off by the caller.
  while i < line.len and line[i] in {' ', '\t'}: inc i
  let start = i
  while i < line.len:
    let c = line[i]
    if c notin ArrowChars: break
    # `-->ox` must leave `ox` to the next node id, not eat it as cardinality
    if c in {'o', 'x'} and i + 1 < line.len and
       (line[i + 1].isAlphaNumeric or line[i + 1] in {'_', '.'}):
      break
    inc i
  if i > start: line[start ..< i] else: ""

proc readWord(line: string, i: var int): string =
  while i < line.len and line[i] in {' ', '\t'}: inc i
  let start = i
  while i < line.len and line[i] notin {' ', '\t', '|'}: inc i
  if i > start: line[start ..< i] else: ""

proc splitLabel(label, defaultRel: string,
                enabled: bool): tuple[rel: string, weight: float32] =
  ## Edge labels may carry a weight: `trusts:0.75` is relation+weight, a bare
  ## `0.75` is a weight for the default relation, anything else is just a
  ## relation name. `enabled` is the `--no-weight-labels` escape hatch.
  result = (label, 1.0'f32)
  if not enabled or label.len == 0: return
  let colon = label.rfind(':')
  if colon > 0:
    var w: float32
    if parseWeight(label[colon + 1 .. ^1], w):
      result = (label[0 ..< colon].strip(), w)
      return
  var w2: float32
  if parseWeight(label, w2):
    result = (defaultRel, w2)

proc scanLine(line: string, kind: DiagramKind,
              labels: var Table[string, string],
              edges: var seq[(string, string, string, float32)],
              groups: var seq[(string, string)],
              groupStack: seq[string],
              defaultRel: string, labelWeights: bool): bool =
  ## Walk one diagram line; true when it made sense (even if it only declared a
  ## node). Partially understood lines may still emit edges before returning.
  let er = kind == dkEr
  var i = 0
  var emitted = 0
  var prev = parseNode(line, i, labels)
  if prev.len == 0: return false
  if groupStack.len > 0: groups.add (prev, groupStack[^1])
  while true:
    let beforeArrow = i
    var op = readArrow(line, i, er)
    if op.len == 0:
      i = beforeArrow
      break
    let source = prev
    var rel = ""
    var opClean = op
    if op[^1] == '|' and not er:
      # `-->|rel|` (readArrow already consumed the opening `|`)
      opClean = op[0 ..< op.len - 1]
      let lstart = i
      while i < line.len and line[i] != '|': inc i
      if i >= line.len: return false
      rel = line[lstart ..< i].strip()
      inc i
    elif op.startsWith("-") and op[^1] in {'-', '.'}:
      # `-- rel -->`: the word between the dashes is the label. Only an *open*
      # operator (ending in `-` or `.`) can start this form — `-->` is already
      # a complete arrow, so `A --> B --> C` must not read `B` as a label.
      let mark = i
      let word = readWord(line, i)
      let op2 = readArrow(line, i, er)
      if word.len > 0 and op2.len > 0 and op2.contains("-"):
        rel = word
        opClean = op2
      else:
        i = mark
    while i < line.len and line[i] in {' ', '\t'}: inc i
    let tail = parseNode(line, i, labels)
    if tail.len == 0:
      if opClean.len > 0: return false
      break
    if groupStack.len > 0: groups.add (tail, groupStack[^1])
    let afterTail = i
    while i < line.len and line[i] in {' ', '\t'}: inc i
    if i < line.len and line[i] == ':':
      # state / class / ER label: `A --> B : rel`
      inc i
      rel = line[i .. ^1].strip()
      i = line.len
    else:
      i = afterTail
    let (useRel, useWeight) =
      if rel.len > 0: splitLabel(rel, defaultRel, labelWeights)
      else: (defaultRel, 1.0'f32)
    # `Animal <|-- Dog : isA` reads Dog -> Animal: a leading `<` (without a
    # matching `>`) puts the arrowhead on the left node, so flip the edge.
    let flipped = opClean.len > 0 and opClean[0] == '<' and opClean[^1] != '>'
    if flipped: edges.add (tail, useRel, source, useWeight)
    else: edges.add (source, useRel, tail, useWeight)
    inc emitted
    prev = tail
    # `A --> B & C` — one source, several targets, same relation
    while true:
      let mark = i
      while i < line.len and line[i] in {' ', '\t'}: inc i
      if i < line.len and line[i] == '&':
        inc i
        let other = parseNode(line, i, labels)
        if other.len == 0:
          i = mark
          break
        if groupStack.len > 0: groups.add (other, groupStack[^1])
        if flipped: edges.add (other, useRel, source, useWeight)
        else: edges.add (source, useRel, other, useWeight)
        inc emitted
        prev = other
      else:
        i = mark
        break
  # Understood when it produced an edge, or when it was a bare node
  # declaration (`A[Alice]`); anything else is a line we did not parse.
  while i < line.len and line[i] in {' ', '\t'}: inc i
  result = emitted > 0 or i >= line.len

proc parseMermaid*(text: string, defaultRel = "relatedTo",
                   groupRel = "inGroup",
                   labelWeights = true): MermaidGraph =
  ## Convert a Mermaid document into triples. Multiple diagrams may be present;
  ## each header resets the diagram kind.
  var labels = initTable[string, string]()
  var edges: seq[(string, string, string, float32)]
  var groups: seq[(string, string)]
  var groupStack: seq[string]
  var kind = dkNone
  var inBlock = false

  for rawLine in text.splitLines:
    var line = rawLine
    let comment = line.find("%%")
    if comment >= 0: line = line[0 ..< comment]
    line = line.strip()
    if line.len == 0: continue
    if inBlock:
      if line.startsWith("}"): inBlock = false
      continue
    let hk = diagramKind(line)
    if hk != dkNone:
      kind = hk
      if hk == dkUnsupported and line notin result.report.unsupported:
        result.report.unsupported.add line
      continue
    if kind == dkUnsupported: continue
    inc result.report.lines
    if line == "end":
      if groupStack.len > 0: discard groupStack.pop()
      continue
    if line.startsWith("subgraph"):
      var j = "subgraph".len
      while j < line.len and line[j] in {' ', '\t'}: inc j
      var title = ""
      var id = ""
      if j < line.len and line[j] notin {'[', '(', '{'}:
        id = parseNode(line, j, labels)
        title = labels.getOrDefault(id, "")
        if title.len == 0 and id.len > 0:
          # `subgraph eng [Engineering]` — shape after the id
          var k = j
          while k < line.len and line[k] in {' ', '\t'}: inc k
          if k < line.len and line[k] in {'[', '(', '{'}:
            title = stripDecoration(readShape(line, k))
          else:
            title = id
      if title.len == 0:
        title = stripDecoration(line[j .. ^1])
      if title.len == 0: title = "group"
      if id.len > 0: labels[id] = title
      groupStack.add title
      continue
    if isDirective(line) and not ((kind == dkEr or kind == dkClass) and
                                  line.endsWith("{")): continue
    if (kind == dkEr or kind == dkClass) and line.endsWith("{"):
      # entity / class body: register the name, skip the members
      var j = 0
      if line.startsWith("class "): j = "class ".len
      discard parseNode(line, j, labels)
      inBlock = true
      continue
    if kind == dkClass and line.startsWith("class ") and not line.endsWith("{"):
      var j = "class ".len
      discard parseNode(line, j, labels)
      continue
    if not scanLine(line, kind, labels, edges, groups, groupStack, defaultRel,
                    labelWeights):
      result.report.skipped.add line

  proc nameOf(id: string): string =
    labels.getOrDefault(id, id)

  var unique = initTable[string, int]() # key -> index into result.triples
  var entities = initHashSet[string]()
  for (h, r, t, w) in edges:
    let head = nameOf(h)
    let tail = nameOf(t)
    let key = head & "\t" & r & "\t" & tail
    if unique.hasKey(key):
      # repeated edge: keep the strongest claim rather than the first
      inc result.report.duplicates
      let i = unique[key]
      if w > result.triples[i].weight: result.triples[i].weight = w
      continue
    unique[key] = result.triples.len
    entities.incl head
    entities.incl tail
    if w != 1.0'f32: inc result.report.weighted
    result.triples.add (head: head, relation: r, tail: tail, weight: w)
  for (node, group) in groups:
    let n = nameOf(node)
    if n == group: continue
    let key = n & "\t" & groupRel & "\t" & group
    if unique.hasKey(key):
      inc result.report.duplicates
      continue
    unique[key] = result.triples.len
    entities.incl n
    entities.incl group
    inc result.report.groups
    result.triples.add (head: n, relation: groupRel, tail: group, weight: 1.0'f32)
  for id in labels.keys:
    entities.incl nameOf(id)
  result.report.nodes = entities.len
  result.report.edges = result.triples.len - result.report.groups

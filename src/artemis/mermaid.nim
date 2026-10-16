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
    attributes*: int           ## triples derived from node payload
    unsupported*: seq[string]  ## diagram kinds we do not attempt
    skipped*: seq[string]      ## lines we could not parse

  MermaidGraph* = object
    triples*: seq[RawTriple]
    report*: MermaidReport

  NodeInfo* = object
    ## What a diagram says about one node: its display name and any payload
    ## lines (`age: 34`, class/ER members, state descriptions).
    label*: string
    payload*: seq[(string, string)]

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


proc splitBr(raw: string): seq[string] =
  ## Split shape text on Mermaid's line breaks (`<br/>` and friends).
  var s = raw
  for tag in ["<br/>", "<br />", "<br>", "<BR/>", "<BR />", "<BR>"]:
    s = s.replace(tag, "\n")
  s.split('\n')

proc splitPayload(raw: string, extract = true
                 ): tuple[label: string, payload: seq[(string, string)]] =
  ## Node text -> (display name, payload).
  ##
  ## `Alice Smith<br/>age: 34<br/>role: engineer` names the node `Alice Smith`
  ## and yields `(age, 34)` / `(role, engineer)`. Every line of the form
  ## `key: value` becomes payload; the remaining lines are joined into the
  ## display name (a payload-only node falls back to its id, so it still has
  ## an entity). Labels that are supposed to *contain* a colon need
  ## `--no-payload`.
  var parts: seq[string]
  for segment in splitBr(raw):
    let s = stripDecoration(segment)
    if s.len == 0: continue
    let colon = if extract: s.find(':') else: -1
    if colon > 0:
      let k = s[0 ..< colon].strip()
      let v = s[colon + 1 .. ^1].strip()
      if k.len > 0 and v.len > 0:
        result.payload.add (k, v)
        continue
    parts.add s
  result.label = parts.join(" ")

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

proc isIdLike(s: string): bool =
  ## True when `s` is exactly one Mermaid node id (used to tell `s1 : desc`
  ## apart from a line that merely contains a colon).
  if s.len == 0: return false
  var i = 0
  readId(s, i) == s

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
               nodes: var OrderedTable[string, NodeInfo],
               payload = true): string =
  ## Parse `id` or `id<shape>`; register the node's display name and payload.
  while i < line.len and line[i] in {' ', '\t'}: inc i
  let id = readId(line, i)
  if id.len == 0: return ""
  result = id
  if i < line.len and line[i] in {'[', '(', '{', '>'}:
    let save = i
    let raw = readShape(line, i)
    let (label, items) = splitPayload(raw, extract = payload)
    if label.len > 0 or items.len > 0:
      if label.len > 0: nodes.mgetOrPut(id, NodeInfo()).label = label
      if items.len > 0: nodes.mgetOrPut(id, NodeInfo()).payload.add items
    else:
      i = save

proc parsePseudo(line: string, i: var int, isFirst: bool,
                 prefix: string): string =
  ## Mermaid's `[*]` entry/exit marker: `Start` when it opens the transition,
  ## `End` when it closes it, qualified by the composite state it sits in.
  while i < line.len and line[i] in {' ', '\t'}: inc i
  if i + 2 < line.len and line[i] == '[' and line[i + 1] == '*' and
     line[i + 2] == ']':
    i += 3
    result = prefix & (if isFirst: "Start" else: "End")

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
              nodes: var OrderedTable[string, NodeInfo],
              edges: var seq[(string, string, string, float32)],
              groups: var seq[(string, string)],
              groupStack: seq[string],
              defaultRel: string, labelWeights: bool, payload = true,
              statePrefix = ""): bool =
  ## Walk one diagram line; true when it made sense (even if it only declared a
  ## node). Partially understood lines may still emit edges before returning.
  let er = kind == dkEr
  var i = 0
  var emitted = 0
  var prev = parseNode(line, i, nodes, payload)
  if prev.len == 0: prev = parsePseudo(line, i, isFirst = true, statePrefix)
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
    var tail = parseNode(line, i, nodes, payload)
    if tail.len == 0: tail = parsePseudo(line, i, isFirst = false, statePrefix)
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
        let other = parseNode(line, i, nodes, payload)
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

type
  Member = object
    ## One line of an ER attribute block or a class body.
    owner*: string
    name*: string
    typ*: string
    visibility*: string
    key*: string   ## ER `PK` / `FK`
    isMethod*: bool

proc parseMember(line: string, er: bool): Member =
  ## `string name PK` (ER) or `+int age` / `+getAge() int` (class).
  var s = line.strip()
  if s.len == 0 or s.startsWith("%%"): return
  if not er and s[0] in {'+', '-', '#', '~'}:
    result.visibility =
      case s[0]
      of '+': "public"
      of '-': "private"
      of '#': "protected"
      else: "package"
    s = s[1 .. ^1].strip()
  let paren = s.find('(')
  if paren >= 0:
    result.isMethod = true
    result.name = s[0 ..< paren].strip()
    let close = s.find(')', paren)
    if close >= 0:
      result.typ = s[close + 1 .. ^1].strip().strip(chars = {':', ' '})
  else:
    let parts = s.splitWhitespace()
    if parts.len >= 2:
      result.typ = parts[0]
      result.name = parts[1]
      if er and parts.len >= 3 and parts[^1].toUpperAscii() in ["PK", "FK", "UK"]:
        result.key = parts[^1].toUpperAscii()
      elif er and parts.len == 2 and parts[0].toUpperAscii() in ["PK", "FK"]:
        # `id PK` (no type)
        result.name = parts[0]
        result.typ = ""
        result.key = parts[1].toUpperAscii()
    elif parts.len == 1:
      result.name = parts[0]
  result.name = result.name.strip(chars = {':', ';', ' '})

proc parseMermaid*(text: string, defaultRel = "relatedTo",
                   groupRel = "inGroup",
                   labelWeights = true,
                   payload = true): MermaidGraph =
  ## Convert a Mermaid document into triples. Multiple diagrams may be present;
  ## each header resets the diagram kind.
  var nodes = initOrderedTable[string, NodeInfo]()
  var edges: seq[(string, string, string, float32)]
  var groups: seq[(string, string)]
  var groupStack: seq[string]
  var kind = dkNone
  var inBlock = false
  var blockOwner = ""
  var members: seq[Member]
  var stateStack: seq[string]

  for rawLine in text.splitLines:
    var line = rawLine
    let comment = line.find("%%")
    if comment >= 0: line = line[0 ..< comment]
    line = line.strip()
    if line.len == 0: continue
    if inBlock:
      if line.startsWith("}"):
        inBlock = false
      elif payload:
        let m = parseMember(line, kind == dkEr)
        if m.name.len > 0:
          var owned = m
          owned.owner = blockOwner
          members.add owned
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
        id = parseNode(line, j, nodes, payload)
        title = nodes.getOrDefault(id, NodeInfo()).label
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
      if id.len > 0: nodes.mgetOrPut(id, NodeInfo()).label = title
      groupStack.add title
      continue
    if isDirective(line) and not ((kind == dkEr or kind == dkClass) and
                                  line.endsWith("{")): continue
    if (kind == dkEr or kind == dkClass) and line.endsWith("{"):
      # entity / class body: register the name, skip the members
      var j = 0
      if line.startsWith("class "): j = "class ".len
      blockOwner = parseNode(line, j, nodes, payload)
      inBlock = true
      continue
    if kind == dkClass and line.startsWith("class ") and not line.endsWith("{"):
      var j = "class ".len
      discard parseNode(line, j, nodes, payload)
      continue
    if kind == dkState:
      if line.startsWith("state "):
        var rest = line["state ".len .. ^1].strip()
        var composite = false
        if rest.endsWith("{"):
          composite = true # composite state: its body is parsed normally
          rest = rest[0 ..< ^1].strip()
        var label = ""
        var id = ""
        let asPos = rest.find(" as ")
        if asPos > 0:
          label = stripDecoration(rest[0 ..< asPos])
          id = rest[asPos + 4 .. ^1].strip()
        else:
          id = rest.strip()
        if id.len > 0:
          if label.len > 0: nodes.mgetOrPut(id, NodeInfo()).label = label
          elif not nodes.hasKey(id): nodes[id] = NodeInfo()
          if composite: stateStack.add id
        continue
      if line == "}":
        if stateStack.len > 0: discard stateStack.pop()
        continue
      if not line.contains(">") and line.find(':') > 0:
        # `waiting : waiting for input` attaches a description to a state
        let colon = line.find(':')
        let id = line[0 ..< colon].strip()
        let desc = line[colon + 1 .. ^1].strip()
        if id.len > 0 and desc.len > 0 and isIdLike(id):
          nodes.mgetOrPut(id, NodeInfo()).payload.add ("description", desc)
          continue
    let statePrefix =
      if kind == dkState and stateStack.len > 0: stateStack[^1] & "."
      else: ""
    if not scanLine(line, kind, nodes, edges, groups, groupStack, defaultRel,
                    labelWeights, payload, statePrefix):
      result.report.skipped.add line

  proc nameOf(id: string): string =
    let info = nodes.getOrDefault(id, NodeInfo())
    if info.label.len > 0: info.label else: id

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
  # ER / class members: `(Owner, hasAttribute, Owner.field)` plus type, key and
  # visibility. The attribute is qualified by its owner so that two entities
  # can both have a `name` without collapsing into one node.
  var payloadTriples: seq[(string, string, string)] # head, relation, tail
  for m in members:
    let ownerName = nameOf(m.owner)
    let attrNode = ownerName & "." & m.name
    payloadTriples.add (ownerName,
                        (if m.isMethod: "hasMethod" else: "hasAttribute"),
                        attrNode)
    if m.typ.len > 0:
      payloadTriples.add (attrNode, (if m.isMethod: "returns" else: "ofType"),
                          m.typ)
    if m.visibility.len > 0:
      payloadTriples.add (attrNode, "visibility", m.visibility)
    if m.key.len > 0:
      payloadTriples.add (attrNode, "key", m.key)
  # Inline `key: value` payload from node shapes and state descriptions.
  if payload:
    for id in nodes.keys:
      for (k, v) in nodes[id].payload:
        payloadTriples.add (nameOf(id), k, v)
  for (h, r, t) in payloadTriples:
    let key = h & "\t" & r & "\t" & t
    if unique.hasKey(key):
      inc result.report.duplicates
      continue
    unique[key] = result.triples.len
    entities.incl h
    entities.incl t
    inc result.report.attributes
    result.triples.add (head: h, relation: r, tail: t, weight: 1.0'f32)
  for id in nodes.keys:
    entities.incl nameOf(id)
  result.report.nodes = entities.len
  result.report.edges = result.triples.len - result.report.groups -
                        result.report.attributes

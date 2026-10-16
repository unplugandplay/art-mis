import std/[unittest, strutils]
import artemis/types
import artemis/mermaid

proc has(g: MermaidGraph, h, r, t: string): bool =
  for tr in g.triples:
    if tr.head == h and tr.relation == r and tr.tail == t: return true

proc count(g: MermaidGraph, h, r, t: string): int =
  for tr in g.triples:
    if tr.head == h and tr.relation == r and tr.tail == t: inc result

suite "mermaid: flowchart":
  let g = parseMermaid("""
flowchart LR
  %% a comment, ignored
  subgraph eng [Engineering]
    alice[Alice] -->|manages| bob[Bob]
    bob -->|mentors| carol[Carol]
  end
  dave["Dave <br/>Jones"] -- reportsTo --> alice
  alice --> bob & carol
  style alice fill:#f9f
  this line is not mermaid at all
""")

  test "labelled, dashed and chained edges":
    check g.has("Alice", "manages", "Bob")
    check g.has("Bob", "mentors", "Carol")
    check g.has("Dave Jones", "reportsTo", "Alice")

  test "shapes become entity names, ids are resolved":
    check g.has("Alice", "relatedTo", "Bob")
    check g.has("Alice", "relatedTo", "Carol")

  test "subgraphs become membership triples":
    check g.has("Alice", "inGroup", "Engineering")
    check g.has("Bob", "inGroup", "Engineering")
    check g.has("Carol", "inGroup", "Engineering")
    check not g.has("Dave Jones", "inGroup", "Engineering")
    check g.report.groups == 3

  test "unparsable lines are reported, not guessed":
    check g.report.skipped.len == 1
    check g.report.skipped[0].startsWith("this line")

  test "node/edge accounting":
    # 5 edge triples + 3 membership triples; entities are Alice, Bob, Carol,
    # Dave Jones and the Engineering group itself
    check g.report.edges == 5
    check g.report.nodes == 5

  test "repeated edges are deduplicated":
    let d = parseMermaid("graph TD\n  a --> b\n  a --> b\n  a -->|r| b\n")
    check d.triples.len == 2 # relatedTo + r
    check d.report.duplicates == 1

suite "mermaid: other diagram kinds":
  test "stateDiagram colon labels":
    let g = parseMermaid("stateDiagram-v2\n  idle --> running : start\n  running --> idle : stop\n")
    check g.has("idle", "start", "running")
    check g.has("running", "stop", "idle")

  test "classDiagram inheritance and members":
    let g = parseMermaid("""
classDiagram
  class Animal {
    +int age
  }
  Animal <|-- Dog : isA
  Dog *-- Collar : has
""")
    check g.has("Dog", "isA", "Animal")
    check g.has("Dog", "has", "Collar")
    check g.report.skipped.len == 0

  test "erDiagram cardinality and attribute blocks":
    let g = parseMermaid("""
erDiagram
  CUSTOMER ||--o{ ORDER : places
  ORDER }o--|| CUSTOMER : belongsTo
  CUSTOMER {
    string name
    string id
  }
""")
    check g.has("CUSTOMER", "places", "ORDER")
    check g.has("ORDER", "belongsTo", "CUSTOMER")
    check g.report.edges == 2
    check g.report.skipped.len == 0

  test "unsupported diagrams are reported, not half-parsed":
    let g = parseMermaid("sequenceDiagram\n  Alice->>Bob: Hi\n  Bob-->>Alice: Hello\n")
    check g.report.unsupported.len == 1
    check g.triples.len == 0
    check g.report.skipped.len == 0

  test "default relation applies to bare arrows":
    let g = parseMermaid("graph TD\n  a --> b\n", defaultRel = "linkedTo")
    check g.has("a", "linkedTo", "b")

  test "custom group relation":
    let g = parseMermaid("graph TD\n  subgraph team\n    a --> b\n  end\n",
                         groupRel = "memberOf")
    check g.has("a", "memberOf", "team")
    check g.has("a", "relatedTo", "b")

  test "bare node declarations register entities without triples":
    let g = parseMermaid("graph TD\n  solo[Only Node]\n")
    check g.triples.len == 0
    check g.report.nodes == 1
    check g.report.skipped.len == 0

  test "numeric and rel:weight labels become edge weights":
    let g = parseMermaid("""
graph LR
  a -->|0.75| b
  a -->|trusts:0.5| c
  b -->|plain relation| c
""")
    check g.triples.len == 3
    check g.report.weighted == 2
    for t in g.triples:
      case t.tail
      of "b": check t.weight == 0.75'f32 and t.relation == "relatedTo"
      of "c":
        if t.head == "a": check t.weight == 0.5'f32 and t.relation == "trusts"
        else: check t.weight == 1.0'f32 and t.relation == "plain relation"
      else: discard

  test "weights can be turned off for label-shaped diagrams":
    let g = parseMermaid("graph LR\n  a -->|0.75| b\n", labelWeights = false)
    check g.triples[0].relation == "0.75"
    check g.triples[0].weight == 1.0'f32
    check g.report.weighted == 0

  test "repeated edges keep the strongest weight":
    let g = parseMermaid("graph LR\n  a -->|0.2| b\n  a -->|0.9| b\n")
    check g.triples.len == 1
    check g.triples[0].weight == 0.9'f32
    check g.report.duplicates == 1

  test "subgraph membership carries no weight":
    let g = parseMermaid("graph LR\n  subgraph g1\n    a -->|3| b\n  end\n")
    for t in g.triples:
      if t.relation == "inGroup": check t.weight == 1.0'f32
      else: check t.weight == 3.0'f32

  test "empty input is not an error":
    let g = parseMermaid("")
    check g.triples.len == 0
    check g.report.nodes == 0

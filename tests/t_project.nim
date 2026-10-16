import std/[unittest, math, os, strutils]
import artemis/types
import artemis/project
import helpers

suite "project: PCA of row vectors":
  test "a one-dimensional cloud projects onto its own direction":
    # 4 points along x in 2D: the first component must be the x axis.
    let rows = @[1.0'f32, 0, 2, 0, 3, 0, 4, 0]
    let (v1, v2, l1, l2) = principalAxes(rows, n = 4, dim = 2)
    check abs(v1[0]) > 0.999
    check abs(v1[1]) < 1e-3
    check l1 > 1.0'f32          # ~1.67
    check l2 < 1e-4'f32         # nothing left on the second axis
    discard v2

  test "two well-separated clusters separate on the first component":
    # clusters at the origin and at (5,5)
    var rows: seq[float32]
    for (x, y) in [(0.0'f32, 0.0'f32), (0.1'f32, 0.0'f32),
                   (5.0'f32, 5.0'f32), (5.1'f32, 5.0'f32)]:
      rows.add x
      rows.add y
    let (v1, _, l1, _) = principalAxes(rows, n = 4, dim = 2)
    let s = sqrt(v1[0] * v1[0] + v1[1] * v1[1])
    check abs(s - 1.0'f32) < 1e-4'f32            # unit axis
    check abs(v1[0] - v1[1]) < 1e-3'f32          # diagonal: (1,1)/sqrt2
    check l1 > 5.0'f32                           # large spread on that axis

  test "degenerate input does not divide by zero":
    let (v1, _, l1, l2) = principalAxes(@[1.0'f32, 1, 1, 1], n = 2, dim = 2)
    check l1 < 1e-6'f32
    check l2 < 1e-6'f32
    discard v1
    let empty = principalAxes(@[], n = 0, dim = 0)
    check empty.l1 == 0.0'f32

suite "project: model projection and outputs":
  let m = modelFromArrays(
    @["a", "b", "c"], @["r"], 3,
    @[1.0'f32, 0, 0,   0, 1, 0,   -1, 0, 0],
    @[0.5'f32, 0.5, 0])

  test "every entity and relation gets a point, deterministically":
    let p = project(m)
    check p.points.len == m.nEntities + m.nRelations
    check p.points[0].kind == "entity"
    check p.points[^1].kind == "relation"
    check p.points[^1].name == "REL/r"
    let q = project(m)
    for i in 0 ..< p.points.len:
      check p.points[i].x == q.points[i].x
      check p.points[i].y == q.points[i].y

  test "relations can be left out":
    check project(m, includeRelations = false).points.len == m.nEntities

  test "an identical-vector model projects every point to the same place":
    let flat = modelFromArrays(@["a", "b"], @["r"], 2,
                               @[1.0'f32, 1, 1, 1], @[1.0'f32, 1])
    let p = project(flat)
    check abs(p.points[0].x - p.points[1].x) < 1e-5'f32
    check abs(p.points[0].y - p.points[1].y) < 1e-5'f32

  test "TSV and SVG writers produce the expected files":
    let dir = getTempDir() / "artemis_project_test"
    createDir(dir)
    let p = project(m)
    let tsv = dir / "projection.tsv"
    let svg = dir / "map.svg"
    writeProjectionTsv(p, tsv)
    writeProjectionSvg(p, svg)
    let lines = readFile(tsv).strip.splitLines
    check lines.len == p.points.len + 1 # header + one row per vector
    check lines[0] == "name\tkind\tx\ty"
    let body = readFile(svg)
    check body.startsWith("<?xml")
    check body.contains("<svg")
    check body.contains("</svg>")
    check body.contains(">a<")          # entity label present
    check body.contains("REL/r")

  test "SVG labels are XML-escaped":
    let odd = modelFromArrays(@["a&b", "<c>"], @["r"], 2,
                              @[1.0'f32, 0, 0, 1], @[1.0'f32, 0])
    let dir = getTempDir() / "artemis_project_test"
    createDir(dir)
    let svg = dir / "escaped.svg"
    writeProjectionSvg(project(odd), svg)
    let body = readFile(svg)
    check body.contains("a&amp;b")
    check body.contains("&lt;c&gt;")

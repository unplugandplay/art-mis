## Visualise trained embeddings: project every vector to 2D with PCA (power
## iteration + deflation, no external maths library) and write a self-contained
## SVG scatter plus the raw coordinates.
##
## The projection is deterministic: the power iteration always starts from the
## same vector, so the same checkpoint always yields the same picture.
import std/[math, strformat, strutils]
import ./types

type
  ProjectionPoint* = object
    name*: string
    kind*: string ## "entity" or "relation"
    x*, y*: float32

  Projection* = object
    points*: seq[ProjectionPoint]
    var1*, var2*: float32 ## variance captured by each component (for the report)

proc matVec(cov: openArray[float32], dim: int, v: openArray[float32],
            outv: var seq[float32]) =
  for i in 0 ..< dim:
    var s = 0.0'f32
    let row = i * dim
    for j in 0 ..< dim:
      s += cov[row + j] * v[j]
    outv[i] = s

proc dot(a, b: openArray[float32]): float32 =
  for i in 0 ..< a.len: result += a[i] * b[i]

proc normalize(v: var seq[float32]): float32 =
  var n = 0.0'f32
  for x in v: n += x * x
  result = sqrt(n)
  if result > 1e-12'f32:
    for i in 0 ..< v.len: v[i] /= result

proc principalAxes*(rows: openArray[float32], n, dim: int,
                    iters = 200): tuple[v1, v2: seq[float32], l1, l2: float32] =
  ## First two principal components of `n` row vectors of length `dim`.
  if n == 0 or dim == 0:
    return (newSeq[float32](dim), newSeq[float32](dim), 0.0'f32, 0.0'f32)
  var mean = newSeq[float32](dim)
  for i in 0 ..< n:
    let row = i * dim
    for j in 0 ..< dim: mean[j] += rows[row + j]
  for j in 0 ..< dim: mean[j] /= n.float32
  # covariance (dim x dim, symmetric)
  var cov = newSeq[float32](dim * dim)
  for i in 0 ..< n:
    let row = i * dim
    for a in 0 ..< dim:
      let xa = rows[row + a] - mean[a]
      if xa == 0.0'f32: continue
      for b in a ..< dim:
        let v = xa * (rows[row + b] - mean[b])
        cov[a * dim + b] += v
        if a != b: cov[b * dim + a] += v
  let scale = if n > 1: 1.0'f32 / (n - 1).float32 else: 0.0'f32
  for i in 0 ..< cov.len: cov[i] *= scale

  proc topEigen(): tuple[v: seq[float32], l: float32] =
    ## Power iteration with a deterministic start that is unlikely to be
    ## orthogonal to the leading eigenvector.
    var v = newSeq[float32](dim)
    for i in 0 ..< dim: v[i] = 1.0'f32 + (i mod 7).float32 * 0.125'f32
    discard normalize(v)
    var tmp = newSeq[float32](dim)
    for _ in 0 ..< iters:
      matVec(cov, dim, v, tmp)
      if normalize(tmp) < 1e-12'f32:
        return (newSeq[float32](dim), 0.0'f32)
      swap(v, tmp)
    result.v = v
    matVec(cov, dim, v, tmp)
    result.l = dot(v, tmp)

  let (v1, l1) = topEigen()
  # deflate and orthogonalise, then take the second component
  var cov2 = cov
  for a in 0 ..< dim:
    for b in 0 ..< dim:
      cov2[a * dim + b] -= l1 * v1[a] * v1[b]
  var v2 = newSeq[float32](dim)
  for i in 0 ..< dim: v2[i] = 1.0'f32 + (i mod 5).float32 * 0.125'f32
  discard normalize(v2)
  var tmp2 = newSeq[float32](dim)
  for _ in 0 ..< iters:
    # keep it orthogonal to v1 (deflation can drift)
    let proj = dot(v2, v1)
    for i in 0 ..< dim: v2[i] -= proj * v1[i]
    matVec(cov2, dim, v2, tmp2)
    if normalize(tmp2) < 1e-12'f32:
      v2 = newSeq[float32](dim)
      break
    swap(v2, tmp2)
  var l2 = 0.0'f32
  if v2.len == dim:
    matVec(cov2, dim, v2, tmp2)
    l2 = dot(v2, tmp2)

  result.v1 = v1
  result.v2 = v2
  result.l1 = max(0.0'f32, l1)
  result.l2 = max(0.0'f32, l2)

proc project*(m: TransEModel, includeRelations = true): Projection =
  ## PCA of the entity rows (relations projected with the same axes, drawn
  ## differently). The caller uses `points` for both the SVG and the TSV.
  let dim = m.dim
  let n = m.nEntities
  if n == 0 or dim == 0: return
  var rows = newSeq[float32](n * dim)
  let ep = m.entityEmb.storage.raw_buffer
  for i in 0 ..< n * dim: rows[i] = ep[i]
  let (v1, v2, l1, l2) = principalAxes(rows, n, dim)
  result.var1 = l1
  result.var2 = l2
  var mean = newSeq[float32](dim)
  for i in 0 ..< n:
    for j in 0 ..< dim: mean[j] += rows[i * dim + j]
  for j in 0 ..< dim: mean[j] /= n.float32

  proc coords(row: ptr UncheckedArray[float32], base: int): (float32, float32) =
    var x, y = 0.0'f32
    for j in 0 ..< dim:
      let c = row[base + j] - mean[j]
      x += c * v1[j]
      y += c * v2[j]
    result = (x, y)

  for e in 0 ..< n:
    let (x, y) = coords(ep, e * dim)
    result.points.add ProjectionPoint(name: m.entities.names[e], kind: "entity",
                                      x: x, y: y)
  if includeRelations:
    let rp = m.relEmb.storage.raw_buffer
    for r in 0 ..< m.nRelations:
      let (x, y) = coords(rp, r * dim)
      result.points.add ProjectionPoint(name: "REL/" & m.relations.names[r],
                                        kind: "relation", x: x, y: y)

proc escapeXml(s: string): string =
  result = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
  result = result.replace("\"", "&quot;")

proc writeProjectionTsv*(p: Projection, path: string) =
  ## `name<TAB>kind<TAB>x<TAB>y`, one line per vector.
  var f = open(path, fmWrite)
  defer: f.close()
  f.writeLine("name\tkind\tx\ty")
  for pt in p.points:
    f.writeLine(&"{pt.name}\t{pt.kind}\t{pt.x:.6f}\t{pt.y:.6f}")

proc writeProjectionSvg*(p: Projection, path: string,
                         width = 1100, height = 780, labels = true) =
  ## Self-contained scatter: entities are circles, relations are diamonds,
  ## both labelled. Axes cross at the centroid, which is the PCA origin.
  const
    MarginL = 70.0
    MarginR = 70.0
    MarginT = 60.0
    MarginB = 60.0
  var f = open(path, fmWrite)
  defer: f.close()
  var minX, maxX, minY, maxY: float32
  if p.points.len > 0:
    minX = p.points[0].x; maxX = p.points[0].x
    minY = p.points[0].y; maxY = p.points[0].y
    for pt in p.points:
      minX = min(minX, pt.x); maxX = max(maxX, pt.x)
      minY = min(minY, pt.y); maxY = max(maxY, pt.y)
  # pad degenerate axes so a single point / a line still renders
  let spanX = max(maxX - minX, 1e-6'f32)
  let spanY = max(maxY - minY, 1e-6'f32)
  let padX = spanX * 0.06 + 1e-6'f32
  let padY = spanY * 0.06 + 1e-6'f32
  minX -= padX; maxX += padX; minY -= padY; maxY += padY
  let plotW = width.float32 - MarginL - MarginR
  let plotH = height.float32 - MarginT - MarginB
  proc sx(x: float32): float32 = MarginL + (x - minX) / (maxX - minX) * plotW
  proc sy(y: float32): float32 =
    MarginT + plotH - (y - minY) / (maxY - minY) * plotH

  f.writeLine &"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="#fbfbfd"/>
  <text x="{MarginL.int}" y="30" font-family="Helvetica,Arial,sans-serif" font-size="18" fill="#222">Artemis — PCA projection of trained embeddings ({p.points.len} vectors)</text>
  <text x="{MarginL.int}" y="48" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#777">entity = circle · relation = diamond · axes cross at the centroid · variance {p.var1:.3g} / {p.var2:.3g}</text>"""
  # plot frame
  f.writeLine &"""  <rect x="{MarginL:.1f}" y="{MarginT:.1f}" width="{plotW:.1f}" height="{plotH:.1f}" fill="#ffffff" stroke="#e3e3ea"/>"""
  # centroid axes
  let (ox, oy) = (sx(0.0'f32), sy(0.0'f32))
  if ox >= MarginL and ox <= MarginL + plotW:
    f.writeLine &"""  <line x1="{ox:.1f}" y1="{MarginT:.1f}" x2="{ox:.1f}" y2="{MarginT + plotH:.1f}" stroke="#e8e8f0"/>"""
  if oy >= MarginT and oy <= MarginT + plotH:
    f.writeLine &"""  <line x1="{MarginL:.1f}" y1="{oy:.1f}" x2="{MarginL + plotW:.1f}" y2="{oy:.1f}" stroke="#e8e8f0"/>"""
  for pt in p.points:
    let (cx, cy) = (sx(pt.x), sy(pt.y))
    if pt.kind == "relation":
      f.writeLine &"""  <rect x="{cx - 4:.1f}" y="{cy - 4:.1f}" width="8" height="8" transform="rotate(45 {cx:.1f} {cy:.1f})" fill="#e08a00" fill-opacity="0.85"/>"""
    else:
      f.writeLine &"""  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.2" fill="#2f6fd0" fill-opacity="0.8"/>"""
  if labels:
    for pt in p.points:
      let (cx, cy) = (sx(pt.x), sy(pt.y))
      let colour = if pt.kind == "relation": "#a26200" else: "#1d4e91"
      f.writeLine &"""  <text x="{cx + 7:.1f}" y="{cy + 4:.1f}" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="{colour}">{escapeXml(pt.name)}</text>"""
  f.writeLine "</svg>"

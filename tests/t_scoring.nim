import std/[unittest, math]
import artemis/types
import artemis/scoring
import helpers

# Fixture: e0=(0,0), e1=(1,0), e2=(0,2); r0=(1,0)
let m = modelFromArrays(
  @["e0", "e1", "e2"], @["r0"], 2,
  @[0.0'f32, 0,  1, 0,  0, 2],
  @[1.0'f32, 0])

suite "scoring: distances (hand-computed)":
  test "exact zero for perfect translation":
    check distance(m, 0, 0, 1, nkL1) == 0.0'f32
    check distance(m, 0, 0, 1, nkL2) == 0.0'f32

  test "L1 and L2 values":
    # d(e1, r0, e2): (1+1-0, 0+0-2) = (2, -2) -> L1=4, L2=2*sqrt(2)
    check distance(m, 1, 0, 2, nkL1) == 4.0'f32
    check abs(distance(m, 1, 0, 2, nkL2) - 2.0'f32 * sqrt(2.0'f32)) < 1e-6'f32
    # d(e0, r0, e0): (1, 0) -> 1 under both norms
    check distance(m, 0, 0, 0, nkL1) == 1.0'f32
    check distance(m, 0, 0, 0, nkL2) == 1.0'f32

  test "scoreAllTails / scoreAllHeads cover every entity":
    var buf: seq[float32]
    scoreAllTails(m, 0, 0, nkL1, buf)
    check buf.len == 3
    check buf[0] == 1.0'f32 # d(e0, r0, e0)
    check buf[1] == 0.0'f32 # d(e0, r0, e1)
    check buf[2] == 3.0'f32 # d(e0, r0, e2): (1, -2) -> 3
    scoreAllHeads(m, 0, 2, nkL1, buf)
    check buf[0] == 3.0'f32 # d(e0, r0, e2)
    check buf[1] == 4.0'f32 # d(e1, r0, e2)
    check buf[2] == 1.0'f32 # d(e2, r0, e2)

import std/[unittest, math, sets, random, os]
import artemis/types
import artemis/scoring
import artemis/calibrate
import artemis/embeddings
import artemis/train
import artemis/sampling
import helpers

suite "calibrate: softmax probabilities":
  test "probabilities are a softmax over the scores":
    let s = @[1.0'f32, 2.0'f32, 3.0'f32]
    # tau = 1: p_i ∝ exp(-s_i)
    let p0 = softmaxProbability(s, 0, 1.0'f32)
    let p1 = softmaxProbability(s, 1, 1.0'f32)
    let p2 = softmaxProbability(s, 2, 1.0'f32)
    check abs(p0 + p1 + p2 - 1.0) < 1e-12
    check p0 > p1 and p1 > p2     # lower dissimilarity wins
    check abs(p0 - exp(-1.0) / (exp(-1.0) + exp(-2.0) + exp(-3.0))) < 1e-12

  test "a small temperature sharpens, a large one flattens":
    let s = @[0.0'f32, 1.0'f32]
    check softmaxProbability(s, 0, 0.01'f32) > 0.999
    check abs(softmaxProbability(s, 0, 1e6'f32) - 0.5) < 1e-5
    check softmaxProbability(s, 0, 0.0'f32) == 0.0 # uncalibrated

  test "identical scores give uniform probabilities":
    let s = @[2.0'f32, 2.0, 2.0, 2.0]
    check abs(softmaxProbability(s, 1, 3.0'f32) - 0.25) < 1e-12

suite "calibrate: fitting the temperature":
  # A model where the true tails are clearly better but on a large scale: the
  # fitted tau should shrink the effective scale (tau > 1 for spread-out
  # distances, i.e. it softens), and the NLL must improve over tau = 1.
  proc fitted(): Calibration =
    let m = modelFromArrays(
      @["a", "b", "c", "d"], @["r"], 2,
      @[0.0'f32, 0,  1, 0,  2, 0,  3, 0],
      @[1.0'f32, 0])
    var v = view(m)
    v.norm = nkL1
    let data = @[Triple(head: 0, rel: 0, tail: 1),
                 Triple(head: 1, rel: 0, tail: 2)]
    fitTemperature(v, data, limit = 10)

  test "the fit improves the NLL and reports a usable temperature":
    let c = fitted()
    check c.n == 2
    check c.temperature > 0.0'f32
    check c.nll <= c.nllUnit                 # never worse than tau = 1
    check c.meanProb >= 0.0 and c.meanProb <= 1.0

  test "sampling is deterministic and spread out":
    var data: seq[Triple]
    for i in 0 ..< 100: data.add Triple(head: 0, rel: 0, tail: i mod 4)
    check sampleTriples(data, 10).len == 10
    check sampleTriples(data, 10) == sampleTriples(data, 10)
    check sampleTriples(data, 1000).len == 100
    check sampleTriples(data, 0).len == 100

  test "an empty calibration set is not an error":
    let m = modelFromArrays(@["a"], @["r"], 2, @[1.0'f32, 0], @[1.0'f32, 0])
    var v = view(m)
    v.norm = nkL1
    let c = fitTemperature(v, @[], limit = 10)
    check c.n == 0
    check c.temperature == 1.0'f32

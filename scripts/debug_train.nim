import std/[random, sets, math]
import artemis/types
import artemis/io
import artemis/embeddings
import artemis/scoring
import artemis/sampling
import artemis/gradients
import artemis/evaluate

var hp = HyperParams(dim: 10, epochs: 60, lr: 0.05'f32, norm: nkL1,
                     quiet: true, seed: 42)
let (data, ents, rels, _) = loadDataset("data/toy_family.tsv", hp)
var rng = initRand(42)
var m = initModel(ents, rels, hp, rng)

proc meanDist(m: TransEModel, data: seq[Triple]): float32 =
  for t in data: result += distance(m, t, nkL1)
  result /= data.len.float32

let truth = buildTruthSet(data.train)
let sampler = initSampler(data.train, rels.len, ents.len, true, truth)
var scratch = initScratch(hp.dim)
var order = data.train

echo "epoch 0 meanDist=", meanDist(m, data.train)
echo evaluate(m, data.train, truth, nkL1)
for epoch in 1 .. 60:
  rng.shuffle(order)
  var epochLoss = 0.0
  for pos in order:
    let neg = sampler.sample(rng, pos)
    epochLoss += trainStep(m, pos, neg, hp, scratch)
  if epoch mod 10 == 0:
    # also check relation vector magnitude
    var rNorm = 0.0'f32
    let rp = m.relEmb.storage.raw_buffer
    for i in 0 ..< rels.len * hp.dim: rNorm += abs(rp[i])
    echo "epoch ", epoch, " meanLoss=", epochLoss / order.len.float,
         " meanDist=", meanDist(m, data.train), " relL1sum=", rNorm
echo evaluate(m, data.train, truth, nkL1)

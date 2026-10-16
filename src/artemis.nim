## Artemis — TransE knowledge-graph embeddings in Nim.
##
## Library usage:
##
## .. code-block:: nim
##   import artemis
##   let (data, entities, relations, _) = loadDataset("train.tsv", HyperParams())
##   var rng = initRand(42)
##   var model = initModel(entities, relations, HyperParams(), rng)
##   ...
import artemis/types
import artemis/io
import artemis/embeddings
import artemis/scoring
import artemis/sampling
import artemis/gradients
import artemis/train
import artemis/evaluate
import artemis/persist

export types, io, embeddings, scoring, sampling, gradients, train, evaluate,
       persist

when isMainModule:
  import artemis/cli
  cli.main()

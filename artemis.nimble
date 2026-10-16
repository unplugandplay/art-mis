# Package
version       = "0.1.0"
author        = "unplugandplay"
description   = "Artemis — TransE knowledge-graph embeddings in Nim"
license       = "MIT"
srcDir        = "src"
bin           = @["artemis"]

# Dependencies
requires "nim >= 2.0.0"
requires "arraymancer >= 0.7.32"

# Tasks
task test, "Run the full test suite":
  exec "nim r tests/all.nim"

task bench, "Run the synthetic-scale release benchmark":
  exec "nim r -d:release scripts/bench_synthetic.nim"

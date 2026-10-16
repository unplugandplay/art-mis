#!/usr/bin/env python3
"""Sanity check: an Artemis embeddings TSV loads with numpy."""
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "embeddings.tsv"
names, vecs = [], []
with open(path) as f:
    for line in f:
        name, nums = line.rstrip("\n").split("\t")
        names.append(name)
        vecs.append([float(x) for x in nums.split(" ")])
m = np.array(vecs, dtype=np.float32)
assert m.ndim == 2 and m.shape[0] == len(names), "bad shape"
assert np.isfinite(m).all(), "non-finite values"
print(f"OK: {m.shape[0]} vectors of dim {m.shape[1]} loaded from {path}")
print(f"entity norms (first 5, ~1.0 expected): "
      f"{np.linalg.norm(m[:5], axis=1)}")

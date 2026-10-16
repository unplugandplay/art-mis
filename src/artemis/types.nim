## Core types for Artemis (TransE in Nim).
##
## Ids are integers; strings live only in `Vocab` at the IO boundary.
import std/[tables, hashes]
import arraymancer

const ArtemisVersion* = "0.3.0"
  ## Keep in sync with `version` in `artemis.nimble` (and the release tag);
  ## `artemis --version` prints it, so a built binary is self-identifying.

type
  Triple* = object
    ## A knowledge-graph fact as internal integer ids.
    head*, rel*, tail*: int

  RawTriple* = tuple
    ## A fact as raw strings, straight from the parser.
    head, relation, tail: string

  Vocab* = object
    ## Bidirectional name <-> id mapping. Ids follow first-appearance order,
    ## which keeps id assignment deterministic by construction.
    toId*: Table[string, int]
    names*: seq[string] ## id -> name

  NormKind* = enum
    nkL1, nkL2

  LrSchedule* = enum
    ## How the SGD learning rate changes across epochs.
    lsConstant    ## `lr` unchanged (v0.1.0 behaviour)
    lsExponential ## lr * lrDecay^(epoch-1), floored at lrMin
    lsLinear      ## lr decays to lrMin over the whole epoch budget

  Dataset* = object
    train*, valid*, test*: seq[Triple]

  HyperParams* = object
    dim*: int = 50
    margin*: float32 = 1.0
    lr*: float32 = 0.01
    lrSchedule*: LrSchedule = lsConstant ## see `scheduledLr`
    lrDecay*: float32 = 1.0 ## per-epoch factor for lsExponential
    lrMin*: float32 = 0.0   ## floor (exp) / final value (linear)
    epochs*: int = 1000
    batchSize*: int = 100
    negPerPos*: int = 1
    norm*: NormKind = nkL1
    bernoulli*: bool = true
    seed*: int64 = 42
    evalEvery*: int = 0   ## epochs between validation evals; 0 = off
    ckptEvery*: int = 0   ## epochs between checkpoints; 0 = only final
    patience*: int = 5    ## early-stop patience, in validation evals
    minCount*: int = 1    ## drop entities/relations seen fewer times
    delimiter*: char = '\t'
    quiet*: bool = false
    threads*: int = 1     ## evaluation worker threads (0 = one per core)

  TransEModel* = object
    ## Entity and relation embeddings. Row-major storage: row `i` of a
    ## `[n, dim]` tensor lives at `storage.raw_buffer[i*dim ..< (i+1)*dim]`.
    entities*, relations*: Vocab
    entityEmb*: Tensor[float32] ## [nEntities, dim]
    relEmb*: Tensor[float32]    ## [nRelations, dim]

proc `$`*(s: LrSchedule): string =
  case s
  of lsConstant: "constant"
  of lsExponential: "exponential"
  of lsLinear: "linear"

proc hash*(t: Triple): Hash =
  var h: Hash = 0
  h = h !& hash(t.head)
  h = h !& hash(t.rel)
  h = h !& hash(t.tail)
  result = !$h

proc `==`*(a, b: Triple): bool =
  a.head == b.head and a.rel == b.rel and a.tail == b.tail

proc `$`*(v: Vocab): string =
  "Vocab(size: " & $v.names.len & ")"

proc len*(v: Vocab): int = v.names.len

proc dim*(m: TransEModel): int {.inline.} = m.entityEmb.shape[1]

proc nEntities*(m: TransEModel): int {.inline.} = m.entityEmb.shape[0]

proc nRelations*(m: TransEModel): int {.inline.} = m.relEmb.shape[0]

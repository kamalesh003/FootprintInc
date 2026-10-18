"""
slabinc -- incgraph as an INVALIDATION PLANNER over SLAB-backed REGION EXECUTORS.

Why this exists (derived from the T4 logs, see bench2.py header)
----------------------------------------------------------------
In the tensor-valued graph every node output is a fresh tensor, so a batched call must
cat its inputs, split its outputs, (and clone the pieces so they do not pin the batch), and every
attention node concatenates K/V.  For a transformer that is ~11 copy-units per layer, about 6 ms of
pure copy traffic per step on a T4 -- the whole reason incgraph[none+batch] trails `ordinary` when
everything is dirty (0.93x at edit-first) and trails a hand-written K/V arena at edit-last.

Design
------
  * Data lives in preallocated SLABS indexed by chunk (x[l], q[l], k[l], v[l]); a region executor reads
    and writes SLICES of them in place.  No cat, no split, no clone, no pinning, fixed memory.
  * The graph keeps doing what it is good at: dependency tracking, demand-driven invalidation,
    topological levels, dirty-set discovery, versioned skipping.  Its node VALUES are tiny tokens
    (the chunk index); the tensors are a side effect of executing the node.
  * Dirty chunks of one (layer, kind) are coalesced into contiguous RUNS [lo,hi) and handed to the
    executor as ONE call per run.  A forward-moving causal edit at p dirties exactly [p, N).

Contract (weaker than incgraph's purity -- stated, not hidden)
-------------------------------------------------------------
  * Executors may mutate their slabs, but must be IDEMPOTENT: re-running a run with the same inputs
    rewrites the same bytes.  (A failed evaluation leaves nodes dirty, so it is simply re-run.)
  * Old values are overwritten in place, so value-based early cutoff is not available here
    (cutoff="none").  Cheap exact gating at the INPUT is provided instead (`gate=True`): an edit whose
    data is bitwise identical to the stored chunk is dropped before it dirties anything.
  * qkv(l, i) must precede blk(l, j>=i) (guaranteed: blk depends on qkv of every chunk <= its own).

Executor protocol (duck-typed):
    write_input(i, data)        copy chunk i's input into x[0]
    same_input(i, data) -> bool bitwise equality with the stored x[0] chunk (only used if gate=True)
    qkv_run(l, lo, hi)          read x[l][lo:hi];  write q/k/v[l][lo:hi]
    blk_run(l, lo, hi)          read q[l][lo:hi], k/v[l][:hi], x[l][lo:hi];  write x[l+1][lo:hi]
    output(n)                   x[L][:n] (view)

Prior art this builds on (not claimed as novel): demand-driven incremental computation (Hammer, Khoo,
Hicks, Foster, "Adapton", PLDI 2014); preallocated K/V arenas and prefix reuse as in LLM serving (e.g.
Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention", SOSP 2023; Zheng et al.,
SGLang/RadixAttention, NeurIPS 2024); operation batching in dynamic graphs (Neubig, Goldberg, Dyer,
NIPS 2017).  What is specific here: the dirty set is DISCOVERED by a general dependency graph instead of
being hard-coded as "suffix".  Other dependency shapes (e.g. windowed attention, whose dirty set is a
cone) need both different edges AND an executor that restricts its key range; that is NOT implemented
or tested here.
"""
from __future__ import annotations

from incgraph2 import Graph


def runs(idx):
    """sorted contiguous [lo, hi) runs covering the chunk indices in idx."""
    out = []
    for i in sorted(idx):
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


class SlabPipeline:
    """L layers x n chunks.  Per (l, i): qkv(l,i) and blk(l,i) (attention + MLP).

    blk(l, i) depends on x[l][i], qkv(l, i) and qkv(l, j) for all j < i (the K/V it attends to).
    """

    def __init__(self, ex, layers, gate=False):
        self.ex, self.L, self.gate = ex, layers, gate
        self.g = Graph(cutoff="none")
        self.xn = [[] for _ in range(layers + 1)]
        self.qn = [[] for _ in range(layers)]
        self.n = 0
        self.gated = 0

    # ---- executors wrapped as graph node functions -------------------------
    def _bq(self, l, al):
        idx = [a[0] for a in al]
        for lo, hi in runs(idx):
            self.ex.qkv_run(l, lo, hi)
        return idx

    def _bb(self, l, al):
        idx = [a[1] for a in al]
        for lo, hi in runs(idx):
            self.ex.blk_run(l, lo, hi)
        return idx

    # ---- construction -------------------------------------------------------
    def append_chunk(self, data):
        g, i = self.g, self.n
        self.ex.write_input(i, data)
        self.xn[0].append(g.input(f"x.{i}", i, group="x"))
        for l in range(self.L):
            qkv = g.op(f"qkv{l}.{i}", lambda t, l=l, i=i: self._bq(l, [[i]])[0], [self.xn[l][i]],
                       batch=(("qkv", l), lambda al, l=l: self._bq(l, al)), group=f"qkv{l}")
            self.qn[l].append(qkv)
            blk = g.op(f"blk{l}.{i}", lambda t, q, *prev, l=l, i=i: self._bb(l, [[t, i]])[0],
                       [self.xn[l][i], qkv] + self.qn[l][:i],
                       batch=(("blk", l), lambda al, l=l: self._bb(l, al)), group=f"blk{l}")
            self.xn[l + 1].append(blk)
        self.n += 1

    def edit(self, i, data):
        if self.gate and self.ex.same_input(i, data):
            self.gated += 1
            return False
        self.ex.write_input(i, data)
        self.g.set_input(self.xn[0][i], i)
        return True

    def evaluate(self):
        self.g.evaluate(self.xn[self.L])
        return self.ex.output(self.n)

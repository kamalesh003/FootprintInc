"""
regioninc -- interval-footprint REGION PLANNER for slab executors (replaces the per-chunk edge graph
inside SlabPipeline).

Why (measured, see prof.py): slabinc.SlabPipeline gives blk(l,i) one graph edge per earlier chunk
(qkv(l,0..i-1)).  That is O(L*N^2) edges, and every evaluate walks them three times (dirty marking,
need discovery, version-key validation `_plan`).  No-op-executor timings on the planner alone:

    N=16: first 0.88 ms  mid 0.44 ms  last 0.10 ms
    N=64: first 8.7  ms  mid 3.9  ms  last 0.12 ms        (x10 for 4x the chunks)
    N=128: first 43  ms  mid 21   ms  last 0.20 ms        (x5 again)

The slab executors never needed per-node values, versions or cutoff (cutoff="none" by construction),
only "which chunks must be recomputed".  That question is answered exactly by INTERVAL ALGEBRA:

  * every chunk-indexed buffer carries a bitmask of chunks that need re-execution per consumer op
  * a read footprint maps a changed set of producer chunks to the consumer chunks that must re-run:
        point        i reads {i}            : identity
        prefix       i reads [0, i]         : [lowest changed, n)         (causal attention)
        window(w)    i reads [i-w+1, i]     : changed set smeared right by w-1   (a CONE, not a suffix)
  * ops are registered in topological order; an op's finished runs are propagated to its readers.

Cost per evaluate: O(#ops + #runs) big-int operations, independent of N.  Construction: O(1) per op.

Contract (unchanged from slabinc): executors are idempotent.  Failure safety: a run is removed from an
op's todo set only AFTER the executor call returned, and its propagation to readers happens at the same
point, so a raising executor leaves exactly the unfinished work pending and a retry converges.

Not implemented / not claimed: value-based early cutoff (impossible with in-place slabs), and for window
footprints the EXECUTOR must also restrict its key range -- this module only plans.
"""
from __future__ import annotations


# ------------------------------------------------------------------ bitmask helpers
def mask_runs(m):
    """contiguous [lo, hi) runs of the set bits of m, ascending.  O(#runs)."""
    out = []
    while m:
        lo = (m & -m).bit_length() - 1
        t = m >> lo
        k = ((t + 1) & ~t).bit_length() - 1          # trailing ones of t
        out.append((lo, lo + k))
        m ^= ((1 << k) - 1) << lo
    return out


def bits(idx):
    m = 0
    for i in idx:
        m |= 1 << i
    return m


# ------------------------------------------------------------------ read footprints: (changed_mask, n) -> consumer mask
def point(m, n):
    return m


def prefix(m, n):
    if not m:
        return 0
    lo = (m & -m).bit_length() - 1
    return ((1 << n) - 1) & ~((1 << lo) - 1)


def window(w):
    if w < 1:
        raise ValueError("window must be >= 1")

    def fp(m, n):
        acc, span = m, 1                              # acc = OR_{k<span} m << k
        while span < w:
            step = min(span, w - span)
            acc |= acc << step
            span += step
        return acc & ((1 << n) - 1)
    return fp


# ------------------------------------------------------------------ planner
class RegionPlanner:
    def __init__(self):
        self.ops, self.todo = [], []
        self.readers = {}                             # buffer -> [(op index, footprint)]
        self.producer = {}                            # buffer -> op index
        self.stats = {}
        self.reset_stats()

    def reset_stats(self):
        self.stats = dict(executed=0, skipped=0, cutoff=0, syncs=0, checks=0, unchecked=0,
                          batch_calls=0, batched_nodes=0, eq_errors=0, calls=0)

    def op(self, run, reads, writes=None):
        """run(lo, hi) executes chunks [lo, hi).  reads: [(buffer, footprint)].  Ops must be added in
        topological order: every read buffer is either an external input or written by an earlier op."""
        idx = len(self.ops)
        if writes is not None:
            if writes in self.producer:
                raise ValueError(f"buffer {writes!r} already has a writer")
            self.producer[writes] = idx
        for buf, _ in reads:
            if buf == writes:
                raise ValueError("op reads its own output")
        for buf, fp in reads:
            self.readers.setdefault(buf, []).append((idx, fp))
        self.ops.append((run, writes))
        self.todo.append(0)
        return idx

    def touch(self, buf, mask, n):
        """chunks `mask` of `buf` changed: schedule exactly the reader chunks that depend on them."""
        for idx, fp in self.readers.get(buf, ()):
            self.todo[idx] |= fp(mask, n)

    def pending(self):
        return sum(bin(m).count("1") for m in self.todo)

    def evaluate(self, n):
        st = self.stats
        for idx, (run, writes) in enumerate(self.ops):
            m = self.todo[idx]
            for lo, hi in mask_runs(m):
                run(lo, hi)                           # may raise: this run stays pending
                done = ((1 << (hi - lo)) - 1) << lo
                self.todo[idx] &= ~done
                st["executed"] += hi - lo
                st["calls"] += 1
                if hi - lo > 1:
                    st["batch_calls"] += 1
                    st["batched_nodes"] += hi - lo
                if writes is not None:
                    self.touch(writes, done, n)


# ------------------------------------------------------------------ drop-in for slabinc.SlabPipeline
class RegionSlabPipeline:
    """Same API/contract as slabinc.SlabPipeline (append_chunk / edit / evaluate, gate, .g.stats)."""

    def __init__(self, ex, layers, gate=False):
        self.ex, self.L, self.gate = ex, layers, gate
        self.pl = self.g = RegionPlanner()            # `.g` keeps `pl.g.stats` / `pl.g.reset_stats()` working
        self.n, self.gated = 0, 0
        for l in range(layers):
            self.pl.op(lambda lo, hi, l=l: ex.qkv_run(l, lo, hi), [(("x", l), point)], ("qkv", l))
            self.pl.op(lambda lo, hi, l=l: ex.blk_run(l, lo, hi),
                       [(("qkv", l), prefix), (("x", l), point)], ("x", l + 1))

    def append_chunk(self, data):
        i = self.n
        self.ex.write_input(i, data)
        self.n += 1
        self.pl.touch(("x", 0), 1 << i, self.n)

    def edit(self, i, data):
        if self.gate and self.ex.same_input(i, data):
            self.gated += 1
            return False
        self.ex.write_input(i, data)
        self.pl.touch(("x", 0), 1 << i, self.n)
        return True

    def evaluate(self):
        self.pl.evaluate(self.n)
        return self.ex.output(self.n)

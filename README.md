# FootprintInc

---

# 1. The problem your approach solves

A chunked causal transformer receives edits, either replacing chunk `p` or appending a chunk. Recomputing the whole sequence every time is wasteful. Because attention is causal, an edit at chunk `p` only changes chunks `p…N-1`. Your code works out which chunks that is and runs only those.

It was built in three stages, each fixing a measured bottleneck of the previous one:

| Stage | File | Idea | Bottleneck it exposed |
|---|---|---|---|
| 1 | `incgraph2.py` | General dependency graph: tensor-valued nodes, versioned skipping, early cutoff, batching | cat/split/clone copies (~11 copy-units per layer, ~6 ms/step on a T4) |
| 2 | `slabinc.py` | Same graph used only as an invalidation planner; data lives in preallocated slabs and is written in place | Edge count O(L·N²), so planning cost explodes with N |
| 3 | `regioninc.py` | Replace the graph with bitmask interval algebra | Cost is now O(#ops + #runs), independent of N |
| – | `bench2.py` | Executors, baselines, scenarios, correctness checks, timing | – |

---

# 2. The computation being planned

For each layer `l` and chunk `i` there are two units of work:

```
qkv(l,i):  x[l][i]  → q,k,v[l][i]            (layernorm + QKV projection; position-wise)
blk(l,i):  q[l][i], k/v[l][0..i], x[l][i]  → x[l+1][i]
           (causal attention + out-proj + residual + MLP)
```

The dependency facts the whole design rests on:

- `qkv(l,i)` reads only `x[l][i]`. This is a **point** dependency.
- `blk(l,i)` reads `q[l][i]`, `x[l][i]` (residual), and `k,v[l][j]` for every `j ≤ i`. This is a **prefix** dependency.
- `blk(l,i)` writes `x[l+1][i]`, which feeds layer `l+1`.

So an edit of chunk `p` dirties `qkv(0,p)`, then `blk(0,p..N-1)`, then `qkv(1,p..N-1)`, and so on. In the first layer the dirty set grows from `{p}` to the suffix `[p,N)`.

---

# 3. incgraph2.py: the general engine

**Node.** Each node holds `fn`, `deps`, `users`, `level`, `value`, `version`, `seen`, `dirty`, `force`, `struct`, plus batch and group info.

**Construction**
1. `input(name, value)` creates a level-0 node with a pending value.
2. `op(name, fn, deps, batch=, group=)` sets `level = 1 + max(dep levels)` and registers the node in each dependency's `users`.
3. `batch=(key, batch_fn)` lets same-level dirty nodes with the same key run in one call.

**Invalidation**
- `set_input(n, value)` stores the pending value and calls `_mark_dirty`.
- `_mark_dirty` walks `users` transitively with a stack. It stops at nodes that are already dirty.

**Evaluate(targets)**
1. **Need discovery.** Walk back from the dirty targets through dirty deps only. This gives the set `need`.
2. **Group by level.** Run levels in ascending order.
3. **Per level (`_run_level_body`).** For each node, `_plan` decides what to do:
   - An input consumes its pending value.
   - An op builds the key `cur = (struct, ((dep.id, dep.version) …))`.
   - If `version > 0`, the node is not forced, and `seen == cur`, the node is **skipped** (`skipped += 1`) and marked clean.
   - Otherwise it must run. Nodes with a batch key go into `groups[key]`; the rest go into `singles`.
4. **Execution.**
   - Singles go through `_exec_one`.
   - A group of one node goes through `_exec_one`.
   - A larger group makes **one** `batch_fn` call with every node's argument list. The result length is checked against the group size, and `batch_calls` and `batched_nodes` are updated.
5. **`_finish`.** The node is marked clean. If `_should_check` is true, the old and new values are compared with `eq`, which may be a lazy 0-d GPU bool, and the result is deferred. Otherwise the new value is committed and `version += 1`.
6. **`_settle`** resolves all deferred flags of a level with **one host sync per device** (`torch_resolve`). If the output is equal, the node is a cutoff hit: the version stays unchanged, so downstream nodes' version keys still match and they skip. If not, the value is committed and the version is bumped.

**Cutoff modes**
- `none`: never compare, never sync.
- `exact`: bitwise equality.
- `tol`: tolerance `atol`.
- `adaptive`: per-group exponential moving average of the hit rate. It stops checking when the rate falls below `off_below`, and re-probes every `probe_every` skipped checks. A skipped check only forgoes a skip, so it is always safe.

**Safety**
- If any node, `eq` or `resolve` raises, `_commit_changed` commits already-computed outputs as "changed". This prevents the stale-value bug listed in the v2.1 notes.
- `torch_eq` compares structure first. It de-duplicates views only when the base and the view geometry (offset and strides) both match.

**Dynamic features**
- `rewire` changes deps or `fn`. It checks for cycles, re-levels descendants, bumps `struct`, and marks the node dirty.
- `remove` deletes a node that has no dependents.
- Stateful nodes use `state_in` and `state_out`, and `tick()` advances them. `always_tick` forces re-execution.
- `invalidate_all` is used by the `chunked-full` baseline.

---

# 4. slabinc.py: graph as planner, slabs as storage

**Key shift.** Node values are tiny tokens (the chunk index). The tensors are a side effect of executing the node, living in preallocated slabs. So there is no cat, split or clone, and memory is fixed.

**Contract.** Executors must be idempotent: re-running a run with the same inputs rewrites the same bytes. In-place overwrite means old values are gone, so the graph is built with `cutoff="none"`. Instead, an optional **input gate** is provided: with `gate=True`, an edit that is bitwise identical to the stored chunk is dropped before it dirties anything (`gated += 1`).

**Executor protocol:** `write_input`, `same_input`, `qkv_run(l,lo,hi)`, `blk_run(l,lo,hi)`, `output(n)`.

**`append_chunk(data)`**
1. `write_input(i, data)`.
2. Create input node `x.i`.
3. For each layer, create `qkv(l,i)` with deps `[x[l][i]]`.
4. For each layer, create `blk(l,i)` with deps `[x[l][i], qkv(l,i)] + qkv(l,0..i-1)`. This is the O(i) edge fan-in.
5. `x[l+1][i]` is the `blk` node.

**`edit(i, data)`**
1. If gated and identical, return `False`.
2. Otherwise `write_input` and `g.set_input(x0[i], i)`.

**`evaluate()`**
1. `g.evaluate(xn[L])` runs the generic algorithm from section 3.
2. Batch functions `_bq` and `_bb` take the list of dirty node arguments, extract chunk indices, sort them, coalesce them into contiguous runs with `runs(idx)`, and call `qkv_run` or `blk_run` once per run. A forward-moving causal edit at `p` produces exactly `[p,N)`.
3. Return `ex.output(n)`.

**Why it is slow at scale** (measured in `prof.py`, quoted in your regioninc docstring). There are O(L·N²) edges, and every evaluate walks them three times: dirty marking, need discovery, and version-key validation. No-op executor planner timings from your docstring:

| N | first chunk edit | mid edit | last chunk edit |
|---|---|---|---|
| 16 | 0.88 ms | 0.44 ms | 0.10 ms |
| 64 | 8.7 ms | 3.9 ms | 0.12 ms |
| 128 | 43 ms | 21 ms | 0.20 ms |

---

# 5. regioninc.py: interval-footprint planner

The slab executors never needed per-node values, versions or cutoff. They only need the answer to "which chunks must be recomputed?", and that question can be answered exactly with bitmask algebra.

## 5.1 Bitmask representation
Each chunk-indexed buffer has a Python big-int bitmask, where bit `i` means "chunk `i` needs re-execution" (per consumer op).

**`mask_runs(m)`** turns a mask into contiguous `[lo, hi)` runs:
1. `lo` is the index of the lowest set bit: `(m & -m).bit_length() - 1`.
2. `t = m >> lo`; the number of trailing ones is `k = ((t+1) & ~t).bit_length() - 1`.
3. Emit `(lo, lo+k)` and clear those bits: `m ^= ((1<<k)-1) << lo`.

Example: `m = 0b11011000` gives runs `(3,5)` and `(6,8)`.

**`bits(idx)`** builds a mask from indices.

## 5.2 Read footprints: (changed mask, n) → consumer mask

| Footprint | Meaning | Rule |
|---|---|---|
| `point` | chunk `i` reads `{i}` | identity |
| `prefix` | chunk `i` reads `[0,i]` (causal attention) | everything from the lowest changed bit up to `n`: `((1<<n)-1) & ~((1<<lo)-1)` |
| `window(w)` | chunk `i` reads `[i-w+1, i]` | changed set smeared right by `w-1`, a **cone**, not a suffix. It is computed by doubling shifts: `acc \|= acc << step`, with `step = min(span, w-span)`, then masked to `n` bits |

For `w=3` and a change at bit 2: after one step the set is `{2,3}`, after the second `{2,3,4}`. Correct.

## 5.3 RegionPlanner

**State**
- `ops`: list of `(run, writes)`.
- `todo`: one bitmask per op.
- `readers`: buffer → list of `(op index, footprint)`.
- `producer`: buffer → op index.
- `stats`: kept key-compatible with the graph (`skipped`, `cutoff`, `syncs` etc. stay 0).

**`op(run, reads, writes)`**, registration:
1. Ops must be added in topological order.
2. A buffer can have only one writer.
3. An op cannot read its own output.
4. Register the op in `readers` for each buffer it reads.
5. Append `todo = 0`.

**`touch(buf, mask, n)`**: for each reader `(idx, fp)` of `buf`, do `todo[idx] |= fp(mask, n)`.

**`evaluate(n)`**
1. Iterate ops in registration order.
2. For each op, split `todo[idx]` into runs.
3. For each run, call `run(lo, hi)`. This may raise.
4. **Only after the call returns**, clear those bits from `todo[idx]` and update stats.
5. Immediately `touch(writes, done_bits, n)` to schedule downstream readers.

Cost per evaluate is O(#ops + #runs) big-int operations. It does not depend on N.

## 5.4 RegionSlabPipeline (drop-in for SlabPipeline)

Registration, per layer `l`, in this order:
```
op(qkv_run(l), reads=[(("x",l), point)],                                   writes=("qkv",l))
op(blk_run(l), reads=[(("qkv",l), prefix), (("x",l), point)],              writes=("x",l+1))
```
- `append_chunk`: `write_input(i)`, `n += 1`, `touch(("x",0), 1<<i, n)`. Note that `n` is updated first, so the prefix mask is bounded correctly.
- `edit`: optional gate, then `write_input` and `touch(("x",0), 1<<i, n)`.
- `evaluate`: `pl.evaluate(n)`, then return `ex.output(n)`.
- `self.g = self.pl`, so `pl.g.stats` and `pl.g.reset_stats()` still work.

---

# 6. Worked trace (L=2, N=6, edit chunk 3)

Op order is `0=qkv0, 1=blk0, 2=qkv1, 3=blk1`.

| Step | Action | Resulting `todo` change |
|---|---|---|
| edit | `touch(("x",0), bit3)` | `todo[0]={3}` (point), `todo[1]={3}` (point) |
| op0 | `qkv_run(0,3,4)`, then `touch(("qkv",0), {3})` with prefix | `todo[1] |= {3,4,5}` |
| op1 | `blk_run(0,3,6)` (one call), then `touch(("x",1), {3,4,5})` | `todo[2]={3,4,5}`, `todo[3]={3,4,5}` |
| op2 | `qkv_run(1,3,6)`, then touch prefix | `todo[3] |= {3,4,5}` |
| op3 | `blk_run(1,3,6)` | done |

Four executor calls in total, so `2L` calls, and chunks 0–2 are never touched.

---

# 7. bench2.py: executors, baselines, verification

## 7.1 Executors

**TorchSlabExecutor** has slabs `X[0..L]` of shape `[B,T,D]` and `Q/K/V[l]` of shape `[B,H,T,Dh]`.
- `qkv_run(l,lo,hi)`: layernorm, linear, then write q/k/v into the slices.
- `blk_run(l,lo,hi)`: take `e = hi*C`. Run `sdpa_tail_causal(Q[s], K[:e], V[:e])`, then `attn_out` (residual) and `mlp_block`, and write into `X[l+1][s]`.
- Keys are limited to `[:e]` because chunks `≥ hi` can't be attended to. The queries are the *last* rows of `K[:e]`, which is why **lower-right-aligned causal** attention is used (`causal_lower_right`, self-tested once per device and dtype, with an explicit-mask fallback).

**TorchPackedExecutor** removes dispatch and copy overhead:
- One packed slab `QKV[l] = [B,T,3,H,Dh]`. The GEMM writes straight into it with `mm(out=)`, and q/k/v are strided views, so there are no per-layer copy kernels and no Q slab.
- Weights are pre-transposed as views.
- Residuals are fused into GEMM epilogues with `addmm`, and the output is written in place into `X[l+1]`.
- B == 1 takes the fast path. B > 1 falls back to `copy_` / `add(out=)`, which is correct but slower.

## 7.2 Systems compared
`ordinary` (full causal forward every step), `handkv` (chunk-by-chunk prefix reuse), `suffixkv` (strongest hand-written: one batched forward over the dirty suffix), `chunked-full` (graph with `invalidate_all`), `incgraph[none/exact/…+batch/+fuse]`, `incgraph[slab]`, `incgraph[region]`, `incgraph[region+packed]`, and `[memo]` variants. The `[memo]` variants skip bitwise-identical edits at the input, so no-op scenarios are compared fairly.

## 7.3 Scenarios
`edit-first`, `edit-mid`, `edit-last`, `noop-mid`, `append`, and an optional `--sweep` that edits every position and reports the crossover against `ordinary`. `--interleave` round-robins systems to cancel GPU clock drift.

## 7.4 Correctness
- `check_equivalence` runs a small model over mixed edits, appends and value-identical edits. Every system is asserted against `ordinary` at dtype tolerances. It also runs with B=2 to exercise the non-fast executor path.
- At the benchmarked config, every system's final output is compared with `ordinary`. A mismatch makes the process exit non-zero.
- `check_no_implicit_sync` uses CUDA sync debug mode to prove `incgraph[none]` issues no host sync.

## 7.5 Metrics
Wall time (p10/p50/p90), host enqueue time, GPU span from CUDA events, kernel count and kernel time and memcpy count from the profiler, syncs, nodes executed, peak memory, and cache size.

Your v2.4 header records the T4 fp16 results: `region+packed` runs 96 kernels/step (was 156), with kernel time 21.4 ms against 22.7 for `ordinary`. Edit-first is 1.03× of `ordinary`; edit-mid is 14.65 ms against 15.43 for `suffixkv`; edit-last is 6.72 against 6.95; append is 6.93 against 7.12. You also note that edit-last and append are host-bound (about 6.7 ms of host time against about 3.4 ms of kernels).

---

# 8. Every execution case

| Case | What happens | Calls (per layer, per op) |
|---|---|---|
| **Initial build** | Every `append_chunk` touches `x0` with one bit. A single `evaluate()` then runs the accumulated mask `[0,N)`. | One run per op |
| **Edit first (p=0)** | Whole-sequence dirty. | One call per op; the worst case |
| **Edit mid (p)** | `qkv` runs `{p}` in layer 0 and `[p,N)` afterwards. `blk` runs `[p,N)`. | One call per op |
| **Edit last** | Single chunk through all layers. | One call per op, one chunk each |
| **Append** | `n` increases first, then `touch(x0, bit n-1)`. The prefix footprint is bounded by the new `n`. This is KV-cache behaviour emerging from the dependency structure. | One call per op |
| **Several edits before one evaluate** | Masks union. Example: chunks 1 and 4 give two `qkv0` runs, but `blk0` is a single run `[1,N)` because prefix starts at the lowest set bit. | Batching comes for free |
| **No-op edit, `gate=False`** | Treated as a real edit and recomputed. | Same as normal edit |
| **No-op edit, `gate=True`** | `same_input` is true, so nothing is dirtied and `evaluate` does no work. Only the edit is gated, not appends. | 0 |
| **Executor raises** | A run is cleared only after its call returns, and its propagation happens at the same point. The failing run and all later work stay pending, so a retry converges (idempotent executors). In the old graph path, `_commit_changed` handles the equivalent case. | – |
| **B > 1** | The packed executor uses its non-fast path. | Same plan, slower kernels |
| **Window attention** | The planner has `window(w)`, but the **executor must also restrict its key range**. That is not implemented or tested. | – |

---

# 9. Implementation steps to reproduce the design

1. Preallocate slabs indexed by chunk, for `x[0..L]` and q/k/v (or the packed QKV).
2. Write the executor with `write_input`, `same_input`, `qkv_run`, `blk_run` and `output`. Make both run functions idempotent and slice-based.
3. For each layer, register `qkv` then `blk`, in topological order, with `point` and `prefix` footprints.
4. On `append` or `edit`: write the input chunk, then `touch(("x",0), 1<<i, n)`.
5. On `evaluate`: walk the ops in order, run each op's contiguous runs, clear the finished bits, and touch the readers of the written buffer.
6. Verify against a full recompute, using the equivalence check and the final-output comparison at the benchmark config.

---

# 10. Limits stated in your code, and small observations

**Stated limits**
- There is no value-based early cutoff in the slab and region paths, because in-place slabs overwrite old values. Only input gating exists.
- Window footprints need an executor that restricts its key range. That part is untested.
- Executors must be idempotent.

**Observations from reading the code**
- In `RegionPlanner.stats`, the keys `skipped`, `cutoff`, `syncs`, `checks` and others stay at 0. They exist only for API compatibility with the graph.
- `edit` does not validate that `i < n`. An out-of-range index would write outside the live region.
- In `blk`, the `("x",l)` point read is redundant with the prefix footprint (a changed `x[l][i]` always reruns `qkv(l,i)`, which triggers the prefix). It is harmless and keeps the dependency declaration honest.


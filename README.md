# FootprintInc

# FootprintInc — exact incremental recompute for a causal transformer

Notebook: `FootprintTransformer_Proto.ipynb` · Hardware used: Tesla T4, torch 2.11.0+cu130, fp16, B=1, T=2048, C=128 (N=16 chunks), D=768, H=12, L=12.

**Scope of this document.** It documents only what is in the uploaded files and in the logs you pasted: `incgraph2.py`, `slabinc.py`, `regioninc.py`, `bench2.py`, `edit_trace_bench.py` and the notebook. `regioninc.py`, `slabinc.py`, the executors and baselines in `bench2.py`, and `edit_trace_bench.py` were read in full or in the relevant sections. `incgraph2.py` was read only through its docstring and method list; its section below is therefore a summary, not a line-by-line analysis. Nothing here was re-run; every number comes from your logs.

---

## 1. The question and the answer

> When a transformer input sequence is edited or appended to, how much can we recompute instead of re-running the whole sequence?

For a **causal** transformer, an edit to chunk `p` of `N` changes the outputs of chunks `p … N-1` and nothing before `p`. Append changes only the new chunk (the KV-cache case). So the minimum exact recompute is the suffix `[p, N)`.

Measured on 200 real Python commit edits (section 9): the mean fraction of chunks that must be re-executed is **0.619**, so the chunk-count ceiling is 1/0.619 = 1.61×. The best system measured **1.43×** wall-clock overall.

This project does not change the model. It is an **exact incremental inference runtime** for an unchanged pre-LN transformer. Reusing an unchanged prefix's K/V is established practice in LLM serving (the `slabinc.py` and `incgraph2.py` docstrings cite SGLang/RadixAttention and PagedAttention for this); the part specific to this code is that the dirty set is *derived* from declared read footprints instead of being hard-coded as "suffix".

---

## 2. The model being computed (`bench2.py`)

A pre-LN transformer with these exact ops per layer, applied to a chunk `x` of shape `[B, C, D]`:

| Step | Code | Math |
|---|---|---|
| LayerNorm (no affine) | `F.layer_norm(x, (D,))` | `h = (x − μ)/σ` over the last dim |
| QKV projection | `F.linear(h, wqkv)` with `wqkv: [3D, D]`, reshaped to `[B, C, 3, H, Dh]` | `[q│k│v] = h Wqkvᵀ`, `Dh = D/H` |
| Attention | `sdpa(q, K, V, is_causal=True)` | `o = softmax(q Kᵀ/√Dh + causal mask) V` per head |
| Output proj + residual | `x + F.linear(o, wo)` | `a = x + o Woᵀ` |
| MLP + residual | `h + F.linear(F.gelu(F.linear(LN(h), w1)), w2)` | `x' = a + W2 · gelu(W1 · LN(a))`, `w1: [4D, D]`, `w2: [D, 4D]` |

Notes taken from the code: there are **no biases, no LayerNorm weights, no positional encoding, no embedding or LM head** in the benchmarked model. Weights come from `make_params`: `randn(out, in) · in^(−1/2)`, seed 0. Parameters per layer are `12·D²` (`3D² + D² + 4D² + 4D²`), about 7.1 M for D=768; 12 layers give about 85 M.

`full_forward` runs this for all layers over the whole sequence and is the reference (`ordinary`).

---

## 3. Dependency structure and why the suffix is exact

Define chunk `i` at layer `l` as `x[l][i]`. Per layer there are two units of work per chunk:

```
qkv(l, i):  reads x[l][i]                                  writes q,k,v[l][i]
blk(l, i):  reads q[l][i], k,v[l][0..i], x[l][i]           writes x[l+1][i]
            (causal attention + out-proj + residual + MLP)
```

- `qkv` is **position-wise**: LayerNorm and the linear map act on each token independently, so chunk `i` depends only on chunk `i`. This is a *point* footprint.
- `blk(l, i)` attends to keys/values of every chunk `j ≤ i` because of the causal mask. This is a *prefix* footprint.

**Claim.** `x[L][i]` depends on input chunk `j` if and only if `j ≤ i`.

*Argument by induction on the layers.* At layer 0, `x[0][i]` depends on input chunk `i` only. If `x[l][i]` depends exactly on input chunks `≤ i`, then `blk(l, i)` reads `k,v[l][j]` for `j ≤ i`, each of which depends on input chunks `≤ j ≤ i`. So `x[l+1][i]` depends on input chunks `≤ i`, and it does depend on chunk `i` itself through the residual path. ∎

Consequently an edit of chunk `p` can change `x[L][i]` only for `i ≥ p`, and for a generic edit it does change all of them. The planner computes the same set layer by layer (section 6).

---

## 4. Cost model (derived from the op list above, not measured)

Per token per layer, in multiply-add FLOPs counted as 2 per MAC:

- QKV projection: `2·D·3D = 6D²`; output projection: `2D²`; MLP: `2·D·4D + 2·4D·D = 16D²`. Linear total: **`24D²`**.
- Attention for a query at position `t` (all heads): scores `2tD` plus weighted sum `2tD` = **`4tD`**.

For a full sequence of `T = N·C` tokens, per layer: linear `24D²T`, attention `Σ_t 4tD ≈ 2DT²`. Their ratio is `T/(12D)`, which is 2048/9216 ≈ 0.22 at your config, so the linear part dominates.

For an edit at chunk `p`, let `x = p/N`. Recomputing the suffix costs:

- linear: fraction `(1 − x)` of the full cost;
- attention: fraction `(1 − x²)` of the full cost (queries `pC … T`, each attending to all earlier keys).

The combined FLOP fraction is `[(1 − x) + ρ(1 − x²)] / (1 + ρ)` with `ρ = T/(12D)`. For `x = 5/16` this evaluates to about 0.73, a little higher than the chunk-count fraction `(1 − x) = 0.69`, because attention shrinks more slowly than the linear part. This is a derived estimate; the logs report chunk counts and times, not FLOPs.

---

## 5. Evolution of the code (what each file is, and the bottleneck that created the next)

| Stage | File | Idea | Measured bottleneck that led to the next stage |
|---|---|---|---|
| 1 | `incgraph2.py` (+ tensor-valued systems in `bench2.py`) | General dependency graph, versioned skipping, early cutoff, level batching | Every node output is a fresh tensor, so batched calls need cat/split/clone: about 11 copy-units per layer, about 6 ms/step on the T4 (from the `bench2.py` header) |
| 2 | `slabinc.py` | Graph kept only as an invalidation planner; data lives in preallocated slabs written in place | Each `blk(l,i)` has `2+i` graph edges, so O(L·N²) edges, walked three times per evaluate |
| 3 | `regioninc.py` | Replace the graph with bitmask interval algebra | Planner cost becomes O(#ops + #runs), independent of N |
| – | `TorchPackedExecutor` in `bench2.py` | Packed QKV slab, fused residual epilogues | Kernel count per step drops from 156 to 96 (edit-first) |

Planner-only timings quoted in the `regioninc.py` docstring (no-op executor, from `prof.py`, which is not among the uploaded files): N=16: 0.88 / 0.44 / 0.10 ms (first / mid / last edit); N=64: 8.7 / 3.9 / 0.12 ms; N=128: 43 / 21 / 0.20 ms.

---

## 6. `regioninc.py` — the planner (the core algorithm)

### 6.1 Bitmask representation

Each op has a Python big-int `todo`. Bit `i` set means "chunk `i` of this op must be re-executed".

**`mask_runs(m)`** converts a mask into contiguous `[lo, hi)` runs:

1. `lo = (m & -m).bit_length() - 1`. `m & -m` isolates the lowest set bit; its bit length minus one is its index.
2. `t = m >> lo`. The number of trailing ones is `k = ((t + 1) & ~t).bit_length() - 1`. Adding 1 flips the trailing ones to zero and the first zero to one, and `& ~t` isolates that bit.
3. Emit `(lo, lo + k)` and clear those bits with `m ^= ((1 << k) - 1) << lo`.

Example: `m = 0b11011000` gives `(3, 5)` and `(6, 8)`.

### 6.2 Read footprints

A footprint is a function `(changed_mask, n) → consumer_mask`.

| Footprint | Consumer chunk `i` reads | Rule in code |
|---|---|---|
| `point` | `{i}` | identity |
| `prefix` | `[0, i]` | all chunks from the lowest changed bit up to `n−1`: `((1<<n)−1) & ~((1<<lo)−1)` |
| `window(w)` | `[i−w+1, i]` | changed set smeared right by `w−1`, masked to `n` bits |

`window(w)` uses doubling. With `acc = m`, `span = 1`, it repeats `step = min(span, w − span)`, `acc |= acc << step`, `span += step` until `span = w`. Invariant: `acc = ⋃_{k<span} (m << k)`. Shifting by `step ≤ span` produces `⋃_{step ≤ k < span+step}`, which joins the existing union without a gap, so the new span is `span + step`. This takes O(log w) shifts. For `w=3` and a change at bit 2 the result is `{2,3,4}`.

`window` is implemented in the planner but **not used or tested** by any executor in these files; the docstring states that an executor would also have to restrict its key range.

### 6.3 `RegionPlanner`

- `op(run, reads, writes)` registers an op. Ops must be added in topological order. It raises if a buffer has two writers or if an op reads its own output. Each read registers `(op index, footprint)` in `readers[buffer]`.
- `touch(buf, mask, n)`: for every reader `(idx, fp)` of `buf`, `todo[idx] |= fp(mask, n)`.
- `evaluate(n)`: iterate ops in registration order. For each op, split `todo[idx]` into runs. For each run call `run(lo, hi)`, and only after it returns clear those bits from `todo[idx]`, update stats, and `touch` the op's output buffer with the finished bits.

Because propagation happens in the same place where the bits are cleared, an executor that raises leaves the failing run and all downstream work pending. Retrying converges, provided the executor is idempotent.

Cost per evaluate is O(#ops + #runs) big-int operations. There is one op per `(layer, kind)`, so `2L` ops regardless of `N`.

### 6.4 `RegionSlabPipeline` — wiring for the transformer

For each layer `l`, in this order:

```
op( qkv_run(l), reads=[(("x", l), point)],                             writes=("qkv", l)   )
op( blk_run(l), reads=[(("qkv", l), prefix), (("x", l), point)],       writes=("x", l+1)   )
```

- `append_chunk(data)`: `write_input(i, data)`, then `n += 1`, then `touch(("x",0), 1<<i, n)`. `n` is incremented first so the prefix mask is bounded by the new length.
- `edit(i, data)`: if `gate` is on and `same_input` says the data is bitwise identical, drop it (`gated += 1`). Otherwise `write_input` and `touch(("x",0), 1<<i, n)`. There is no check that `i < n`.
- `evaluate()`: `pl.evaluate(n)`, then return `ex.output(n)`.
- `self.g = self.pl` so that `pl.g.stats` and `pl.g.reset_stats()` keep working with code written for the graph version.

### 6.5 Worked trace

L=2, N=6, edit chunk 3. Op order: 0=`qkv0`, 1=`blk0`, 2=`qkv1`, 3=`blk1`.

| Step | Action | `todo` afterwards |
|---|---|---|
| edit | `touch(("x",0), {3})` (point to ops 0 and 1) | `todo[0]={3}`, `todo[1]={3}` |
| op 0 | `qkv_run(0,3,4)`; touch `("qkv",0)` with `prefix` | `todo[1]={3,4,5}` |
| op 1 | `blk_run(0,3,6)` (one call); touch `("x",1)` | `todo[2]={3,4,5}`, `todo[3]={3,4,5}` |
| op 2 | `qkv_run(1,3,6)`; touch `("qkv",1)` with `prefix` | `todo[3]={3,4,5}` |
| op 3 | `blk_run(1,3,6)` | all empty |

Four executor calls (`2L`). Chunks 0–2 are never touched.

### 6.6 Executed-chunk count, checked against your log

Layer 0's `qkv` runs only the edited chunk (point footprint); every later op runs the whole suffix. So the number of chunk-executions is

```
1 + (2L − 1)(N − p)
```

With L=12, N=16 this gives **369** for `p=0`, **24** for `p=15` (edit-last), and **185** for `p=8`. These are exactly the `nodes` values in the `incgraph[region]` and `incgraph[region+packed]` rows of the benchmark log (369, 24, 185), which suggests edit-mid is `p=8`; the benchmark code that fixes `p` was not among the uploaded files.

### 6.7 Several edits before one evaluate

Masks are unions. Edits at chunks 1 and 4 give two `qkv0` runs `[1,2)` and `[4,5)`, but `blk0` gets one run `[1, N)` because `prefix` starts from the lowest set bit. This follows from the code in 6.2–6.3; it is not covered by a benchmark scenario.

---

## 7. `slabinc.py` — the graph-based planner (kept as ablation)

`SlabPipeline` builds a `Graph(cutoff="none")`. Node values are chunk-index tokens; the tensors are side effects of executing nodes. `blk(l, i)` depends on `[x[l][i], qkv(l,i)] + qkv(l, 0..i−1)`, which is the O(i) fan-in that makes the edge count O(L·N²). Batch functions `_bq` / `_bb` sort the dirty chunk indices, coalesce them into runs with `runs(idx)`, and call `qkv_run` / `blk_run` once per run.

**Executor contract** (shared with `regioninc.py`): executors must be idempotent; old values are overwritten in place, so value-based early cutoff is impossible (`cutoff="none"`). The substitute is the input gate: an edit bitwise equal to the stored chunk is dropped before it dirties anything.

Executor protocol:

```
write_input(i, data)      same_input(i, data) -> bool
qkv_run(l, lo, hi)        reads x[l][lo:hi];                       writes q/k/v[l][lo:hi]
blk_run(l, lo, hi)        reads q[l][lo:hi], k/v[l][:hi], x[l][lo:hi];  writes x[l+1][lo:hi]
output(n)                 x[L][:n]
```

---

## 8. `incgraph2.py` — the general engine (summary)

From its docstring and method list. `Graph(cutoff=...)` holds `Node`s with `fn`, `deps`, `users`, levels and versions.

- `input`, `op(..., batch=(key, batch_fn), group=...)`, `set_input`, `invalidate_all`, `rewire`, `remove`, `evaluate(targets)`, `tick`, `policy_snapshot`, `cache_bytes`.
- **Versioned skipping:** a node is skipped when the versions of its dependencies match what it last saw.
- **Early cutoff modes:** `none`, `exact`, `tol`, `adaptive`. The adaptive mode stops checking where checks never hit and re-probes occasionally; a skipped check only forgoes a skip, so it is always safe.
- **Level-coalesced batching:** dirty nodes of one level that share a batch key run through one `batch_fn` call.
- **Exception safety (v2.1):** a failing node, `eq` or `resolve` no longer leaves other nodes marked clean with stale values.
- **Contract:** user functions are pure and must not mutate inputs or outputs.

Its docstring states the prior art (Adapton; Build Systems à la Carte; Neubig et al. on batching) and that the edit-last and append speedups are the known KV-reuse effect. In the final slab/region path only `slabinc.py` imports it; `regioninc.py` does not.

---

## 9. `bench2.py` — executors, baselines, checks

### 9.1 The attention detail that makes suffix recompute correct

A run of dirty chunks `[lo, hi)` is the **last rows** of the key sequence `K[:hi·C]`. With `L = (hi−lo)·C` query rows and `S = hi·C` keys, query row `r` sits at global position `lo·C + r = S − L + r` and may attend to keys `≤ S − L + r`. That is exactly a **lower-right-aligned causal mask**.

`sdpa_tail_causal(q, K, V)`:
- if `L == S`: ordinary `is_causal=True`;
- else if `_lr_ok(device, dtype)`: `attn_mask=causal_lower_right(L, S)`;
- else: an explicit boolean mask. `_lr_ok` self-tests the lower-right mask against the explicit mask once per `(device, dtype)`.

The `bench2.py` header (v2.2) records the reason: a plain explicit `m×N` mask did more attention work than full causal for small `p`.

### 9.2 Executors

**`TorchSlabExecutor`**: slabs `X[0..L]` `[B,T,D]` and `Q/K/V[l]` `[B,H,T,Dh]`, with `T = tmax·C`. `qkv_run` computes `qkv_proj` on `X[l][:, s]` and writes the three slices. `blk_run` calls `sdpa_tail_causal(Q[s], K[:e], V[:e])` with `e = hi·C`, then `mlp_block(attn_out(...))` into `X[l+1][:, s]`. Keys are limited to `[:e]` because chunks `≥ hi` are never attended to by these queries.

**`TorchPackedExecutor`** (same protocol, fewer kernels):
- one packed slab `QKV[l] = [B, T, 3, H, Dh]`; the QKV GEMM writes straight into it with `torch.mm(..., out=)`; q/k/v are strided views, so there is no Q slab and no per-layer copy kernels;
- weights are pre-transposed views (`WT`);
- residuals are fused into GEMM epilogues with `torch.addmm`, and the block output is written in place into `X[l+1]`;
- `B == 1` takes this fast path; `B > 1` falls back to `copy_` / `add(out=)`, which is correct but slower. The docstring notes numerics differ from the unfused path by one fewer fp16 rounding per residual.

### 9.3 Systems compared

| Name | What it is |
|---|---|
| `ordinary` | Full causal forward over the whole sequence every step (`full_forward`) |
| `handkv` | Hand-written prefix reuse, chunk by chunk (documented in the header as a weak baseline) |
| `suffixkv` | Strongest hand-written baseline: per-layer K/V arena plus one batched forward over the dirty suffix `[p, N)` as a single `(N−p)·C`-row sequence |
| `chunked-full`, `incgraph[none/exact/adaptive(+batch/+fuse)]` | Tensor-valued graph variants |
| `incgraph[slab]` | `SlabPipeline` + `TorchSlabExecutor` |
| `incgraph[region]` | `RegionSlabPipeline` + `TorchSlabExecutor` |
| `incgraph[region+packed]` | `RegionSlabPipeline` + `TorchPackedExecutor` (the final system) |
| `…[memo]` variants | Skip bitwise-identical edits at the input, so no-op scenarios are compared fairly |

### 9.4 Correctness checks

- `check_equivalence`: a small model (default L=2, D=32, H=4, C=8, N=6, 12 steps) with a mix of edits at random positions, an append every third step, and value-identical edits every fourth edit step; every system is asserted `allclose` to `ordinary` at the tolerances below. The header says it also runs with B=2 to exercise the packed executor's non-fast path.
- At the benchmarked config, every system's final output is compared with `ordinary`; a mismatch exits non-zero (v2.1 header).
- Tolerances (`TOL`): fp32 1e-3, fp16 5e-2, bf16 1.5e-1.

### 9.5 Benchmark methodology

Scenarios: `edit-first`, `edit-mid`, `edit-last`, `noop-mid`, `append`; optional `--sweep` over every edit position. `--interleave` round-robins systems per iteration to cancel T4 clock drift (the v2.2 header reports about 10% drift on identical kernels). Reported per row: wall p50 [p10–p90], host enqueue time, kernels per step, kernel time, memcpy count, syncs, nodes executed, peak and cache memory.

---

## 10. Results on the T4 (fp16), from your logs

### 10.1 `bench2.py --interleave --iters 40`, wall p50 in ms

| Scenario | `ordinary` | `suffixkv` | `incgraph[region]` | `incgraph[region+packed]` |
|---|---|---|---|---|
| edit-first | 25.48 | 26.73 (0.95×) | 26.97 (0.94×) | **24.83 (1.03×)** |
| edit-mid | 24.61 | 16.21 (1.52×) | 16.46 (1.49×) | **15.46 (1.59×)** |
| edit-last | 22.43 | 6.76 (3.32×) | 8.08 (2.78×) | **6.53 (3.44×)** |
| noop-mid (no gating) | 24.83 | 16.60 (1.50×) | 16.82 (1.48×) | **15.66 (1.59×)** |
| append | 17.10 | 6.74 (2.54×) | 7.95 (2.15×) | **6.51 (2.63×)** |

Kernels per step, edit-first: `ordinary` 120, `suffixkv` 145, `region` 156, `region+packed` 96. Kernel time, edit-first: 23.68 / 24.31 / 24.72 / 22.28 ms. Cache: `suffixkv` 72 MiB, `region` and `region+packed` 147 MiB (13 `X` slabs plus the q/k/v storage at T=2048).

Host time versus kernel time for `region+packed`: edit-last 6.45 ms host against 3.43 ms kernels; append 6.48 against 3.39. In these two scenarios wall time is about equal to host time, so they are **host-bound**.

### 10.2 `edit_trace_bench.py` on 200 real commits

| | mean ms | p50 | speedup | first / mid / last third |
|---|---|---|---|---|
| `ordinary` | 25.23 | 25.33 | 1.00× | 1.00 / 1.00 / 1.00 |
| `suffixkv` | 18.71 | 20.24 | 1.35× | 1.03 / 1.36 / 2.71 |
| `incgraph[region+packed]` | 17.63 | 19.28 | **1.43×** | 1.10 / 1.44 / 2.82 |

Median edited chunk 5/16; mean recompute fraction 0.619. Correctness: all systems matched `ordinary` (checked every 25 steps and at the end).

Run history for this script: the first run fell back to a synthetic end-biased trace (Beta(3,1); fraction 0.282; `region+packed` 2.35×, `suffixkv` 2.24×); an earlier second run on the real trace gave 1.43× / 1.35× with `ordinary` 23.93 ms, and the run above reproduced the same ratios with `ordinary` 25.23 ms. The current script has no synthetic fallback.

---

## 11. `edit_trace_bench.py` — how the real trace is built

1. Download the CommitPackFT Python JSONL directly from the Hugging Face URL in the script (the dataset loader script is bypassed because current `datasets` versions reject script-based datasets).
2. For each commit with non-empty, different `old_contents` and `new_contents`: tokenize both (first 30 000 characters, GPT-2 tokenizer, no special tokens). Skip if the old file has fewer than 256 tokens.
3. Find the first differing token `d` over the common length; for a pure insertion or append, `d = common length`; skip if lengths and tokens are identical.
4. Relative position `r = d / len(old_ids)`; chunk `p = min(N−1, int(r·N))`. Replacement content is `new_ids[d : d+1024]`, truncated or zero-padded to `C = 128` tokens.
5. The base context is the first `T = 2048` tokens of the concatenated old files (zero-padded if short). Tokens become embeddings through a **random** table (`randn · 0.02`), so chunks are float tensors `[1, C, D]`.
6. Each commit becomes one `("edit", p, chunk)` step, applied in order to the same 16-chunk state. Each step is run on every system in rotating order, with `cuda.synchronize()` around the timer. The first 3 steps are dropped from the statistics.
7. Every 25 steps and at the end: maximum absolute difference to `ordinary`, divided by `ordinary`'s maximum absolute value, must be below `TOL[dtype]`.
8. Print the recompute fraction `mean((N−p)/N)`, the table above, and write `edit_trace_results.json`.

---

## 12. Reproduction steps

1. Put `incgraph2.py`, `slabinc.py`, `regioninc.py`, `bench2.py`, `edit_trace_bench.py` in `/content` under exactly these names (your uploads carry a `_1` suffix; the notebook calls the plain names). The uploaded `_1` copies are byte-identical to the earlier versions.
2. Run `!cd /content && python bench2.py --interleave --iters 40 --dtype float16 --systems "ordinary,suffixkv,incgraph[region],incgraph[region+packed]"`.
3. Run `!pip -q install datasets transformers` then `!python edit_trace_bench.py`.

---

## 13. Limits and corrections

**Stated by the code**
- No value-based early cutoff in the slab and region paths (in-place overwrite); only input gating.
- `window(w)` is untested, and the executor would have to restrict its key range.
- Executors must be idempotent.
- `edit` does not validate `i < n`.
- The `RegionPlanner.stats` keys `skipped`, `cutoff`, `syncs`, `checks` and others stay 0; they exist for compatibility with the graph's stats.
- The next lever named in the `bench2.py` v2.4 header for edit-last and append is per-suffix CUDA-graph replay, not implemented.

**Read from the code and results**
- The benchmarked model has no biases, LayerNorm weights, positional encoding, embedding or LM head, and uses random weights. Nothing here shows it reproduces a real model's logits.
- Only chunk **replacement** and **append** are modelled. Inserting or deleting tokens shifts chunk boundaries; the planner has no concept of that. In the trace script, each commit is reduced to a single chunk replacement at its first differing position.
- The trace applies 200 commits from different files to one shared 16-chunk state, so it measures edit *positions*, not a real editing session.
- The "1.61× ceiling" printed by the script counts chunk executions only. Attention cost and the fixed host launch overhead do not scale with the chunk count, so it is not a time bound; the FLOP-weighted fraction in section 4 differs slightly. "1.43× is 89% of 1.61×" is therefore a statement about chunk counts only.
- One T4 session, fp16, B=1, one configuration. The 3–6% lead of `region+packed` over `suffixkv` in the logs is consistent in sign across rows but comes from single runs.

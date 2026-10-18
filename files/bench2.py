#!/usr/bin/env python3
"""
[v2.4] T4 fp16 log: planner fix (regioninc.py) + packed-QKV executor landed as modelled.
  incgraph[region+packed]: 96 kernels/step (was 156), kernel ms 21.4 (ordinary 22.7), edit-first 1.03x of
  ordinary, edit-mid 14.65 vs suffixkv 15.43, edit-last 6.72 vs 6.95, append 6.93 vs 7.12.
  incgraph[slab] / incgraph[region] stay as ablations.  Self-check now also covers B=2 (the packed
  executor's non-fast path).  Remaining known limit: edit-last/append are HOST-bound (host ~6.7 ms vs
  ~3.4 ms of kernels) -- the next lever is per-suffix CUDA-graph replay, not implemented here.
[v2.3] from the v2.2 T4 log: the attention fix landed exactly as modelled (edit-mid attention 34.9 -> 27.5 ms,
  model 27.3), but piece-cloning (OWN_PIECES) added ~500 tiny kernels and made edit-first WORSE (0.95x ->
  0.93x).  Root cause is structural: tensor-valued nodes force cat/split/clone (~11 copy-units/layer,
  ~6 ms/step on a T4).  New system incgraph[slab]: the graph plans invalidation, slab-backed REGION
  EXECUTORS (slabinc.py) run each contiguous dirty run as one in-place call (~4 units/layer).  Cloning is
  now opt-in (--compact) and only matters for the legacy tensor-valued batched variants.
[v2.2] driven by the T4 log (see ANALYSIS in the reply / model in this docstring):
  * tail-causal attention: a run of dirty chunks is exactly the LAST rows of the key sequence, so
    attention is lower-right-aligned causal (torch.nn.attention.bias.causal_lower_right).  Tiles above the
    diagonal are skipped, no mask is materialised.  The v2.1 explicit mask computed the full m*N rectangle:
    for p < N/4 that is MORE attention work than ordinary full-sequence causal.  Self-tested at first use
    per (device, dtype); falls back to an explicit mask if unavailable or wrong.
  * fused block node (--systems incgraph[none+batch+fuse]): attention+MLP are one node, so the attention
    output is never cached (-1/5 of cache) and there are fewer nodes/levels.
  * piece compaction (opt-in since v2.3): split() views of a batched output pin the WHOLE batch tensor.
  * suffixkv: the strongest hand-written baseline (prefix K/V + ONE batched forward over the dirty suffix).
    `handkv` processes chunk by chunk (M=128 GEMMs) and is a weak baseline.
  * --interleave: A/B/C... round-robin per iteration so T4 clock drift cancels (v2.1 timed systems in
    sequence; the log shows ~10% drift of identical kernels between scenarios).
[v2.1] fixes vs v2:
  * batched ATTENTION: dirty chunks of one layer form contiguous runs (causality => an edit at p
    dirties p..N-1), so each run is ONE sdpa call over [run queries] x [one K/V concat], instead of one
    call + one K/V concat per chunk.  Same math (checked against per-chunk attention).
  * every system's final output is verified against `ordinary` AT THE BENCHMARKED CONFIG (v2 verified
    only a tiny model); mismatches are reported and make the process exit non-zero.
  * fair no-op baselines: ordinary[memo]/handkv[memo] skip when the edited chunk is bitwise identical
    (what any production system would do), so a graph "no-op" win is not measured against a baseline
    that recomputes everything on an identical input.
  * --sweep: edit EVERY chunk position and report the crossover vs ordinary.
[v2] adds incgraph[none+batch] / incgraph[adaptive+batch] and --breakdown (top kernels by time).
Benchmark incgraph against ordinary execution on a causal pre-LN transformer.

Systems (all compute the same function; correctness is asserted before timing):
  ordinary         full forward over the whole sequence every step, flash/SDPA is_causal
                   (what a conventional runtime does; best-case single-kernel attention)
  handkv           hand-written prefix reuse: recompute chunks >= edit point, reuse K/V
                   below it (what an expert would write by hand; no graph runtime)
  chunked-full     the SAME chunked graph, but invalidate_all() every step.  Isolates
                   runtime + chunking overhead from the benefit of reuse.
  incgraph[none]   incremental, no early cutoff, never syncs the host
  incgraph[exact]  incremental + level-batched bitwise early cutoff (1 sync per level)

Scenarios:
  edit-first/mid/last   replace one chunk of an existing sequence (causal => invalidates
                        chunk p and everything after it; 'first' is the worst case)
  noop-mid              "edit" with a value-identical tensor (cutoff should make it free)
  append                streaming: add a chunk to the sequence (KV-cache behaviour emerges
                        from the dependency structure, nobody wrote a cache)

Metrics per step: wall latency (host start -> cudaSynchronize), host enqueue time,
GPU span (CUDA events), kernel count + kernel time + memcpy count (torch.profiler),
host syncs issued by the runtime, nodes executed, peak allocated memory, resident cache.

    python bench.py --device cuda --T 2048 --C 128 --D 768 --H 12 --L 12 --iters 50
"""
import argparse
import functools
import gc
import json
import time

import torch
import torch.nn.functional as F

from incgraph2 import Graph
from slabinc import SlabPipeline
from regioninc import RegionSlabPipeline

sdpa = F.scaled_dot_product_attention
torch.set_grad_enabled(False)


# --------------------------------------------------------------------------- model math
def make_params(L, D, device, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)

    def w(o, i):
        return (torch.randn(o, i, generator=g) * i ** -0.5).to(device=device, dtype=dtype)

    return [dict(wqkv=w(3 * D, D), wo=w(D, D), w1=w(4 * D, D), w2=w(D, 4 * D)) for _ in range(L)]


@functools.lru_cache(None)
def chunk_mask(i, C, device):
    """Chunk i (i>0) attends to all of chunks < i and causally to itself."""
    full = torch.ones(C, i * C, dtype=torch.bool)
    tri = torch.tril(torch.ones(C, C, dtype=torch.bool))
    return torch.cat([full, tri], 1).to(device)


def qkv_proj(x, p, H):
    B, C, D = x.shape
    t = F.linear(F.layer_norm(x, (D,)), p["wqkv"]).view(B, C, 3, H, D // H).permute(2, 0, 3, 1, 4)
    return t[0], t[1], t[2]


def attn_out(o, x, p):
    B, H, C, Dh = o.shape
    return x + F.linear(o.transpose(1, 2).reshape(B, C, H * Dh), p["wo"])


def mlp_block(h, p):
    return h + F.linear(F.gelu(F.linear(F.layer_norm(h, (h.shape[-1],)), p["w1"])), p["w2"])


def full_forward(params, H, x):
    for p in params:
        q, k, v = qkv_proj(x, p, H)
        x = mlp_block(attn_out(sdpa(q, k, v, is_causal=True), x, p), p)
    return x


try:
    from torch.nn.attention.bias import causal_lower_right
except Exception:                                    # old torch
    causal_lower_right = None

OWN_PIECES = False                                   # clone cached split() pieces (see header)
_LR = {}


def _own(t):
    return t.clone() if OWN_PIECES else t


def _lr_ok(device, dtype):
    """One-time self-test: lower-right causal bias == explicit mask on this device/dtype."""
    key = (device.type, dtype)
    if key not in _LR:
        ok = False
        if causal_lower_right is not None:
            try:
                g = torch.Generator().manual_seed(0)
                mk = lambda n: torch.randn(1, 2, n, 16, generator=g).to(device, dtype)
                q, k, v = mk(8), mk(24), mk(24)
                rows, cols = 16 + torch.arange(8, device=device), torch.arange(24, device=device)
                want = sdpa(q, k, v, attn_mask=cols[None, :] <= rows[:, None])
                got = sdpa(q, k, v, attn_mask=causal_lower_right(8, 24))
                ok = bool(torch.allclose(got, want, atol=TOL[dtype], rtol=TOL[dtype]))
            except Exception:
                ok = False
        _LR[key] = ok
    return _LR[key]


def sdpa_tail_causal(q, K, V):
    """q holds the LAST L query rows of a key sequence of length S>=L: row r attends keys <= S-L+r."""
    L, S = q.shape[2], K.shape[2]
    if L == S:
        return sdpa(q, K, V, is_causal=True)
    if _lr_ok(q.device, q.dtype):
        return sdpa(q, K, V, attn_mask=causal_lower_right(L, S))
    rows = (S - L) + torch.arange(L, device=q.device)
    cols = torch.arange(S, device=q.device)
    return sdpa(q, K, V, attn_mask=cols[None, :] <= rows[:, None])


def chunk_attention(q, K, V, j=None, C=None):       # j, C kept for API compatibility
    return sdpa_tail_causal(q, K, V)


def batch_qkv(argl, p, H):
    """ONE projection for k dirty chunks (position-wise op => identical math), then split."""
    xs = [a[0] for a in argl]
    C = xs[0].shape[1]
    q, k, v = qkv_proj(torch.cat(xs, 1), p, H)
    out = list(zip(q.split(C, 2), k.split(C, 2), v.split(C, 2)))
    return [tuple(_own(t) for t in o) for o in out] if len(out) > 1 else out


def batch_mlp(argl, p):
    hs = [a[0] for a in argl]
    pieces = list(mlp_block(torch.cat(hs, 1), p).split(hs[0].shape[1], 1))
    return [_own(t) for t in pieces] if len(pieces) > 1 else pieces


@functools.lru_cache(64)
def range_mask(a, cnt, C, device):
    """Queries = chunks a..a+cnt-1 (global rows a*C+r), keys = chunks 0..a+cnt-1; key <= query position."""
    rows = a * C + torch.arange(cnt * C)
    cols = torch.arange((a + cnt) * C)
    return (cols[None, :] <= rows[:, None]).to(device)


def batch_att(argl, p, C, own=True):
    """Attention for k dirty chunks of one layer.  Node i's args are [x_i, qkv_i, qkv_0..qkv_{i-1}], so
    its chunk index is len(args)-2.  K/V of chunks 0..max are concatenated ONCE; each contiguous run of
    dirty chunks is one sdpa call (top-left causal if it starts at chunk 0, else an explicit mask)."""
    items = sorted(((len(a) - 2, a) for a in argl), key=lambda t: t[0])
    amax = items[-1][1]
    kv = list(amax[2:]) + [amax[1]]
    K = torch.cat([t[1] for t in kv], 2) if len(kv) > 1 else kv[0][1]
    V = torch.cat([t[2] for t in kv], 2) if len(kv) > 1 else kv[0][2]
    runs = []
    for i, a in items:
        if runs and runs[-1][-1][0] == i - 1:
            runs[-1].append((i, a))
        else:
            runs.append([(i, a)])
    out = {}
    for run in runs:
        m = run[-1][0]
        x = torch.cat([a[0] for _, a in run], 1) if len(run) > 1 else run[0][1][0]
        q = torch.cat([a[1][0] for _, a in run], 2) if len(run) > 1 else run[0][1][1][0]
        Kl, Vl = K[:, :, :(m + 1) * C], V[:, :, :(m + 1) * C]
        o = sdpa_tail_causal(q, Kl, Vl)              # run = last rows of Kl  => lower-right causal
        pieces = attn_out(o, x, p).split(C, 1)
        for (i, _), piece in zip(run, pieces):
            out[i] = _own(piece) if (own and len(run) > 1) else piece
    return [out[len(a) - 2] for a in argl]


# --------------------------------------------------------------------------- systems
class Ordinary:
    def __init__(self, params, H, chunks, tmax, memo=False):
        self.name = "ordinary[memo]" if memo else "ordinary"
        self.memo = memo
        self.p, self.H = params, H
        B, C, D = chunks[0].shape
        self.C = C
        self.buf = torch.zeros(B, tmax * C, D, device=chunks[0].device, dtype=chunks[0].dtype)
        for i, c in enumerate(chunks):
            self.buf[:, i * C:(i + 1) * C] = c
        self.n, self.out = len(chunks), None
        memo, self.memo = self.memo, False        # first forward must really run
        self.step(("edit", 0, chunks[0]))
        self.memo = memo

    def step(self, arg):
        C = self.C
        if arg[0] == "edit":
            cur = self.buf[:, arg[1] * C:(arg[1] + 1) * C]
            if self.memo and torch.equal(cur, arg[2]):
                return                       # identical input: previous output is still exact
            cur.copy_(arg[2])
        else:
            self.buf[:, self.n * C:(self.n + 1) * C].copy_(arg[1])
            self.n += 1
        self.out = full_forward(self.p, self.H, self.buf[:, :self.n * C])

    def output(self):
        return self.out

    def stats(self):
        return {}

    def reset_stats(self):
        pass

    def cache_bytes(self):
        return 0


class HandKV:
    """Hand-written prefix reuse: K/V buffers per layer; edit at i recomputes chunks >= i."""
    def __init__(self, params, H, chunks, tmax, memo=False):
        self.name = "handkv[memo]" if memo else "handkv"
        self.memo = memo
        self.p, self.H = params, H
        B, C, D = chunks[0].shape
        self.C = C
        mk = lambda: torch.zeros(B, H, tmax * C, D // H, device=chunks[0].device, dtype=chunks[0].dtype)
        self.K = [mk() for _ in params]
        self.V = [mk() for _ in params]
        self.raw, self.outs = list(chunks), [None] * len(chunks)
        self._run_from(0)

    def _run_from(self, i):
        C = self.C
        for j in range(i, len(self.raw)):
            x = self.raw[j]
            for l, p in enumerate(self.p):
                q, k, v = qkv_proj(x, p, self.H)
                self.K[l][:, :, j * C:(j + 1) * C] = k
                self.V[l][:, :, j * C:(j + 1) * C] = v
                o = chunk_attention(q, self.K[l][:, :, :(j + 1) * C], self.V[l][:, :, :(j + 1) * C], j, C)
                x = mlp_block(attn_out(o, x, p), p)
            self.outs[j] = x

    def step(self, arg):
        if arg[0] == "edit":
            if self.memo and torch.equal(self.raw[arg[1]], arg[2]):
                return
            self.raw[arg[1]] = arg[2]
            self._run_from(arg[1])
        else:
            self.raw.append(arg[1])
            self.outs.append(None)
            self._run_from(len(self.raw) - 1)

    def output(self):
        return torch.cat(self.outs, 1)

    def stats(self):
        return {}

    def reset_stats(self):
        pass

    def cache_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.K + self.V)


class TorchSlabExecutor:
    """Region executor over preallocated slabs: x[0..L], q/k/v[0..L-1], each [.., tmax*C, ..]."""

    def __init__(self, params, H, chunk0, tmax):
        self.p, self.H, self.L = params, H, len(params)
        B, C, D = chunk0.shape
        self.C, T = C, tmax * C
        dv, dt = chunk0.device, chunk0.dtype
        z = lambda *shape: torch.zeros(*shape, device=dv, dtype=dt)
        self.X = [z(B, T, D) for _ in range(self.L + 1)]
        self.Q = [z(B, H, T, D // H) for _ in params]
        self.K = [z(B, H, T, D // H) for _ in params]
        self.V = [z(B, H, T, D // H) for _ in params]

    def write_input(self, i, data):
        self.X[0][:, i * self.C:(i + 1) * self.C] = data

    def same_input(self, i, data):
        return torch.equal(self.X[0][:, i * self.C:(i + 1) * self.C], data)

    def qkv_run(self, l, lo, hi):
        s = slice(lo * self.C, hi * self.C)
        q, k, v = qkv_proj(self.X[l][:, s], self.p[l], self.H)
        self.Q[l][:, :, s] = q
        self.K[l][:, :, s] = k
        self.V[l][:, :, s] = v

    def blk_run(self, l, lo, hi):
        s, e = slice(lo * self.C, hi * self.C), hi * self.C
        o = sdpa_tail_causal(self.Q[l][:, :, s], self.K[l][:, :, :e], self.V[l][:, :, :e])
        self.X[l + 1][:, s] = mlp_block(attn_out(o, self.X[l][:, s], self.p[l]), self.p[l])

    def output(self, n):
        return self.X[self.L][:, :n * self.C]

    def cache_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.X + self.Q + self.K + self.V)


class SlabInc:
    """incgraph as invalidation planner + slab region executors (see slabinc.py).  cutoff is 'none' by
    construction; memo=True adds exact gating of bitwise-identical edits at the input."""

    def __init__(self, params, H, chunks, tmax, memo=False):
        self.name = "incgraph[slab+memo]" if memo else "incgraph[slab]"
        self.ex = TorchSlabExecutor(params, H, chunks[0], tmax)
        self.pl = SlabPipeline(self.ex, len(params), gate=memo)
        for c in chunks:
            self.pl.append_chunk(c)
        self.pl.evaluate()
        self.pl.g.reset_stats()

    def step(self, arg):
        if arg[0] == "edit":
            self.pl.edit(arg[1], arg[2])
        else:
            self.pl.append_chunk(arg[1])
        self.pl.evaluate()

    def output(self):
        return self.ex.output(self.pl.n)

    def stats(self):
        return dict(self.pl.g.stats)

    def reset_stats(self):
        self.pl.g.reset_stats()

    def cache_bytes(self):
        return self.ex.cache_bytes()


class RegionInc(SlabInc):
    """incgraph planner replaced by the interval-footprint region planner (regioninc.py)."""

    def __init__(self, params, H, chunks, tmax, memo=False):
        self.name = "incgraph[region+memo]" if memo else "incgraph[region]"
        self.ex = TorchSlabExecutor(params, H, chunks[0], tmax)
        self.pl = RegionSlabPipeline(self.ex, len(params), gate=memo)
        for c in chunks:
            self.pl.append_chunk(c)
        self.pl.evaluate()
        self.pl.g.reset_stats()


class TorchPackedExecutor:
    """Slab executor with the dispatch/copy overhead removed (same protocol as TorchSlabExecutor).

    * ONE packed slab QKV[l] = [B, T, 3, H, Dh]: the qkv GEMM writes straight into it (mm out=), and
      q/k/v are STRIDED VIEWS of it -> no per-layer q/k/v copy kernels, no Q slab.
    * residual adds are fused into the GEMM epilogue (addmm: h + a @ W^T) and the block output is
      written in place into X[l+1] -> no separate add / memcpy kernels.
    * B == 1 fast path; B > 1 falls back to copy_/add(out=) (correct, slower).
    Numerics differ from the unfused path only by one fewer fp16 rounding per residual."""

    def __init__(self, params, H, chunk0, tmax):
        self.p, self.H, self.L = params, H, len(params)
        B, C, D = chunk0.shape
        self.B, self.C, self.D, self.Dh = B, C, D, D // H
        T, dv, dt = tmax * C, chunk0.device, chunk0.dtype
        z = lambda *shape: torch.zeros(*shape, device=dv, dtype=dt)
        self.X = [z(B, T, D) for _ in range(self.L + 1)]
        self.QKV = [z(B, T, 3, H, self.Dh) for _ in params]
        self.WT = [{k: v.t() for k, v in p.items()} for p in params]      # transposed VIEWS, no copy
        self.fast = B == 1

    def write_input(self, i, data):
        self.X[0][:, i * self.C:(i + 1) * self.C].copy_(data)

    def same_input(self, i, data):
        return torch.equal(self.X[0][:, i * self.C:(i + 1) * self.C], data)

    def qkv_run(self, l, lo, hi):
        C, D, H = self.C, self.D, self.H
        s, rows = slice(lo * C, hi * C), (hi - lo) * C
        ln = F.layer_norm(self.X[l][:, s], (D,))
        dst = self.QKV[l][:, s]
        if self.fast:
            torch.mm(ln.view(rows, D), self.WT[l]["wqkv"], out=dst.view(rows, 3 * D))
        else:
            dst.copy_(F.linear(ln, self.p[l]["wqkv"]).view(self.B, rows, 3, H, self.Dh))

    def blk_run(self, l, lo, hi):
        C, D, B = self.C, self.D, self.B
        s, e, rows = slice(lo * C, hi * C), hi * C, (hi - lo) * C
        qkv = self.QKV[l]
        o = sdpa_tail_causal(qkv[:, s, 0].transpose(1, 2), qkv[:, :e, 1].transpose(1, 2),
                             qkv[:, :e, 2].transpose(1, 2))
        x, dst, W, Wt = self.X[l][:, s], self.X[l + 1][:, s], self.p[l], self.WT[l]
        if self.fast:
            h = torch.addmm(x.reshape(rows, D), o.transpose(1, 2).reshape(rows, D), Wt["wo"])
            a = F.gelu(torch.mm(F.layer_norm(h, (D,)), Wt["w1"]))
            torch.addmm(h, a, Wt["w2"], out=dst.view(rows, D))
        else:
            h = attn_out(o, x, W)
            torch.add(h, F.linear(F.gelu(F.linear(F.layer_norm(h, (D,)), W["w1"])), W["w2"]), out=dst)

    def output(self, n):
        return self.X[self.L][:, :n * self.C]

    def cache_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.X + self.QKV)


class RegionPacked(RegionInc):
    def __init__(self, params, H, chunks, tmax, memo=False):
        self.name = "incgraph[region+packed+memo]" if memo else "incgraph[region+packed]"
        self.ex = TorchPackedExecutor(params, H, chunks[0], tmax)
        self.pl = RegionSlabPipeline(self.ex, len(params), gate=memo)
        for c in chunks:
            self.pl.append_chunk(c)
        self.pl.evaluate()
        self.pl.g.reset_stats()


class SuffixKV:
    """Strongest hand-written baseline: per-layer prefix K/V arena + ONE batched forward over the dirty
    suffix (all chunks >= edit point as a single M=(N-p)*C sequence).  This is what an expert writes."""

    def __init__(self, params, H, chunks, tmax, memo=False):
        self.name = "suffixkv[memo]" if memo else "suffixkv"
        self.memo, self.p, self.H = memo, params, H
        B, C, D = chunks[0].shape
        self.C = C
        dv, dt = chunks[0].device, chunks[0].dtype
        mk = lambda: torch.zeros(B, H, tmax * C, D // H, device=dv, dtype=dt)
        self.K = [mk() for _ in params]
        self.V = [mk() for _ in params]
        self.outbuf = torch.zeros(B, tmax * C, D, device=dv, dtype=dt)
        self.raw = list(chunks)
        self._run_from(0)

    def _run_from(self, i):
        C, n = self.C, len(self.raw)
        lo, hi = i * C, n * C
        x = torch.cat(self.raw[i:], 1) if n - i > 1 else self.raw[i]
        for l, p in enumerate(self.p):
            q, k, v = qkv_proj(x, p, self.H)
            self.K[l][:, :, lo:hi] = k
            self.V[l][:, :, lo:hi] = v
            o = sdpa_tail_causal(q, self.K[l][:, :, :hi], self.V[l][:, :, :hi])
            x = mlp_block(attn_out(o, x, p), p)
        self.outbuf[:, lo:hi] = x

    def step(self, arg):
        if arg[0] == "edit":
            if self.memo and torch.equal(self.raw[arg[1]], arg[2]):
                return
            self.raw[arg[1]] = arg[2]
            self._run_from(arg[1])
        else:
            self.raw.append(arg[1])
            self._run_from(len(self.raw) - 1)

    def output(self):
        return self.outbuf[:, :len(self.raw) * self.C]

    def stats(self):
        return {}

    def reset_stats(self):
        pass

    def cache_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.K + self.V)


class IncSystem:
    """The chunked transformer expressed as a dependency graph (v2: optional batching/adaptive cutoff).

    per layer l, chunk i:  qkv[l][i](x[l][i]) -> att[l][i](x[l][i], qkv[l][i], qkv[l][:i]) -> x[l+1][i]
    Causality lives in the edges: attn of chunk i reads K/V of chunks <= i only, so an edit of
    chunk p invalidates exactly the dependent chunks.  qkv/mlp are position-wise, so dirty
    siblings in a level can be executed as one batched call (batch=True).
    """

    def __init__(self, params, H, chunks, tmax, cutoff="none", full=False, batch=False, fuse=False):
        self.fuse = fuse
        tag = cutoff + ("+batch" if batch else "") + ("+fuse" if fuse else "")
        self.name = "chunked-full" if full else f"incgraph[{tag}]"
        self.p, self.H, self.L, self.full, self.batch = params, H, len(params), full, batch
        self.g = Graph(cutoff=cutoff)
        self.x = [[] for _ in range(self.L + 1)]
        self.qkv = [[] for _ in range(self.L)]
        self.n = 0
        for c in chunks:
            self.append_chunk(c)
        self.g.evaluate(self.x[self.L])
        self.g.reset_stats()

    def append_chunk(self, xc):
        g, i, C = self.g, self.n, xc.shape[1]
        self.x[0].append(g.input(f"x.{i}", xc, group="x"))
        for l, p in enumerate(self.p):
            H = self.H

            def attn(x, own, *prev, p=p, i=i, C=C):
                q, k, v = own
                K = torch.cat([pv[1] for pv in prev] + [k], 2) if prev else k
                V = torch.cat([pv[2] for pv in prev] + [v], 2) if prev else v
                return attn_out(chunk_attention(q, K, V, i, C), x, p)

            bq = (("qkv", l), lambda al, p=p, H=H: batch_qkv(al, p, H)) if self.batch else None
            bm = (("mlp", l), lambda al, p=p: batch_mlp(al, p)) if self.batch else None
            ba = (("att", l), lambda al, p=p, C=C: batch_att(al, p, C)) if self.batch else None
            qkv = g.op(f"qkv{l}.{i}", lambda x, p=p, H=H: qkv_proj(x, p, H), [self.x[l][i]],
                       batch=bq, group=f"qkv{l}")
            self.qkv[l].append(qkv)
            if self.fuse:                         # attention + MLP as ONE node: att output is never cached
                bb = (("blk", l), lambda al, p=p, C=C: batch_mlp([[y] for y in batch_att(al, p, C, own=False)], p)) \
                    if self.batch else None
                self.x[l + 1].append(g.op(f"blk{l}.{i}", lambda x, own, *prev, p=p, attn=attn: mlp_block(attn(x, own, *prev), p),
                                          [self.x[l][i], qkv] + self.qkv[l][:i], batch=bb, group=f"blk{l}"))
                continue
            att = g.op(f"att{l}.{i}", attn, [self.x[l][i], qkv] + self.qkv[l][:i], batch=ba, group=f"att{l}")
            self.x[l + 1].append(g.op(f"mlp{l}.{i}", lambda h, p=p: mlp_block(h, p), [att],
                                      batch=bm, group=f"mlp{l}"))
        self.n += 1

    def step(self, arg):
        if arg[0] == "edit":
            self.g.set_input(self.x[0][arg[1]], arg[2])
        else:
            self.append_chunk(arg[1])
        if self.full:
            self.g.invalidate_all()
        self.g.evaluate(self.x[self.L])

    def output(self):
        return torch.cat([n.value for n in self.x[self.L]], 1)

    def stats(self):
        return dict(self.g.stats)

    def reset_stats(self):
        self.g.reset_stats()

    def cache_bytes(self):
        return self.g.cache_bytes()


def factories(params, H):
    return {
        "ordinary": lambda ch, tmax: Ordinary(params, H, ch, tmax),
        "ordinary[memo]": lambda ch, tmax: Ordinary(params, H, ch, tmax, memo=True),
        "handkv": lambda ch, tmax: HandKV(params, H, ch, tmax),
        "handkv[memo]": lambda ch, tmax: HandKV(params, H, ch, tmax, memo=True),
        "chunked-full": lambda ch, tmax: IncSystem(params, H, ch, tmax, "none", full=True),
        "incgraph[none]": lambda ch, tmax: IncSystem(params, H, ch, tmax, "none"),
        "incgraph[exact]": lambda ch, tmax: IncSystem(params, H, ch, tmax, "exact"),
        "incgraph[none+batch]": lambda ch, tmax: IncSystem(params, H, ch, tmax, "none", batch=True),
        "incgraph[none+batch+fuse]": lambda ch, tmax: IncSystem(params, H, ch, tmax, "none", batch=True, fuse=True),
        "incgraph[slab]": lambda ch, tmax: SlabInc(params, H, ch, tmax),
        "incgraph[slab+memo]": lambda ch, tmax: SlabInc(params, H, ch, tmax, memo=True),
        "incgraph[region]": lambda ch, tmax: RegionInc(params, H, ch, tmax),
        "incgraph[region+memo]": lambda ch, tmax: RegionInc(params, H, ch, tmax, memo=True),
        "incgraph[region+packed]": lambda ch, tmax: RegionPacked(params, H, ch, tmax),
        "incgraph[region+packed+memo]": lambda ch, tmax: RegionPacked(params, H, ch, tmax, memo=True),
        "suffixkv": lambda ch, tmax: SuffixKV(params, H, ch, tmax),
        "suffixkv[memo]": lambda ch, tmax: SuffixKV(params, H, ch, tmax, memo=True),
        "incgraph[adaptive+batch]": lambda ch, tmax: IncSystem(params, H, ch, tmax, "adaptive", batch=True),
    }


# --------------------------------------------------------------------------- correctness
TOL = {torch.float32: 1e-3, torch.float16: 5e-2, torch.bfloat16: 1.5e-1}


def check_equivalence(device="cpu", dtype=torch.float32, L=2, D=32, H=4, C=8, N=6, steps=12, B=1):
    dev = torch.device(device)
    params = make_params(L, D, dev, dtype)
    gen = torch.Generator().manual_seed(5)
    rnd = lambda: torch.randn(B, C, D, generator=gen).to(dev, dtype)
    n0 = N // 2
    chunks = [rnd() for _ in range(n0)]
    systems = {k: f(list(chunks), N + steps) for k, f in factories(params, H).items()}
    ref = systems["ordinary"]
    rng = torch.Generator().manual_seed(9)
    n = n0
    for s in range(steps):
        if s % 3 == 2 and n < N:
            arg, n = ("append", rnd()), n + 1
        else:
            arg = ("edit", int(torch.randint(0, n, (1,), generator=rng)), rnd())
        if s % 4 == 3 and arg[0] == "edit":  # value-identical edit: cutoff must keep results exact
            i = arg[1]
            arg = ("edit", i, ref.buf[:, i * C:(i + 1) * C].clone())
        for sysm in systems.values():
            sysm.step(arg)
        want = ref.output()
        for k, sysm in systems.items():
            got = sysm.output()
            assert torch.allclose(got, want, atol=TOL[dtype], rtol=TOL[dtype]), \
                f"{k} diverged at step {s}: max err {(got - want).abs().max().item():.3e}"
    return True


def check_no_implicit_sync(sysm, arg):
    """On CUDA: prove incgraph[none] issues no host sync during an update."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        sysm.step(arg)
        return True
    except RuntimeError as e:
        return f"FAILED: {e}"
    finally:
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- measurement
def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def profile_steps(sysm, args):
    from collections import defaultdict
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for a in args:
            sysm.step(a)
        torch.cuda.synchronize()
    k = kt = mc = 0
    by_name = defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            d = getattr(e, "device_time", None)
            if d is None:
                d = getattr(e, "cuda_time", 0)
            if "memcpy" in e.name.lower() or "memset" in e.name.lower():
                mc += 1
            else:
                k += 1
                kt += d
            by_name[e.name[:70]] += d
    n = len(args)
    top = sorted(by_name.items(), key=lambda kv: -kv[1])[:8]
    return dict(kernels=k / n, kernel_ms=kt / 1e3 / n, memcpy=mc / n,
                top_kernels=[(nm, round(t / 1e3 / n, 3)) for nm, t in top])


def measure(sysm, args, warm, nprof, dev):
    cuda = dev.type == "cuda"
    timed = args[warm:len(args) - nprof]
    for a in args[:warm]:
        sysm.step(a)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    sysm.reset_stats()
    wall, host, gpu = [], [], []
    for a in timed:
        if cuda:
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            e0.record()
            sysm.step(a)
            t1 = time.perf_counter()
            e1.record()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            gpu.append(e0.elapsed_time(e1))
        else:
            t0 = time.perf_counter()
            sysm.step(a)
            t1 = t2 = time.perf_counter()
        wall.append((t2 - t0) * 1e3)
        host.append((t1 - t0) * 1e3)
    n = len(timed)
    r = dict(wall_p50=pct(wall, .5), wall_p10=pct(wall, .1), wall_p90=pct(wall, .9),
             host_p50=pct(host, .5), n=n)
    if cuda:
        r["gpu_span_p50"] = pct(gpu, .5)
        r["peak_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20
    st = sysm.stats()
    r["executed"] = st.get("executed", 0) / n if st else None
    r["syncs"] = st.get("syncs", 0) / n if st else 0
    r["cutoff_hits"] = st.get("cutoff", 0) / n if st else 0
    r["checks"] = st.get("checks", 0) / n if st else None
    r["batch_calls"] = st.get("batch_calls", 0) / n if st else None
    r["cache_MiB"] = sysm.cache_bytes() / 2 ** 20
    if cuda and nprof:
        r.update(profile_steps(sysm, args[len(args) - nprof:]))
    elif nprof:
        for a in args[len(args) - nprof:]:
            sysm.step(a)
    return r


# --------------------------------------------------------------------------- sweep
def run_sweep(a, dev, dtype, params, facs, N, rnd):
    """Edit each chunk position p=0..N-1; wall p50 per system; crossover = first p from which the
    system beats `ordinary` at every later position (None if it never does / only at some p)."""
    names = [n for n in a.sweep_systems.split(",") if n in facs]
    if "ordinary" not in names:
        names.insert(0, "ordinary")
    base = [rnd() for _ in range(N)]
    for i in range(1, N):
        chunk_mask(i, a.C, dev)
        range_mask(i, N - i, a.C, dev)
    warm, total = 3, 3 + a.sweep_iters
    rows = []
    print("\n== sweep: wall p50 ms by edited chunk position (dirty chunks = N-p)")
    print(f"{'pos':>4}" + "".join(f"{n:>26}" for n in names))
    for p in range(N):
        args = [("edit", p, rnd()) for _ in range(total)]
        row = dict(pos=p)
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if a.interleave:
            sysms = {nm: facs[nm](list(base), N) for nm in names}
            res = measure_interleaved(sysms, args, warm, 0, dev)
            for nm in names:
                row[nm] = res[nm]["wall_p50"]
            del sysms
        else:
            for nm in names:
                sysm = facs[nm](list(base), N)
                row[nm] = measure(sysm, args, warm, 0, dev)["wall_p50"]
                del sysm
        rows.append(row)
        print(f"{p:>4}" + "".join(f"{row[n]:>26.2f}" for n in names))
    for nm in names:
        if nm == "ordinary":
            continue
        win = [r[nm] < r["ordinary"] for r in rows]
        cross = next((r["pos"] for r in rows if all(win[r["pos"]:])), None)
        print(f"  {nm:<26} beats ordinary at {sum(win)}/{N} positions; "
              f"crossover (wins from here to the end): {cross if cross is not None else 'none'}")
    return rows


def measure_interleaved(sysms, args, warm, nprof, dev):
    """Round-robin over systems each iteration (rotating start) so thermal/clock drift hits all equally."""
    cuda = dev.type == "cuda"
    names = list(sysms)
    timed = args[warm:len(args) - nprof]
    for a in args[:warm]:
        for s in sysms.values():
            s.step(a)
    if cuda:
        torch.cuda.synchronize()
    for s in sysms.values():
        s.reset_stats()
    wall = {n: [] for n in names}
    host = {n: [] for n in names}
    for k, a in enumerate(timed):
        r0 = k % len(names)
        for n in names[r0:] + names[:r0]:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            sysms[n].step(a)
            t1 = time.perf_counter()
            if cuda:
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            wall[n].append((t2 - t0) * 1e3)
            host[n].append((t1 - t0) * 1e3)
    nt, out = len(timed), {}
    for n in names:
        st = sysms[n].stats()
        out[n] = dict(wall_p50=pct(wall[n], .5), wall_p10=pct(wall[n], .1), wall_p90=pct(wall[n], .9),
                      host_p50=pct(host[n], .5), n=nt,
                      executed=st.get("executed", 0) / nt if st else None,
                      syncs=st.get("syncs", 0) / nt if st else 0,
                      cutoff_hits=st.get("cutoff", 0) / nt if st else 0,
                      checks=st.get("checks", 0) / nt if st else None,
                      batch_calls=st.get("batch_calls", 0) / nt if st else None,
                      cache_MiB=sysms[n].cache_bytes() / 2 ** 20)
    tail = args[len(args) - nprof:] if nprof else []
    for n in names:                                   # profiling is sequential (profiler cannot interleave)
        if cuda and nprof:
            out[n].update(profile_steps(sysms[n], tail))
        else:
            for a in tail:
                sysms[n].step(a)
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--C", type=int, default=128, help="chunk (node) granularity in tokens")
    ap.add_argument("--D", type=int, default=768)
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--nprof", type=int, default=5)
    ap.add_argument("--systems", default="all")
    ap.add_argument("--out", default="bench_results_v2.json")
    ap.add_argument("--breakdown", action="store_true", help="print top-8 CUDA kernels by time per row")
    ap.add_argument("--interleave", action="store_true", help="round-robin systems per iteration (cancels clock drift)")
    ap.add_argument("--compact", action="store_true", help="clone cached split() pieces (memory fix, costs kernels)")
    ap.add_argument("--sweep", action="store_true", help="edit every chunk position; report crossover vs ordinary")
    ap.add_argument("--sweep-systems", default="ordinary,suffixkv,incgraph[slab],incgraph[region+packed],incgraph[none+batch]")
    ap.add_argument("--sweep-iters", type=int, default=15)
    a = ap.parse_args()

    global OWN_PIECES
    OWN_PIECES = a.compact
    dev, dtype = torch.device(a.device), getattr(torch, a.dtype)
    if dev.type == "cpu" and dtype != torch.float32:
        raise SystemExit("use float32 on CPU")
    N = a.T // a.C
    print(f"device={dev} dtype={a.dtype} B={a.B} T={a.T} C={a.C} (N={N} chunks) D={a.D} H={a.H} L={a.L}")
    if dev.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(dev), "| torch", torch.__version__)
    else:
        print("WARNING: CPU run. No kernel/sync/GPU-memory metrics; timings are not GPU-representative.")

    print("verifying correctness (small model) ...", end=" ", flush=True)
    check_equivalence(a.device, dtype, L=2, D=64, H=4, C=16, N=6, steps=14)
    check_equivalence(a.device, dtype, L=2, D=64, H=4, C=16, N=6, steps=14, B=2)   # non-fast executor paths
    print("ok")

    params = make_params(a.L, a.D, dev, dtype)
    gen = torch.Generator().manual_seed(1)
    rnd = lambda: torch.randn(a.B, a.C, a.D, generator=gen).to(dev, dtype)
    base = [rnd() for _ in range(N)]
    for i in range(1, N):
        chunk_mask(i, a.C, dev)                          # keep mask construction out of timings
    facs = factories(params, a.H)
    names = list(facs) if a.systems == "all" else a.systems.split(",")
    unknown = [n for n in names if n not in facs]
    if unknown:
        raise SystemExit(f"unknown systems {unknown}; choose from {list(facs)}")
    names.sort(key=lambda n: n != "ordinary")          # reference first, so outputs can be verified
    total = a.warmup + a.iters + a.nprof

    scenarios = {
        "edit-first": (base, [("edit", 0, rnd()) for _ in range(total)]),
        "edit-mid": (base, [("edit", N // 2, rnd()) for _ in range(total)]),
        "edit-last": (base, [("edit", N - 1, rnd()) for _ in range(total)]),
        "noop-mid": (base, [("edit", N // 2, base[N // 2].clone()) for _ in range(total)]),
    }
    n0 = N // 2
    app = [("append", rnd()) for _ in range(N - n0)]
    scenarios["append"] = (base[:n0], app)

    results, bad = [], []
    for sc, (init, args) in scenarios.items():
        ref_out = None
        warm, nprof = a.warmup, a.nprof
        if sc == "append":
            warm, nprof = 1, min(a.nprof, 2)   # keep most of the few append steps for timing
        sysms, pre = None, None
        if a.interleave:                               # all systems alive at once; peak MiB not reported
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            sysms = {nm: facs[nm](list(init), N) for nm in names}
            pre = measure_interleaved(sysms, args, warm, nprof, dev)
        for nm in names:
            if a.interleave:
                sysm, r = sysms[nm], pre[nm]
            else:
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                sysm = facs[nm](list(init), N)
                if sc == "edit-mid" and nm == "incgraph[none]" and dev.type == "cuda":
                    print("  implicit host sync during incgraph[none] update:",
                          check_no_implicit_sync(sysm, args[0]) is True and "none (verified)" or "DETECTED")
                r = measure(sysm, args, warm, nprof, dev)
            r.update(scenario=sc, system=nm)
            out = sysm.output()
            if nm == "ordinary":
                ref_out = out.clone()
            if ref_out is not None:                    # same edit stream => same final state
                err = (out - ref_out).abs().max().item()
                r["max_err_vs_ordinary"] = err
                r["verified"] = bool(torch.allclose(out, ref_out, atol=TOL[dtype], rtol=TOL[dtype]))
                if not r["verified"]:
                    bad.append((sc, nm, err))
            else:
                r["verified"] = None
            results.append(r)
            del sysm
        sysms = pre = None

    # ---- report
    for sc in scenarios:
        rows = [r for r in results if r["scenario"] == sc]
        ref = next((r for r in rows if r["system"] == "ordinary"), None)
        print(f"\n== {sc}")
        hdr = f"{'system':<26}{'wall p50 [p10-p90] ms':>26}{'speedup':>9}{'host ms':>9}{'gpuspan':>9}" \
              f"{'kern/step':>10}{'kern ms':>9}{'memcpy':>7}{'syncs':>6}{'nodes':>7}{'checks':>7}{'bcalls':>7}{'peak MiB':>10}{'cache MiB':>10}"
        print(hdr)
        for r in rows:
            sp = f"{ref['wall_p50'] / r['wall_p50']:.2f}x" if ref else "-"
            f = lambda k, fmt: (fmt % r[k]) if r.get(k) is not None else "-"
            tag = {True: "", False: " !MISMATCH", None: " (unverified)"}[r.get("verified")]
            print(f"{r['system'] + tag:<26}{r['wall_p50']:>12.2f} [{r['wall_p10']:.2f}-{r['wall_p90']:.2f}]".ljust(54)
                  + f"{sp:>9}{r['host_p50']:>9.2f}{f('gpu_span_p50', '%.2f'):>9}{f('kernels', '%.0f'):>10}"
                  f"{f('kernel_ms', '%.2f'):>9}{f('memcpy', '%.1f'):>7}{r['syncs']:>6.1f}"
                  f"{f('executed', '%.0f'):>7}{f('checks', '%.0f'):>7}{f('batch_calls', '%.0f'):>7}"
                  f"{f('peak_MiB', '%.0f'):>10}{r['cache_MiB']:>10.0f}")
            if a.breakdown and r.get("top_kernels"):
                for nm, t in r["top_kernels"][:5]:
                    print(f"      {t:8.3f} ms  {nm}")
    sweep_rows = run_sweep(a, dev, dtype, params, facs, N, rnd) if a.sweep else None
    with open(a.out, "w") as fh:
        json.dump(dict(config=vars(a), results=results, sweep=sweep_rows), fh, indent=1)
    print(f"\nwrote {a.out}")
    if bad:
        print("\nCORRECTNESS FAILURES (system diverged from ordinary at the benchmarked config):")
        for sc, nm, err in bad:
            print(f"  {sc:<12}{nm:<26}max abs err {err:.3e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
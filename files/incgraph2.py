"""
incgraph v2.1 -- dependency-tracked incremental tensor graph, re-engineered from the T4 results.

v2.1 correctness fixes (all covered by tests):
  * exception safety: a failing node (or a failing eq/resolve) no longer leaves other nodes of the
    same level marked clean while still holding stale values (this was silent wrong output).
  * torch_eq: structure-aware (tuple/list/dict nesting must match), mixed tensor/non-tensor leaves
    compare unequal instead of raising, and view de-duplication now also requires identical view
    geometry (offset/stride) -- previously two *different* views of equal bases were reported equal.

What the v1 benchmark measured, and what v2 does about it
---------------------------------------------------------
1. Early cutoff cost 37 host syncs and ~1.8x the kernels per update, and paid off only for
   no-op edits.                     -> cutoff="adaptive": per-group hit-rate policy that stops
   checking where checks never hit, and re-probes occasionally.  A skipped check is always
   safe (it only forgoes a skip).  Plus: view-deduplicated equality (fewer kernels per check).
2. incgraph[none] lost to ordinary execution when many chunks were dirty, because 128-token
   nodes launch ~23x more kernels and run small GEMMs.
                                    -> level-coalesced batching: dirty nodes of one level that
   declare the same batch key execute through ONE user-supplied batch_fn call.

Everything else (versioned validation, demand-driven evaluation, dynamic rewiring, functional
stateful nodes with ticks) is unchanged in meaning from v1; see the test-suite for the contract.

Prior art this builds on (not claimed as novel): demand-driven incremental computation
(Hammer, Khoo, Hicks, Foster, "Adapton", PLDI 2014); early cutoff / constructive traces (Mokhov,
Mitchell, Peyton Jones, "Build Systems a la Carte", ICFP 2018; extended: Mokhov, Mitchell, Peyton
Jones, Marlow, "Build systems a la carte: Theory and practice", JFP 2020); on-the-fly operation
batching in dynamic graphs (Neubig, Goldberg, Dyer, NIPS 2017, now NeurIPS).
Reuse of an unchanged token prefix's KV cache is established practice in LLM serving (e.g. SGLang's
RadixAttention, Zheng et al., NeurIPS 2024): the edit-last / append speedups are that known effect,
not a new one.  What is specific here is arbitrary-DAG invalidation + early cutoff + dynamic rewiring.
See PRIOR_ART_AND_POSITIONING.md.

Contract on user functions: pure; must not mutate inputs/outputs in place.  A batch_fn for key
k must return, for each node's argument list, what that node's own fn would return (bitwise
identical is not required; equal up to floating-point reassociation is fine, cutoff just sees 'changed').
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional, Sequence

try:
    import torch
except ImportError:
    torch = None

_KEEP = object()


class CycleError(Exception):
    pass


# --------------------------------------------------------------------------- helpers
def _is_t(x):
    return torch is not None and isinstance(x, torch.Tensor)


def _items(d):
    return sorted(d.items(), key=lambda kv: repr(kv[0]))


def _flat(x):
    if _is_t(x):
        return [x]
    if isinstance(x, (tuple, list)):
        out = []
        for y in x:
            out += _flat(y)
        return out
    if isinstance(x, dict):
        out = []
        for _, y in _items(x):
            out += _flat(y)
        return out
    return [x]


def _sig(x):
    """Nesting structure with tensor/non-tensor leaves; two outputs can only be equal if this matches."""
    if _is_t(x):
        return "T"
    if isinstance(x, (tuple, list)):
        return (type(x).__name__, tuple(_sig(y) for y in x))
    if isinstance(x, dict):
        return ("dict", tuple((repr(k), _sig(v)) for k, v in _items(x)))
    return "L"


def _root(t):
    return t._base if t._base is not None else t


def _geom(t):
    return (t.shape, t.dtype, t.stride(), t.storage_offset())


def torch_eq(a, b, atol=0.0):
    """bool, or a 0-d bool tensor (no host sync here).

    View dedup: leaf pairs that are views of the SAME-GEOMETRY bases with the SAME view geometry
    (offset + strides) are compared once, through the base.  Equal bases + identical view geometry
    imply equal views, so this is sound.  Anything else is compared view-to-view.  A pair that is
    reported 'changed' while its values happen to match only forgoes a skip (safe)."""
    if _sig(a) != _sig(b):
        return False
    flags, done = [], set()
    for x, y in zip(_flat(a), _flat(b)):
        if x is y:
            continue
        if not _is_t(x):
            if x != y:
                return False
            continue
        if x.shape != y.shape or x.dtype != y.dtype or x.device != y.device:
            return False
        if x.data_ptr() == y.data_ptr() and x.stride() == y.stride():
            continue                           # literally the same memory (both alive: cannot be recycled)
        bx, by = _root(x), _root(y)
        if _geom(x)[2:] == _geom(y)[2:] and _geom(bx) == _geom(by):
            key = (id(bx), id(by))
            cx, cy = bx, by
        else:
            key = (id(x), id(y))
            cx, cy = x, y
        if key in done:
            continue
        done.add(key)
        flags.append((cx == cy).all() if atol == 0 else ((cx - cy).abs() <= atol).all())
    if not flags:
        return True
    return flags[0] if len(flags) == 1 else torch.stack(flags).all()


def torch_resolve(flags):
    """list[bool | 0-d tensor] -> (list[bool], n_host_syncs).  One sync per device."""
    out = [f if isinstance(f, bool) else None for f in flags]
    pend = [(i, f) for i, f in enumerate(flags) if out[i] is None]
    if not pend:
        return out, 0
    by_dev = defaultdict(list)
    for i, f in pend:
        by_dev[f.device].append((i, f))
    syncs = 0
    for items in by_dev.values():
        vals = torch.stack([f for _, f in items]).tolist()
        syncs += 1
        for (i, _), v in zip(items, vals):
            out[i] = bool(v)
    return out, syncs


# --------------------------------------------------------------------------- nodes
class Node:
    __slots__ = ("g", "id", "name", "fn", "deps", "kind", "users", "level", "value",
                 "version", "seen", "dirty", "force", "struct", "alive", "pending",
                 "has_pending", "stateful", "always_tick", "state_in", "state_out",
                 "ran_this_tick", "batch", "group")

    def __init__(self, g, nid, name, fn, deps, kind):
        self.g, self.id, self.name, self.fn, self.kind = g, nid, name, fn, kind
        self.deps = list(deps)
        self.users = {}
        self.level = 0
        self.value = None
        self.version = 0
        self.seen = None
        self.dirty = True
        self.force = False
        self.struct = 0
        self.alive = True
        self.pending = None
        self.has_pending = False
        self.stateful = False
        self.always_tick = False
        self.state_in = None
        self.state_out = None
        self.ran_this_tick = False
        self.batch = None          # (key, batch_fn) or None
        self.group = nid           # cutoff-policy sharing group

    def __repr__(self):
        return f"<Node {self.name}#{self.id} v{self.version}{' dirty' if self.dirty else ''}>"


# --------------------------------------------------------------------------- graph
class Graph:
    """cutoff: "none" | "exact" | "tol" | "adaptive" (exact equality + hit-rate policy)."""

    def __init__(self, cutoff: str = "none", atol: float = 0.0,
                 eq: Optional[Callable] = None, resolve: Optional[Callable] = None,
                 alpha: float = 0.25, off_below: float = 0.05, probe_every: int = 32,
                 min_trials: int = 3):
        if cutoff not in ("none", "exact", "tol", "adaptive"):
            raise ValueError(cutoff)
        self.cutoff = cutoff
        self.alpha, self.off_below, self.probe_every, self.min_trials = alpha, off_below, probe_every, min_trials
        if cutoff != "none":
            if eq is None:
                if torch is None:
                    raise RuntimeError("cutoff needs torch or explicit eq=/resolve=")
                a = atol if cutoff == "tol" else 0.0
                eq = lambda x, y: torch_eq(x, y, a)
            resolve = resolve or torch_resolve
        self.eq, self.resolve = eq, resolve
        self.nodes: dict[int, Node] = {}
        self.stateful_nodes: list[Node] = []
        self._ids = 0
        self._pol: dict = {}       # group -> [hit_rate_ema, since_probe, trials]
        self.stats = {}
        self.reset_stats()

    def reset_stats(self):
        self.stats = dict(executed=0, skipped=0, cutoff=0, syncs=0, checks=0,
                          unchecked=0, batch_calls=0, batched_nodes=0, eq_errors=0)

    # ---- construction ------------------------------------------------------
    def _new(self, name, fn, deps, kind):
        self._ids += 1
        n = Node(self, self._ids, name, fn, deps, kind)
        self.nodes[n.id] = n
        return n

    def input(self, name, value=None, *, group=None) -> Node:
        n = self._new(name, None, [], "input")
        if group is not None:
            n.group = group
        if value is not None:
            n.pending, n.has_pending = value, True
        return n

    def op(self, name, fn, deps: Sequence[Node], *, stateful=False, state=None,
           always_tick=False, batch=None, group=None) -> Node:
        deps = list(deps)
        for d in deps:
            if d.g is not self or not d.alive:
                raise ValueError(f"bad dependency {d}")
        if batch is not None and stateful:
            raise ValueError("stateful nodes cannot be batched")
        n = self._new(name, fn, deps, "op")
        n.level = 1 + max((d.level for d in deps), default=0)
        for d in deps:
            d.users[n.id] = n
        n.batch = batch
        if group is not None:
            n.group = group
        if stateful:
            n.stateful, n.state_in, n.always_tick = True, state, always_tick
            self.stateful_nodes.append(n)
        return n

    # ---- invalidation ------------------------------------------------------
    def _mark_dirty(self, n):
        stack = [n]
        while stack:
            m = stack.pop()
            if m.dirty:
                continue
            m.dirty = True
            stack.extend(m.users.values())

    def set_input(self, n: Node, value):
        if n.kind != "input":
            raise ValueError("set_input on a non-input node")
        n.pending, n.has_pending = value, True
        self._mark_dirty(n)

    def invalidate_all(self):
        for n in self.nodes.values():
            if n.kind == "op":
                n.dirty = n.force = True

    # ---- dynamic wiring ----------------------------------------------------
    def _descendants(self, n):
        seen, stack = {n.id}, [n]
        while stack:
            for u in stack.pop().users.values():
                if u.id not in seen:
                    seen.add(u.id)
                    stack.append(u)
        return seen

    def _relevel(self, n):
        stack = [n]
        while stack:
            m = stack.pop()
            lv = 1 + max((d.level for d in m.deps), default=0) if m.kind == "op" else 0
            if lv != m.level:
                m.level = lv
                stack.extend(m.users.values())

    def rewire(self, n: Node, deps: Optional[Sequence[Node]] = None, fn=None, batch=_KEEP):
        """Changing fn without passing batch= clears the batch spec (it may no longer match)."""
        if n.kind != "op":
            raise ValueError("cannot rewire an input")
        if deps is not None:
            deps = list(deps)
            desc = self._descendants(n)
            for d in deps:
                if d.g is not self or not d.alive:
                    raise ValueError(f"bad dependency {d}")
                if d.id in desc:
                    raise CycleError(f"{d} depends on {n}")
            for d in n.deps:
                d.users.pop(n.id, None)
            n.deps = deps
            for d in deps:
                d.users[n.id] = n
            self._relevel(n)
        if fn is not None:
            n.fn = fn
            if batch is _KEEP:
                n.batch = None
        if batch is not _KEEP:
            if batch is not None and n.stateful:
                raise ValueError("stateful nodes cannot be batched")
            n.batch = batch
        n.struct += 1
        self._mark_dirty(n)

    def remove(self, n: Node):
        if n.users:
            raise ValueError(f"{n} still has dependents")
        for d in n.deps:
            d.users.pop(n.id, None)
        del self.nodes[n.id]
        n.alive = False
        n.value = n.pending = n.state_in = n.state_out = None
        if n.stateful:
            self.stateful_nodes.remove(n)

    # ---- execution ---------------------------------------------------------
    def evaluate(self, targets=None):
        if targets is None:
            targets = [n for n in self.nodes.values() if not n.users]
        elif isinstance(targets, Node):
            targets = [targets]
        need, stack = {}, [t for t in targets if t.dirty]
        while stack:
            n = stack.pop()
            if n.id in need:
                continue
            need[n.id] = n
            stack.extend(d for d in n.deps if d.dirty)
        levels = defaultdict(list)
        for n in need.values():
            levels[n.level].append(n)
        for lv in sorted(levels):
            self._run_level(levels[lv])
        return [t.value for t in targets]

    def _plan(self, n):
        """None -> nothing to execute (served from cache); otherwise the validation key."""
        if n.kind == "input":
            if not n.has_pending:
                raise RuntimeError(f"input {n.name} has no value")
            return ()
        cur = (n.struct, tuple((d.id, d.version) for d in n.deps))
        if n.version and not n.force and n.seen == cur:
            n.dirty = False
            self.stats["skipped"] += 1
            return None
        return cur

    def _run_level(self, nodes):
        deferred = []
        try:
            self._run_level_body(nodes, deferred)
        except BaseException:
            # Outputs already computed (and marked clean) but not yet cutoff-checked must be committed,
            # otherwise those nodes would stay clean while holding stale values.  Committing as
            # 'changed' is always safe: it only forgoes a skip.
            self._commit_changed(deferred)
            raise
        if deferred:
            self._settle(deferred)

    def _commit_changed(self, deferred):
        for n, out, _ in deferred:
            n.value, n.version = out, n.version + 1
        deferred.clear()

    def _run_level_body(self, nodes, deferred):
        st = self.stats
        groups, singles = defaultdict(list), []
        for n in nodes:
            cur = self._plan(n)
            if cur is None:
                continue
            if n.kind == "input":
                out, n.pending, n.has_pending = n.pending, None, False
                self._finish(n, out, deferred)
            elif n.batch is not None:
                groups[n.batch[0]].append((n, cur))
            else:
                singles.append((n, cur))
        for n, cur in singles:
            self._exec_one(n, cur, deferred)
        for items in groups.values():
            if len(items) == 1:
                self._exec_one(*items[0], deferred)
                continue
            outs = items[0][0].batch[1]([[d.value for d in n.deps] for n, _ in items])
            if len(outs) != len(items):
                raise RuntimeError("batch_fn returned wrong number of outputs")
            st["batch_calls"] += 1
            st["batched_nodes"] += len(items)
            for (n, cur), out in zip(items, outs):
                n.seen, n.force = cur, False
                st["executed"] += 1
                self._finish(n, out, deferred)

    def _exec_one(self, n, cur, deferred):
        args = [d.value for d in n.deps]
        if n.stateful:
            out, new_state = n.fn(n.state_in, *args)
            n.state_out, n.ran_this_tick = new_state, True
        else:
            out = n.fn(*args)
        n.seen, n.force = cur, False
        self.stats["executed"] += 1
        self._finish(n, out, deferred)

    def _should_check(self, n):
        if self.cutoff == "none" or not n.version:
            return False
        if self.cutoff != "adaptive":
            return True
        p = self._pol.setdefault(n.group, [1.0, 0, 0])
        if p[2] < self.min_trials or p[0] >= self.off_below:
            return True
        p[1] += 1
        if p[1] >= self.probe_every:
            p[1] = 0
            return True
        return False

    def _finish(self, n, out, deferred):
        n.dirty = False
        if self._should_check(n):
            try:
                flag = self.eq(n.value, out)
            except Exception:
                self.stats["eq_errors"] += 1
                flag = False                       # cannot prove equality -> treat as changed (safe)
            deferred.append((n, out, flag))
        else:
            if self.cutoff != "none" and n.version:
                self.stats["unchecked"] += 1
            n.value, n.version = out, n.version + 1

    def _settle(self, deferred):
        try:
            res, syncs = self.resolve([f for _, _, f in deferred])
        except BaseException:
            self._commit_changed(deferred)
            raise
        st = self.stats
        st["syncs"] += syncs
        st["checks"] += len(deferred)
        for (n, out, _), same in zip(deferred, res):
            if self.cutoff == "adaptive":
                p = self._pol[n.group]
                p[0] = (1 - self.alpha) * p[0] + self.alpha * (1.0 if same else 0.0)
                p[2] += 1
            if same:
                st["cutoff"] += 1
            else:
                n.value, n.version = out, n.version + 1

    def tick(self):
        for n in self.stateful_nodes:
            if n.ran_this_tick:
                n.state_in, n.ran_this_tick = n.state_out, False
            if n.always_tick:
                n.force = True
                self._mark_dirty(n)

    # ---- introspection -----------------------------------------------------
    def policy_snapshot(self):
        return {g: dict(hit_rate=round(p[0], 3), trials=p[2]) for g, p in self._pol.items()}

    def cache_bytes(self) -> int:
        if torch is None:
            return 0
        seen, tot = set(), 0
        for n in self.nodes.values():
            for obj in (n.value, n.pending, n.state_in, n.state_out):
                for t in _flat(obj):
                    if isinstance(t, torch.Tensor):
                        s = t.untyped_storage()
                        key = (str(t.device), s.data_ptr())
                        if key not in seen:
                            seen.add(key)
                            tot += s.nbytes()
        return tot

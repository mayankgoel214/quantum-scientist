"""Choosing which physical qubits host the GHZ state, and in which tree.

All strategies return {"root", "qubits", "tree_edges", "depth"} where tree_edges are
[parent, child] physical-qubit pairs in CNOT schedule order and depth is the number of
parallel CNOT layers of the tree schedule (see schedule_tree).
"""
from __future__ import annotations

import heapq
import math
from collections import deque

UNUSABLE_ERROR = 0.5  # edges / readouts at or above this are treated as broken
LAYER_US = 0.15  # rough duration of one CNOT layer on Heron (CZ 84 ns + 1q gates)
READOUT_WEIGHT = 1.0
DEFAULT_EDGE_ERROR = 0.02  # used when calibration has no number for an edge
DEFAULT_RO_ERROR = 0.03
DEFAULT_T2_US = 100.0


# ---------------------------------------------------------------- graph helpers
def _graph(cal: dict, exclude) -> tuple[dict[int, dict[int, float]], dict[int, float], dict[int, float]]:
    excl = set(int(q) for q in exclude or ())
    ro, t2 = {}, {}
    for q in cal["qubits"]:
        qid = int(q["id"])
        r = q.get("readout_error")
        r = DEFAULT_RO_ERROR if r is None else float(r)
        if qid in excl or r >= UNUSABLE_ERROR:
            continue
        ro[qid] = r
        t = q.get("t2_us")
        t2[qid] = float(t) if t not in (None, 0) else DEFAULT_T2_US
    adj: dict[int, dict[int, float]] = {q: {} for q in ro}
    for e in cal["edges"]:
        a, b = int(e["a"]), int(e["b"])
        if a not in adj or b not in adj:
            continue
        err = e.get("error")
        err = DEFAULT_EDGE_ERROR if err is None else float(err)
        if err >= UNUSABLE_ERROR:
            continue
        c = -math.log(1.0 - err)
        adj[a][b] = min(c, adj[a].get(b, math.inf))
        adj[b][a] = adj[a][b]
    return adj, ro, t2


def _ro_cost(r: float) -> float:
    return -READOUT_WEIGHT * math.log(1.0 - r)


# ---------------------------------------------------------------- scheduling
def schedule_tree(root: int, tree_edges: list) -> list[list[tuple[int, int]]]:
    """Optimal broadcast schedule on a fixed tree: each qubit that already holds the GHZ
    state may CNOT one new child per layer. Children with the deepest subtrees go first.
    Returns a list of layers, each a list of (parent, child)."""
    children: dict[int, list[int]] = {}
    for p, c in tree_edges:
        children.setdefault(int(p), []).append(int(c))
    need: dict[int, int] = {}

    def solve(v: int) -> int:  # iterative-safe for our sizes (<=156)
        kids = children.get(v, [])
        for k in kids:
            solve(k)
        kids.sort(key=lambda k: -need[k])
        need[v] = max([i + 1 + need[k] for i, k in enumerate(kids)], default=0)
        return need[v]

    solve(root)
    layers: dict[int, list[tuple[int, int]]] = {}
    stack = [(root, 0)]
    while stack:
        v, t = stack.pop()
        for i, k in enumerate(children.get(v, [])):
            tk = t + i + 1
            layers.setdefault(tk, []).append((v, k))
            stack.append((k, tk))
    return [layers[t] for t in sorted(layers)]


def linear_schedule(root: int, tree_edges: list) -> list[list[tuple[int, int]]]:
    """One CNOT per layer, in BFS order of the tree (the 'linear' circuit variant)."""
    children: dict[int, list[int]] = {}
    for p, c in tree_edges:
        children.setdefault(int(p), []).append(int(c))
    out, q = [], deque([root])
    while q:
        v = q.popleft()
        for k in children.get(v, []):
            out.append([(v, k)])
            q.append(k)
    return out


def _result(root: int, tree_edges: list) -> dict:
    layers = schedule_tree(root, tree_edges)
    ordered = [[int(p), int(c)] for layer in layers for p, c in layer]
    qubits = [int(root)] + [c for _, c in ordered]
    return {"root": int(root), "qubits": qubits, "tree_edges": ordered, "depth": len(layers)}


def _component_size(adj, start) -> int:
    seen, q = {start}, deque([start])
    while q:
        v = q.popleft()
        for w in adj[v]:
            if w not in seen:
                seen.add(w)
                q.append(w)
    return len(seen)


def _check(adj, n, root):
    if n < 2:
        raise ValueError("n must be >= 2")
    if root is not None and root not in adj:
        raise ValueError(f"root {root} is excluded or unusable")
    if len(adj) < n:
        raise ValueError(f"only {len(adj)} usable qubits, asked for {n}")


# ---------------------------------------------------------------- weighted tree
def _weighted_tree_from(adj, ro, t2, n, root, depth_weight):
    """Prim-style growth: repeatedly attach the frontier qubit with the lowest cost
    -log(1-e_edge) - log(1-e_readout) + T2 penalty + (extra cost if it deepens the tree)."""
    time = {root: 0}
    nkids = {root: 0}
    edges = []
    total = _ro_cost(ro[root])
    max_t = 0
    in_tree = {root}
    t2_pen = {q: LAYER_US * (math.log2(max(n, 2)) + 2) / t2[q] for q in adj}

    def push(heap, u):
        for v, c in adj[u].items():
            if v in in_tree:
                continue
            tv = time[u] + nkids[u] + 1
            cost = c + _ro_cost(ro[v]) + t2_pen[v] + depth_weight * max(0, tv - max_t) + 1e-4 * tv
            heapq.heappush(heap, (cost, tv, u, v, nkids[u]))

    heap: list = []
    push(heap, root)
    while len(in_tree) < n and heap:
        cost, tv, u, v, k_at_push = heapq.heappop(heap)
        if v in in_tree:
            continue
        if k_at_push != nkids[u]:
            # parent gained children since this entry was pushed -> stale time; re-push fresh
            heapq.heappush(heap, _fresh(adj, ro, t2_pen, time, nkids, max_t, depth_weight, u, v))
            continue
        in_tree.add(v)
        edges.append((u, v))
        time[v] = tv
        nkids[u] += 1
        nkids[v] = 0
        max_t = max(max_t, tv)
        total += cost
        push(heap, v)
    if len(in_tree) < n:
        return None
    return _result(root, edges), total


def _fresh(adj, ro, t2_pen, time, nkids, max_t, depth_weight, u, v):
    tv = time[u] + nkids[u] + 1
    cost = adj[u][v] + _ro_cost(ro[v]) + t2_pen[v] + depth_weight * max(0, tv - max_t) + 1e-4 * tv
    return (cost, tv, u, v, nkids[u])


def _depth_weight(n, t2) -> float:
    med = sorted(t2.values())[len(t2) // 2] if t2 else DEFAULT_T2_US
    # one extra layer exposes every GHZ qubit to ~LAYER_US of dephasing
    return n * LAYER_US / med


def plan_weighted_tree(cal, n, root=None, exclude=()):
    adj, ro, t2 = _graph(cal, exclude)
    _check(adj, n, root)
    dw = _depth_weight(n, t2)
    candidates = [root] if root is not None else [q for q in adj if len(adj[q]) >= 2] or list(adj)
    best = None
    for r in candidates:
        if _component_size(adj, r) < n:
            continue
        out = _weighted_tree_from(adj, ro, t2, n, r, dw)
        if out is None:
            continue
        res, cost = out
        score = _score(res, adj, ro, t2, dw)
        if best is None or score < best[0]:
            best = (score, res)
    if best is None:
        raise ValueError(f"no connected set of {n} usable qubits")
    return best[1]


def _score(res, adj, ro, t2, dw) -> float:
    """Total predicted -log fidelity proxy of a planned tree (lower is better)."""
    s = sum(adj[p][c] for p, c in res["tree_edges"])
    s += sum(_ro_cost(ro[q]) for q in res["qubits"])
    s += dw * res["depth"]
    return s


# ---------------------------------------------------------------- BFS tree
def _bfs_from(adj, n, root):
    seen = {root}
    edges = []
    q = deque([root])
    while q and len(seen) < n:
        u = q.popleft()
        for v in sorted(adj[u], key=lambda w: adj[u][w]):
            if v not in seen and len(seen) < n:
                seen.add(v)
                edges.append((u, v))
                q.append(v)
    return None if len(seen) < n else _result(root, edges)


def plan_bfs_tree(cal, n, root=None, exclude=()):
    adj, ro, t2 = _graph(cal, exclude)
    _check(adj, n, root)
    dw = _depth_weight(n, t2)
    candidates = [root] if root is not None else list(adj)
    best = None
    for r in candidates:
        res = _bfs_from(adj, n, r)
        if res is None:
            continue
        key = (res["depth"], _score(res, adj, ro, t2, dw))
        if best is None or key < best[0]:
            best = (key, res)
    if best is None:
        raise ValueError(f"no connected set of {n} usable qubits")
    return best[1]


# ---------------------------------------------------------------- best chain
def plan_best_chain(cal, n, root=None, exclude=(), beam: int = 256):
    """Beam search for the lowest-cost simple path of n qubits (grown at both ends).
    The GHZ root is the middle of the path, so the tree schedule has depth ~ n/2."""
    adj, ro, t2 = _graph(cal, exclude)
    _check(adj, n, root)
    starts = [root] if root is not None else list(adj)
    states = [(_ro_cost(ro[s]), (s,)) for s in starts]
    for _ in range(n - 1):
        nxt = {}
        for cost, path in states:
            ps = set(path)
            for end, front in ((path[0], True), (path[-1], False)):
                for v, c in adj[end].items():
                    if v in ps:
                        continue
                    np_ = (v,) + path if front else path + (v,)
                    key = frozenset(np_), min(np_[0], np_[-1]), max(np_[0], np_[-1])
                    nc = cost + c + _ro_cost(ro[v])
                    if key not in nxt or nc < nxt[key][0]:
                        nxt[key] = (nc, np_)
        if not nxt:
            raise ValueError(f"no simple path of {n} usable qubits")
        states = heapq.nsmallest(beam, nxt.values(), key=lambda s: s[0])
    path = states[0][1]
    if root is not None and root in path:
        mid = path.index(root)
    else:
        mid = (len(path) - 1) // 2
    r = path[mid]
    edges = []
    for i in range(mid, 0, -1):
        edges.append((path[i], path[i - 1]))
    for i in range(mid, len(path) - 1):
        edges.append((path[i], path[i + 1]))
    return _result(r, edges)


STRATEGIES = {"weighted_tree": plan_weighted_tree, "bfs_tree": plan_bfs_tree, "best_chain": plan_best_chain}


def plan_layout(calibration: dict, n: int, strategy: str, root: int | None = None, exclude=()) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown layout_strategy {strategy!r}; expected one of {sorted(STRATEGIES)}")
    return STRATEGIES[strategy](calibration, int(n), None if root is None else int(root), list(exclude or ()))

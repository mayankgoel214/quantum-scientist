"""Generate a realistic, fully synthetic run directory for the mission-control UI.

    uv run python -m scripts.make_demo_run            # -> runs/demo-<timestamp>-qpu/
    (re-running overwrites the same demo-* directory; non-demo runs are never touched)

The output follows CONTRACTS.md exactly (calibration.json, events.jsonl, experiments/*.json,
budget.json). Calibration comes from qiskit's FakeFez (real ibm_fez snapshot); GHZ trees are real
connected subtrees of its coupling map, grown greedily over the lowest-error couplers. Fidelities are
hand-scripted to tell a plausible overnight story; nothing here touches real hardware.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import shutil
import statistics
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RNG = random.Random(1107)


# ---------------------------------------------------------------- calibration
def heavy_hex_xy(q: int) -> tuple[float, float]:
    """Heron r2 (156q) layout: 8 rows of 16 qubits at 20r..20r+15, 4 bridge qubits at 20r+16..19."""
    r, k = divmod(q, 20)
    if k < 16:
        return float(k), float(2 * r)
    # bridge k-16 in row-gap r: columns 3,7,11,15 for even r, 1,5,9,13 for odd r
    cols = (3, 7, 11, 15) if r % 2 == 0 else (1, 5, 9, 13)
    return float(cols[k - 16]), float(2 * r + 1)


def build_calibration() -> dict:
    from qiskit_ibm_runtime.fake_provider import FakeFez

    backend = FakeFez()
    props = backend.properties()
    n = backend.num_qubits

    def safe(fn, *a):
        try:
            v = fn(*a)
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        except Exception:
            return None

    qubits = []
    for q in range(n):
        x, y = heavy_hex_xy(q)
        t1, t2 = safe(props.t1, q), safe(props.t2, q)
        qubits.append({
            "id": q,
            "t1_us": round(t1 * 1e6, 1) if t1 else None,
            "t2_us": round(t2 * 1e6, 1) if t2 else None,
            "readout_error": round(safe(props.readout_error, q) or 0, 5) or None,
            "sx_error": safe(props.gate_error, "sx", q),
            "x": x, "y": y,
        })
    seen, edges = set(), []
    for a, b in backend.coupling_map.get_edges():
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        err = safe(props.gate_error, "cz", [a, b]) or safe(props.gate_error, "cz", [b, a])
        edges.append({"a": key[0], "b": key[1], "error": err, "gate": "cz"})
    edges.sort(key=lambda e: (e["a"], e["b"]))
    return {
        "backend": "ibm_fez", "num_qubits": n, "source": "fake",
        "timestamp": None,  # filled by caller
        "qubits": qubits, "edges": edges,
    }


# ---------------------------------------------------------------- layout
def grow_tree(cal: dict, n: int, root: int, exclude=(), weighted=True) -> dict:
    """Prim-style growth from root over lowest-error couplers (or plain BFS)."""
    adj: dict[int, list[tuple[int, float]]] = {}
    bad = set(exclude)
    for e in cal["edges"]:
        if e["a"] in bad or e["b"] in bad:
            continue
        w = e["error"] if e["error"] is not None else 0.05
        w = min(w, 0.2)
        adj.setdefault(e["a"], []).append((e["b"], w))
        adj.setdefault(e["b"], []).append((e["a"], w))
    depth = {root: 0}
    order, edges = [root], []
    if weighted:
        # cost mixes coupler error with depth so the tree stays bushy (shallow)
        heap = [(w + 0.004 * 1, root, c) for c, w in adj.get(root, [])]
        heapq.heapify(heap)
        while heap and len(order) < n:
            cost, p, c = heapq.heappop(heap)
            if c in depth:
                continue
            depth[c] = depth[p] + 1
            order.append(c)
            edges.append([p, c])
            for nb, w in adj.get(c, []):
                if nb not in depth:
                    heapq.heappush(heap, (w + 0.004 * (depth[c] + 1), c, nb))
    else:
        dq = deque([root])
        while dq and len(order) < n:
            p = dq.popleft()
            for c, _ in sorted(adj.get(p, [])):
                if c not in depth and len(order) < n:
                    depth[c] = depth[p] + 1
                    order.append(c)
                    edges.append([p, c])
                    dq.append(c)
    if len(order) < n:
        raise ValueError(f"could not grow tree of {n} from {root}")
    return {"root": root, "qubits": order, "tree_edges": edges, "depth": schedule_depth(root, edges)}


def grow_chain(cal: dict, n: int, root: int) -> dict:
    """Greedy best-error path starting at root."""
    adj: dict[int, list[tuple[int, float]]] = {}
    for e in cal["edges"]:
        w = e["error"] if e["error"] is not None else 0.05
        adj.setdefault(e["a"], []).append((e["b"], w))
        adj.setdefault(e["b"], []).append((e["a"], w))
    # find a simple path via DFS preferring low-error links
    best: list[int] = []

    def dfs(path, seen):
        nonlocal best
        if len(path) == n:
            best = list(path)
            return True
        for nb, _ in sorted(adj[path[-1]], key=lambda t: t[1]):
            if nb not in seen:
                seen.add(nb); path.append(nb)
                if dfs(path, seen):
                    return True
                path.pop(); seen.discard(nb)
        return False

    dfs([root], {root})
    # GHZ chain from the middle outward (depth ~ n/2): root = middle element
    mid = len(best) // 2
    r = best[mid]
    edges = [[best[mid], best[mid - 1]]] + [[best[i], best[i - 1]] for i in range(mid - 1, 0, -1)]
    edges += [[best[i], best[i + 1]] for i in range(mid, len(best) - 1)]
    return {"root": r, "qubits": best, "tree_edges": edges, "depth": schedule_depth(r, edges)}


def schedule_depth(root: int, edges: list[list[int]]) -> int:
    kids: dict[int, list[int]] = {}
    for p, c in edges:
        kids.setdefault(p, []).append(c)

    def size(v):
        return 1 + sum(size(c) for c in kids.get(v, []))

    t = {root: 0}
    stack = [root]
    while stack:
        v = stack.pop()
        for i, c in enumerate(sorted(kids.get(v, []), key=size, reverse=True)):
            t[c] = t[v] + 1 + i
            stack.append(c)
    return max(t.values())


# ---------------------------------------------------------------- results
def parity_curve(n: int, coherence: float) -> list[dict]:
    # ideal GHZ parity oscillates N times per 2π: <P_j> = C·cos(N·φ_j + δ), φ_j = jπ/(N+1)
    delta = RNG.gauss(0, 0.06)
    pts = []
    for j in range(2 * n + 2):
        phi = j * math.pi / (n + 1)
        val = coherence * math.cos(n * phi + delta) + RNG.gauss(0, 0.015)
        pts.append({"phi": round(phi, 5), "parity": round(max(-1, min(1, val)), 4)})
    return pts


def make_result(exp_id: str, spec: dict, layout: dict, fidelity: float | None, half_width: float,
                qpu_seconds: float, wall_seconds: float, error: str | None = None) -> dict:
    n = spec["n_qubits"]
    base = {
        "exp_id": exp_id, "spec": spec,
        "physical_qubits": layout["qubits"], "tree_edges": layout["tree_edges"], "root": layout["root"],
        "circuit_depth": layout["depth"] + 3, "two_qubit_gate_count": n - 1,
        "backend": "ibm_fez" if spec["mode"] == "qpu" else "fake_fez (aer, mps)",
        "job_ids": [f"d{RNG.getrandbits(64):016x}"[:20]] if spec["mode"] == "qpu" else [],
        "qpu_seconds": qpu_seconds, "wall_seconds": wall_seconds,
    }
    if error:
        base.update({"fidelity": None, "ci_low": None, "ci_high": None, "populations": None,
                     "coherence": None, "parity_curve": None, "certified": False, "error": error})
        return base
    gap = 0.07 + 0.04 * (n / 50)
    pops = min(0.99, fidelity + gap)
    coh = max(0.0, 2 * fidelity - pops)
    lo, hi = fidelity - half_width, fidelity + half_width
    base.update({
        "fidelity": round(fidelity, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
        "populations": round(pops, 4), "coherence": round(coh, 4),
        "parity_curve": parity_curve(n, coh), "certified": lo > 0.5, "error": None,
    })
    return base


def spec_of(n, mode, strategy="weighted_tree", variant="tree", root=None, exclude=(), shots=2000, hyp=""):
    return {
        "n_qubits": n, "layout_strategy": strategy, "root": root, "exclude_qubits": list(exclude),
        "circuit_variant": variant, "dynamical_decoupling": True, "measure_twirling": True,
        "readout_mitigation": True, "shots": shots, "mode": mode, "hypothesis": hyp,
    }


# ---------------------------------------------------------------- script
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--run-id", default="demo-20261107-221400-qpu")
    args = ap.parse_args()

    run_dir = Path(args.runs_dir) / args.run_id
    if run_dir.exists():
        if not run_dir.name.startswith("demo-"):
            raise SystemExit("refusing to overwrite a non-demo run")
        shutil.rmtree(run_dir)
    (run_dir / "experiments").mkdir(parents=True)

    t0 = datetime(2026, 11, 8, 3, 14, 0, tzinfo=timezone.utc)  # 22:14 US-Eastern, Nov 7
    clock = [t0]
    cal = build_calibration()
    cal["timestamp"] = t0.isoformat()
    (run_dir / "calibration.json").write_text(json.dumps(cal, indent=1))

    errs = [e["error"] for e in cal["edges"] if e["error"] is not None]
    med_cz = statistics.median(errs)
    worst = sorted(cal["edges"], key=lambda e: -(e["error"] or 0))[:3]
    ro = [q["readout_error"] for q in cal["qubits"] if q["readout_error"]]

    events: list[dict] = []
    budget = {"llm_calls": 0, "qpu_seconds": 0.0, "max_llm_calls": 150, "max_qpu_seconds": 600.0}

    def emit(type_: str, dt: float = 5.0, **data):
        clock[0] += timedelta(seconds=dt)
        events.append({"ts": clock[0].isoformat(timespec="seconds"), "seq": len(events),
                       "type": type_, "data": data})

    def think(text: str, dt: float = 9.0):
        budget["llm_calls"] += 1
        emit("thought", dt, text=text)

    def emit_budget():
        emit("budget", 1, **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in budget.items()})

    counter = [0]

    def experiment(spec, layout, fidelity=None, hw=0.028, qpu_s=0.0, error=None, wall=None):
        counter[0] += 1
        exp_id = f"e{counter[0]:03d}"
        budget["llm_calls"] += 1  # the tool-call decision itself
        emit("experiment_started", 6, exp_id=exp_id, spec=spec)
        n = spec["n_qubits"]
        if wall is None:
            wall = (8 + 0.9 * n + RNG.uniform(-2, 4)) if spec["mode"] == "sim" else (240 + RNG.uniform(0, 900))
        if spec["mode"] == "qpu":
            wall = wall * 4.3  # includes time waiting in the IBM fair-share queue overnight
        res = make_result(exp_id, spec, layout, fidelity, hw, qpu_s, round(wall, 1), error)
        (run_dir / "experiments" / f"{exp_id}.json").write_text(json.dumps(res, indent=1))
        budget["qpu_seconds"] += qpu_s
        emit("experiment_result", wall, exp_id=exp_id, n_qubits=n, mode=spec["mode"],
             fidelity=res["fidelity"], ci_low=res["ci_low"], ci_high=res["ci_high"],
             certified=res["certified"], physical_qubits=res["physical_qubits"],
             tree_edges=res["tree_edges"], root=res["root"], qpu_seconds=qpu_s, error=error)
        emit_budget()
        return res

    def milestone(text, n, f):
        emit("milestone", 2, text=text, n_qubits=n, fidelity=f)

    C = 67    # central root on the chip
    R2 = 107  # second, disjoint region for replication

    emit("run_started", 0, run_id=args.run_id, mode="qpu", backend="ibm_fez",
         goal="Certify genuine multipartite entanglement (GHZ fidelity > 0.5) on as many physical "
              "qubits of ibm_fez as possible, within 10 minutes of QPU time.",
         max_llm_calls=budget["max_llm_calls"], max_qpu_seconds=budget["max_qpu_seconds"])
    emit_budget()
    think(f"Calibration snapshot: 156 qubits, {len(cal['edges'])} CZ couplers, median CZ error "
          f"{med_cz:.1e}, median readout error {statistics.median(ro):.1e}. The tail matters more than the "
          f"median: {sum(1 for e in errs if e >= 0.5)} couplers (e.g. {worst[0]['a']}–{worst[0]['b']}, "
          f"{worst[2]['a']}–{worst[2]['b']}) are reported dead and must stay out of any large state.")
    think("Strategy: QPU seconds are the scarce resource, simulation is nearly free. I'll map the "
          "fidelity-vs-N curve in a noisy simulator first, then spend real hardware time only on sizes "
          "the simulator says can clear 0.5.")
    think("Start small to validate the whole pipeline end to end: an 8-qubit linear chain on the "
          "lowest-error path through the chip's centre.")
    r = experiment(spec_of(8, "sim", "best_chain", "linear", root=C,
                           hyp="Pipeline check: an 8-qubit chain should land well above 0.9."),
                   grow_chain(cal, 8, C), 0.931, 0.022)
    milestone("First certified GHZ state (simulation): 8 qubits", 8, r["fidelity"])
    think("Pipeline validated: F = 0.93, populations and parity coherence agree. A chain's depth grows "
          "linearly with N; a tree rooted in the middle of the chip grows only logarithmically. "
          f"Switching to weighted trees rooted at qubit {C}.")
    experiment(spec_of(12, "sim", hyp="A shallow tree at N=12 should lose < 5 points vs N=8."),
               grow_tree(cal, 12, C), 0.893, 0.024)
    r16 = experiment(spec_of(16, "sim", hyp="Baseline the tree at N=16 for a head-to-head with a chain."),
                     grow_tree(cal, 16, C), 0.846, 0.025)
    experiment(spec_of(16, "sim", "best_chain", "linear", root=C,
                       hyp="Same N as a chain. Prediction: noticeably worse because of the extra depth."),
               grow_chain(cal, 16, C), 0.772, 0.027)
    think(f"Confirmed: the tree beats the chain by 7 points at N=16 (depth {r16['circuit_depth']} vs 11). "
          "Idle decoherence during the CNOT cascade dominates. Trees from here on.")
    experiment(spec_of(20, "sim", hyp="Continue the scan in steps of 4."), grow_tree(cal, 20, C), 0.792, 0.026)
    r = experiment(spec_of(24, "sim", hyp="Expect ~0.74; still comfortably certified."),
                   grow_tree(cal, 24, C), 0.741, 0.027)
    milestone("Simulation record: 24-qubit GHZ certified", 24, r["fidelity"])
    think("The slope is steady at about −1.2 points per qubit. Readout error is the largest single "
          "term at this size, so readout mitigation and measurement twirling stay on for every run.")
    experiment(spec_of(28, "sim", hyp="Linear extrapolation says 0.69."), grow_tree(cal, 28, C), 0.688, 0.028)
    experiment(spec_of(32, "sim", hyp="Linear extrapolation says 0.64."), grow_tree(cal, 32, C), 0.637, 0.028)
    experiment(spec_of(36, "sim", "bfs_tree", hyp="Ablation: does error-weighting the tree actually matter?"),
               grow_tree(cal, 36, C, weighted=False), 0.553, 0.029)
    experiment(spec_of(36, "sim", hyp="Same N with error-weighted growth."), grow_tree(cal, 36, C), 0.601, 0.029)
    think("Ablation result: the plain BFS tree is 5 points worse than the error-weighted tree at the same "
          "size. Choosing which couplers to use matters as much as how many qubits we use.")
    r = experiment(spec_of(40, "sim", hyp="Should still certify, with a thin margin."),
                   grow_tree(cal, 40, C), 0.562, 0.029)
    milestone("Simulation record: 40-qubit GHZ certified", 40, r["fidelity"])
    experiment(spec_of(44, "sim", hyp="Right at the edge of the threshold."), grow_tree(cal, 44, C), 0.519, 0.030)
    bad44 = [worst[0]["a"], worst[0]["b"], worst[1]["b"]]
    think("N=44 straddles the threshold (CI lower bound 0.489). Before giving up on it, I'll exclude the "
          f"qubits on the worst couplers ({', '.join(map(str, bad44))}) and let the tree route around them.")
    experiment(spec_of(44, "sim", exclude=bad44, hyp="Routing around the bad couplers buys ~1–2 points."),
               grow_tree(cal, 44, C, exclude=bad44), 0.536, 0.029)
    experiment(spec_of(48, "sim", exclude=bad44, hyp="Probably past the frontier, but this pins it down."),
               grow_tree(cal, 48, C, exclude=bad44), 0.472, 0.031)
    think("Simulated frontier is about 44 qubits. Real hardware usually lands 5–10 points below its own "
          "noise model (drift, crosstalk). With 600 QPU-seconds and about 40–120 s per run, the plan is "
          "to confirm at 16, then step up while the measured offset still predicts > 0.5.", 8)

    r = experiment(spec_of(16, "qpu", hyp="Hardware anchor point. Sim says 0.85; I predict 0.75–0.80."),
                   grow_tree(cal, 16, C), 0.744, 0.026, qpu_s=38.4, wall=412)
    milestone("First certified GHZ state on real IBM hardware: 16 qubits", 16, r["fidelity"])
    r = experiment(spec_of(24, "qpu", hyp="If the offset holds at ~10 points, expect ~0.64."),
                   grow_tree(cal, 24, C), 0.633, 0.027, qpu_s=51.7, wall=538)
    milestone("Real hardware: 24-qubit GHZ certified", 24, r["fidelity"])
    think("Hardware sits about 11 points below simulation, and the gap is roughly constant. That predicts "
          "N=32 at about 0.53. Close, but worth the 70 seconds.")
    experiment(spec_of(32, "qpu", hyp="Predicted 0.53 ± 0.03."), grow_tree(cal, 32, C), None, qpu_s=0.0,
               error="IBMBackendError: backend ibm_fez entered a maintenance window; job cancelled before "
                     "execution (no QPU time charged).", wall=96)
    emit("error", 1, text="QPU job for e017 cancelled: ibm_fez entered maintenance. No QPU time charged.")
    drift = [89]
    think("The job was cancelled before running, so no budget was lost. After maintenance, a fresh "
          "calibration shows coupler 78–89 drifted to 1.9e-2. I'll exclude qubit 89, re-check in "
          "simulation, then resubmit.", 14)
    experiment(spec_of(32, "sim", exclude=drift, hyp="Re-verify the 32-qubit tree without qubit 89."),
               grow_tree(cal, 32, C, exclude=drift), 0.641, 0.028)
    r = experiment(spec_of(32, "qpu", exclude=drift, hyp="Resubmission. Predicted 0.53–0.56."),
                   grow_tree(cal, 32, C, exclude=drift), 0.556, 0.028, qpu_s=70.9, wall=655)
    milestone("Record on ibm_fez: 32-qubit GHZ certified", 32, r["fidelity"])
    think("Certified: F = 0.556, 95% CI [0.528, 0.584]. N=36 is predicted at ~0.52, which is a coin flip "
          "at 2000 shots. I'll double the shots to 4000 to narrow the interval and try once.")
    experiment(spec_of(36, "qpu", exclude=drift, shots=4000, hyp="Predicted 0.51–0.53: a genuine coin flip."),
               grow_tree(cal, 36, C, exclude=drift), 0.521, 0.024, qpu_s=118.2, wall=903)
    think("Inconclusive: the point estimate is above 0.5 but the CI lower bound is 0.497, so I can't claim "
          "it. Before spending more, I want to know whether 32 was a lucky patch. Replicating on a "
          f"disjoint region rooted at qubit {R2}.")
    experiment(spec_of(32, "sim", root=R2, exclude=drift, hyp="Replication region, simulated first."),
               grow_tree(cal, 32, R2, exclude=drift), 0.612, 0.028)
    experiment(spec_of(32, "qpu", root=R2, exclude=drift, hyp="If 32 is real, this region certifies too."),
               grow_tree(cal, 32, R2, exclude=drift), 0.541, 0.029, qpu_s=72.4, wall=611)
    think("Replicated: two independent 32-qubit regions both certify, so the result is not a lucky patch. "
          "With 249 s left, one more attempt fits: N=34, a smaller step than 36, at 4000 shots.")
    r = experiment(spec_of(34, "qpu", exclude=drift, shots=4000, hyp="Predicted 0.53; the 4000-shot CI "
                                                                     "should clear 0.5."),
                   grow_tree(cal, 34, C, exclude=drift), 0.529, 0.023, qpu_s=120.3, wall=884)
    milestone("New record: 34-qubit GHZ certified on real hardware", 34, r["fidelity"])
    experiment(spec_of(52, "sim", exclude=bad44, hyp="Complete the simulated curve past the frontier for the paper."),
               grow_tree(cal, 52, C, exclude=bad44), 0.431, 0.031)
    think("128 QPU-seconds remain. A 36-qubit retry at 4000 shots is estimated at ~135 s, which exceeds "
          "the budget, and the budget is a hard cap. Stopping here with a clean, replicated result.")
    emit("run_finished", 4, best_certified_n=34, best_exp_id=f"e{counter[0] - 1:03d}",
         summary="Certified genuine 34-qubit GHZ entanglement on ibm_fez (F = 0.529, 95% CI [0.506, "
                 "0.552]) and replicated a 32-qubit result on two disjoint regions of the chip. Error-weighted "
                 "trees beat chains by 7 points and beat unweighted trees by 5. Hardware ran ~11 points "
                 "below the noise model. Used 472 of 600 QPU-seconds.")

    with (run_dir / "events.jsonl").open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    budget["qpu_seconds"] = round(budget["qpu_seconds"], 1)
    (run_dir / "budget.json").write_text(json.dumps(budget, indent=2))
    n_exp = counter[0]
    print(f"wrote {run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir}: "
          f"{len(events)} events, {n_exp} experiments, qpu {budget['qpu_seconds']}s, llm {budget['llm_calls']}")


if __name__ == "__main__":
    main()

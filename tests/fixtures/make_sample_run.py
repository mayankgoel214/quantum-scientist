"""Regenerate tests/fixtures/sample_run/ — a realistic, hand-scripted finished run.

    uv run python -m tests.fixtures.make_sample_run

Calibration comes from qiskit's FakeFez (156-qubit Heron r2). The experiment story is scripted
(fidelities chosen by hand) but every derived quantity — parity curves, coherence, fidelity, CIs,
tree layouts on the real coupling map — is computed consistently so the paper code sees data
shaped exactly like CONTRACTS.md.
"""
from __future__ import annotations

import heapq
import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "sample_run"
RUN_ID = "20261107-221500-qpu"
T0 = datetime(2026, 11, 7, 22, 15, 0, tzinfo=timezone.utc)
rng = np.random.default_rng(7)


def heavy_hex_xy(q: int) -> tuple[float, float]:
    """Heron r2: 8 rows of 16 qubits (ids r*20..r*20+15), 4 bridge qubits between rows."""
    r, k = divmod(q, 20)
    if k < 16:
        return float(k), float(-2 * r)
    bridge_cols = [3, 7, 11, 15] if r % 2 == 0 else [1, 5, 9, 13]
    return float(bridge_cols[k - 16]), float(-2 * r - 1)


def build_calibration() -> dict:
    from qiskit_ibm_runtime.fake_provider import FakeFez

    b = FakeFez()
    p = b.properties()
    qubits = []
    for q in range(b.num_qubits):
        x, y = heavy_hex_xy(q)

        def safe(fn):
            try:
                v = fn()
                return None if v is None or (isinstance(v, float) and math.isnan(v)) else v
            except Exception:
                return None

        t1 = safe(lambda: p.t1(q))
        t2 = safe(lambda: p.t2(q))
        qubits.append({
            "id": q,
            "t1_us": round(t1 * 1e6, 2) if t1 else None,
            "t2_us": round(t2 * 1e6, 2) if t2 else None,
            "readout_error": round(safe(lambda: p.readout_error(q)) or 0, 5),
            "sx_error": safe(lambda: round(p.gate_error("sx", [q]), 6)),
            "x": x, "y": y,
        })
    seen, edges = set(), []
    for a, c in b.coupling_map.get_edges():
        key = (min(a, c), max(a, c))
        if key in seen:
            continue
        seen.add(key)
        err = None
        for pair in ([a, c], [c, a]):
            try:
                err = p.gate_error("cz", pair)
                break
            except Exception:
                pass
        edges.append({"a": key[0], "b": key[1], "error": round(err, 6) if err is not None else None, "gate": "cz"})
    return {
        "backend": "ibm_fez", "num_qubits": b.num_qubits, "source": "fake",
        "timestamp": T0.isoformat(timespec="seconds"), "qubits": qubits, "edges": edges,
    }


def plan_tree(cal: dict, n: int, root: int, exclude=(), chain=False) -> tuple[list[int], list[list[int]], int]:
    """Prim-style growth weighted by CZ error + readout error (mirrors a 'weighted_tree' layout)."""
    ro = {q["id"]: q["readout_error"] or 0.05 for q in cal["qubits"]}
    adj: dict[int, list[tuple[int, float]]] = {}
    for e in cal["edges"]:
        w = (e["error"] if e["error"] is not None and e["error"] < 0.5 else 1.0)
        adj.setdefault(e["a"], []).append((e["b"], w))
        adj.setdefault(e["b"], []).append((e["a"], w))
    excl = set(exclude)
    inside, edges, depth = {root}, [], {root: 0}
    if chain:
        def dfs(path: list[int], seen: set[int], budget: list[int]) -> list[int] | None:
            if len(path) == n:
                return path
            budget[0] -= 1
            if budget[0] < 0:
                return None
            cands = sorted((w + ro[v], v) for v, w in adj[path[-1]] if v not in seen and v not in excl)
            for _, v in cands:
                seen.add(v)
                got = dfs(path + [v], seen, budget)
                if got:
                    return got
                seen.discard(v)
            return None
        path = dfs([root], {root}, [200000]) or [root]
        for u, v in zip(path, path[1:]):
            edges.append([u, v]); inside.add(v); depth[v] = depth[u] + 1
    else:
        heap = [(w + ro[v] + 0.002 * 0, root, v) for v, w in adj[root]]
        heapq.heapify(heap)
        while heap and len(inside) < n:
            w, u, v = heapq.heappop(heap)
            if v in inside or v in excl:
                continue
            inside.add(v); edges.append([u, v]); depth[v] = depth[u] + 1
            for x, wx in adj[v]:
                if x not in inside:
                    # mild depth penalty keeps the tree bushy (shallower circuit)
                    heapq.heappush(heap, (wx + ro[x] + 0.004 * depth[v], v, x))
    order = [root] + [c for _, c in edges]
    return order, edges, max(depth.values())


# (mode, N, strategy, variant, target_F, ci_half, minutes_after_start, exclude, error, hypothesis)
SCRIPT = [
    ("sim", 8, "bfs_tree", "tree", 0.932, 0.012, 2, [], None, "Baseline: an 8-qubit BFS tree should be trivially certified in simulation."),
    ("sim", 16, "bfs_tree", "tree", 0.861, 0.014, 6, [], None, "Doubling N in simulation to see how fast fidelity decays."),
    ("sim", 24, "weighted_tree", "tree", 0.792, 0.016, 11, [], None, "Error-weighted tree should beat BFS by avoiding the worst CZ links."),
    ("qpu", 8, "weighted_tree", "tree", 0.843, 0.019, 19, [], None, "First hardware shot: confirm the pipeline and readout mitigation on real qubits."),
    ("qpu", 16, "weighted_tree", "tree", 0.712, 0.024, 31, [], None, "16 qubits on hardware; simulation predicts ~0.86, expect a gap."),
    ("sim", 32, "weighted_tree", "tree", 0.703, 0.018, 38, [], None, "Check that 32 is comfortably above threshold in simulation before spending QPU time."),
    ("qpu", 24, "weighted_tree", "tree", 0.618, 0.027, 52, [], None, "24 on hardware with DD and twirling; margin should be ~0.1."),
    ("qpu", 32, "weighted_tree", "tree", 0.493, 0.029, 71, [], None, "Push to 32. Risky: extrapolated decay puts us right at 0.5."),
    ("sim", 40, "weighted_tree", "tree", 0.612, 0.020, 80, [], None, "Simulation headroom at 40, to decide whether a hardware attempt is worth it."),
    ("qpu", 32, "weighted_tree", "tree", 0.552, 0.028, 97, [62, 63, 81], None, "Retry 32 excluding qubits 62, 63 and 81 (high readout error in the snapshot)."),
    ("sim", 48, "weighted_tree", "tree", 0.511, 0.022, 106, [], None, "Find the simulation ceiling: 48 qubits."),
    ("qpu", 40, "weighted_tree", "tree", 0.441, 0.031, 124, [62, 63, 81], None, "Attempt 40 on hardware with the same exclusions."),
    ("qpu", 36, "weighted_tree", "tree", 0.541, 0.030, 146, [62, 63, 81], None, "Bisect between the certified 32 and the failed 40: try 36."),
    ("qpu", 40, "weighted_tree", "tree", None, None, 163, [62, 63, 81], "RuntimeJobFailureError: job cancelled by backend (calibration window)", "Second attempt at 40 after a fresh shallower tree."),
    ("sim", 44, "weighted_tree", "tree", 0.563, 0.021, 171, [], None, "Bracket the simulation record between 40 and 48."),
    ("qpu", 40, "best_chain", "linear", 0.462, 0.032, 189, [62, 63, 81], None, "Last idea for 40: a low-error chain trades depth for better links."),
]

ROOTS = {"sim": 47, "qpu": 47}

THOUGHTS = [
    (0.5, "Goal: certify genuine multipartite entanglement (fidelity lower bound > 0.5) on as many qubits of ibm_fez as possible. I will calibrate expectations in the noisy simulator before touching the QPU budget."),
    (9, "Simulation decays roughly 0.009 per added qubit for BFS trees. Switching to an error-weighted tree rooted near the chip centre (qubit 47) to keep circuit depth logarithmic."),
    (17, "Simulator certifies 24 with a wide margin. Time to validate on hardware at small N first so I can measure the sim-to-hardware gap rather than guess it."),
    (34, "Hardware 16-qubit fidelity is 0.71 versus 0.86 in simulation: a gap of about 0.15 that grows with N. I will only send hardware jobs where the simulator shows at least 0.15 of headroom."),
    (75, "N=32 failed on hardware (F=0.49). The calibration snapshot shows qubits 62, 63 and 81 with readout error above 4%; the weighted tree used two of them. Excluding them and retrying."),
    (101, "Excluding the bad qubits recovered N=32 (F=0.55, CI lower bound above 0.5). New hardware record. Remaining QPU budget suggests two or three more large attempts."),
    (128, "N=40 failed clearly (0.44). The truth lies between 32 and 40; bisecting to 36 is the best use of the remaining budget."),
    (150, "N=36 certified on hardware with a lower bound of about 0.51. This is a narrow margin; I should report it as certified but flag the statistical fragility."),
    (166, "The second N=40 job was cancelled by the backend. Not charging this as evidence either way."),
    (193, "Linear chain at N=40 is also below threshold. The hardware record stands at 36 qubits; the simulation ceiling is 44. Stopping to preserve budget and writing up."),
]


def main() -> None:
    cal = build_calibration()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "experiments").mkdir(parents=True)
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=1))

    events: list[dict] = []

    def emit(t: datetime, type_: str, **data):
        events.append({"ts": t.isoformat(timespec="seconds"), "seq": 0, "type": type_, "data": data})

    max_llm, max_qpu = 300, 480.0
    emit(T0, "run_started", run_id=RUN_ID, mode="qpu", backend="ibm_fez",
         goal="Certify GHZ genuine multipartite entanglement on as many qubits as possible",
         max_llm_calls=max_llm, max_qpu_seconds=max_qpu)
    for minute, text in THOUGHTS:
        emit(T0 + timedelta(minutes=minute), "thought", text=text)

    llm_calls, qpu_s = 1, 0.0
    best_qpu = (0, None)
    for i, (mode, n, strat, variant, F, half, minute, excl, err, hyp) in enumerate(SCRIPT, start=1):
        exp_id = f"e{i:03d}"
        t_start = T0 + timedelta(minutes=minute)
        shots = 2000 if mode == "qpu" else 1000
        spec = {
            "n_qubits": n, "layout_strategy": strat, "root": None, "exclude_qubits": excl,
            "circuit_variant": variant, "dynamical_decoupling": True, "measure_twirling": True,
            "readout_mitigation": True, "shots": shots, "mode": mode, "hypothesis": hyp,
        }
        emit(t_start, "experiment_started", exp_id=exp_id, spec=spec)
        llm_calls += 3
        order, tree, height = plan_tree(cal, n, ROOTS[mode], excl, chain=(variant == "linear"))
        depth = height + 3 if variant == "tree" else n + 2
        wall = round(8 + 0.9 * n + rng.uniform(0, 5), 1) if mode == "sim" else round(60 + 4 * n + rng.uniform(0, 30), 1)
        qsec = round(4.0 + 0.16 * n + rng.uniform(0, 1.5), 2) if mode == "qpu" else 0.0
        res = {
            "exp_id": exp_id, "spec": spec, "physical_qubits": order, "tree_edges": tree,
            "root": ROOTS[mode], "circuit_depth": depth, "two_qubit_gate_count": n - 1,
            "fidelity": None, "ci_low": None, "ci_high": None, "populations": None,
            "coherence": None, "parity_curve": [], "certified": False, "qpu_seconds": qsec,
            "wall_seconds": wall, "backend": "ibm_fez" if mode == "qpu" else "fake_fez (aer, noisy)",
            "job_ids": [f"d4{rng.integers(16**10):010x}g{rng.integers(16**6):06x}"] if mode == "qpu" else [],
            "error": err,
        }
        if err is None:
            pop_target = 1 - (1 - F) * 0.55
            coh_target = 2 * F - pop_target
            phis = [j * math.pi / (n + 1) for j in range(2 * n + 2)]
            delta = rng.uniform(-0.25, 0.25) if mode == "qpu" else 0.0
            sigma = 1.0 / math.sqrt(shots) * 1.2
            par = [coh_target * math.cos(n * p + delta) + rng.normal(0, sigma) for p in phis]
            par = [max(-1.0, min(1.0, v)) for v in par]
            coh = 2 * abs(np.mean(np.exp(1j * n * np.array(phis)) * np.array(par)))
            pop = 2 * F - coh
            fid = (pop + coh) / 2
            res.update({
                "fidelity": round(fid, 4), "ci_low": round(fid - half, 4), "ci_high": round(fid + half, 4),
                "populations": round(pop, 4), "coherence": round(float(coh), 4),
                "parity_curve": [{"phi": round(p, 6), "parity": round(v, 4)} for p, v in zip(phis, par)],
                "certified": bool(fid - half > 0.5),
            })
        if mode == "qpu":
            qpu_s += qsec
        t_end = t_start + timedelta(seconds=wall)
        (OUT / "experiments" / f"{exp_id}.json").write_text(json.dumps(res, indent=1))
        emit(t_end, "experiment_result", exp_id=exp_id, n_qubits=n, mode=mode,
             fidelity=res["fidelity"], ci_low=res["ci_low"], ci_high=res["ci_high"],
             certified=res["certified"], physical_qubits=order, tree_edges=tree,
             root=res["root"], qpu_seconds=qsec, error=err)
        emit(t_end, "budget", llm_calls=llm_calls, max_llm_calls=max_llm,
             qpu_seconds=round(qpu_s, 2), max_qpu_seconds=max_qpu)
        if err:
            emit(t_end, "error", text=err)
        if res["certified"] and mode == "qpu" and n > best_qpu[0]:
            best_qpu = (n, exp_id)
            emit(t_end, "milestone", text=f"New certified hardware record: {n}-qubit GHZ", n_qubits=n, fidelity=res["fidelity"])

    t_fin = T0 + timedelta(minutes=196)
    emit(t_fin, "run_finished", best_certified_n=best_qpu[0], best_exp_id=best_qpu[1],
         summary=f"Certified genuine multipartite entanglement on {best_qpu[0]} qubits of ibm_fez.")
    events.sort(key=lambda e: e["ts"])
    for i, e in enumerate(events):
        e["seq"] = i
    (OUT / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (OUT / "budget.json").write_text(json.dumps({
        "llm_calls": llm_calls, "max_llm_calls": max_llm, "qpu_seconds": round(qpu_s, 3), "max_qpu_seconds": max_qpu,
    }, indent=2))
    print(f"wrote {OUT} (record qpu N={best_qpu[0]})")


if __name__ == "__main__":
    main()

"""Tests for qsci.core. Fast by default; `pytest -m slow` runs the noisy-sim / local-sampler ones."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qsci.core import estimate_qpu_seconds, get_backend, get_calibration, plan_layout, run_experiment
from qsci.core.circuits import build_logical_circuits, phis
from qsci.core.execute import build_qpu_pubs, prepare, sampler_options
from qsci.core.fidelity import analyze, coherence_from_parities


@pytest.fixture(scope="module")
def backend():
    return get_backend("sim")


@pytest.fixture(scope="module")
def cal(backend):
    return get_calibration(backend)


def _ideal_counts(circuits, shots=4000, seed=3):
    sim = AerSimulator(method="matrix_product_state")
    r = sim.run(circuits, shots=shots, seed_simulator=seed).result()
    return [r.get_counts(i) for i in range(len(circuits))]


# ------------------------------------------------------------------ calibration
def test_calibration_schema(cal):
    assert cal["backend"] == "fake_fez" and cal["num_qubits"] == 156 and cal["source"] == "fake"
    assert len(cal["qubits"]) == 156 and len(cal["edges"]) == 176
    q = cal["qubits"][0]
    assert set(q) == {"id", "t1_us", "t2_us", "readout_error", "sx_error", "x", "y"}
    assert len({(q["x"], q["y"]) for q in cal["qubits"]}) == 156  # no overlapping chip-map points
    assert all(e["gate"] == "cz" and 0 <= e["error"] <= 1 for e in cal["edges"])
    json.dumps(cal)


def test_coordinates_heavy_hex_other_devices():
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    c = get_calibration(FakeTorino())
    assert len({(q["x"], q["y"]) for q in c["qubits"]}) == 133
    # coupled qubits are drawn next to each other
    pos = {q["id"]: (q["x"], q["y"]) for q in c["qubits"]}
    assert all(math.dist(pos[e["a"]], pos[e["b"]]) <= 1.01 for e in c["edges"])


def test_calibration_missing_values():
    from qsci.core.layout import plan_layout as pl

    cal = {
        "qubits": [{"id": i, "t1_us": None, "t2_us": None, "readout_error": None, "sx_error": None, "x": None, "y": None} for i in range(5)],
        "edges": [{"a": i, "b": i + 1, "error": None, "gate": "cz"} for i in range(4)],
    }
    p = pl(cal, 5, "weighted_tree")
    assert sorted(p["qubits"]) == [0, 1, 2, 3, 4]


# ------------------------------------------------------------------ layout
@pytest.mark.parametrize("strategy", ["weighted_tree", "bfs_tree", "best_chain"])
@pytest.mark.parametrize("n", [2, 8, 24, 40])
def test_layout_is_connected_tree(cal, strategy, n):
    edges = {(e["a"], e["b"]) for e in cal["edges"]} | {(e["b"], e["a"]) for e in cal["edges"]}
    p = plan_layout(cal, n, strategy)
    assert len(p["qubits"]) == n == len(set(p["qubits"]))
    assert len(p["tree_edges"]) == n - 1
    assert p["qubits"][0] == p["root"]
    reached = {p["root"]}
    for a, b in p["tree_edges"]:
        assert (a, b) in edges and a in reached and b not in reached
        reached.add(b)
    assert p["depth"] >= math.ceil(math.log2(n))


def test_layout_exclude_and_root(cal):
    p = plan_layout(cal, 16, "weighted_tree", root=64, exclude=[65, 66])
    assert p["root"] == 64 and not {65, 66} & set(p["qubits"])
    c = plan_layout(cal, 10, "best_chain", exclude=list(range(0, 40)))
    assert min(c["qubits"]) >= 40
    with pytest.raises(ValueError):
        plan_layout(cal, 8, "weighted_tree", root=5, exclude=[5])


def test_tree_shallower_than_chain(cal):
    t = plan_layout(cal, 32, "weighted_tree")
    c = plan_layout(cal, 32, "best_chain")
    assert t["depth"] < c["depth"] == 16


# ------------------------------------------------------------------ fidelity math
def test_coherence_formula_ideal_any_phase():
    for n in (3, 8, 17):
        for theta in (0.0, 0.7, 2.0):
            par = [math.cos(n * f + theta) for f in phis(n)]
            assert coherence_from_parities(par, n) == pytest.approx(1.0, abs=1e-9)
        # the contract's (-1)^j form would give 0 here -- documented deviation
        par = [math.cos(n * f) for f in phis(n)]
        assert abs(sum((-1) ** j * p for j, p in enumerate(par))) < 1e-9


@pytest.mark.parametrize("n", [4, 8, 16])
def test_ideal_ghz_fidelity_is_one(cal, n):
    p = plan_layout(cal, n, "weighted_tree")
    L = build_logical_circuits(p, "tree")
    par = [L["parity"].assign_parameters({L["phi"]: f}) for f in phis(n)]
    counts = _ideal_counts([L["populations"]] + par + [L["cal0"], L["cal1"]])
    a = analyze(n, counts[0], counts[1 : 2 * n + 3])
    assert a["fidelity"] > 0.97 and a["ci_high"] >= 0.99 and a["certified"]
    # ideal readout calibration -> mitigation is the identity
    m = analyze(n, counts[0], counts[1 : 2 * n + 3], counts[-2], counts[-1], mitigate=True)
    assert m["fidelity"] == pytest.approx(a["fidelity"], abs=1e-9)


@pytest.mark.parametrize("n", [4, 8])
def test_product_and_mixed_states_not_certified(n):
    # product state |+>^n and the classical mixture of |0..0>,|1..1> (dephased GHZ)
    for kind in ("product", "mixed"):
        circs = []
        for f in [None] + phis(n):
            qc = QuantumCircuit(n, n)
            if kind == "product":
                qc.h(range(n))
            else:
                qc.h(0)
                qc.measure(0, 0)  # collapses the superposition -> mixture
                qc.reset(range(1, n))
                with qc.if_test((qc.clbits[0], 1)):
                    qc.x(range(1, n))
            if f is not None:
                qc.rz(f, range(n))
                qc.h(range(n))
            qc.measure(range(n), range(n))
            circs.append(qc)
        sim = AerSimulator()
        r = sim.run(circs, shots=4000, seed_simulator=5).result()
        cs = [r.get_counts(i) for i in range(len(circs))]
        a = analyze(n, cs[0], cs[1:])
        assert a["fidelity"] <= 0.55 and not a["certified"], (kind, a["fidelity"])


def test_readout_mitigation_recovers_flips():
    n = 3
    rng = np.random.default_rng(0)

    def noisy(bitstring_p, e0=0.05, e1=0.1, shots=200000):
        out = {}
        states = list(bitstring_p)
        idx = rng.choice(len(states), size=shots, p=list(bitstring_p.values()))
        for i in idx:
            s = [int(c) for c in states[i][::-1]]
            m = [(1 - b if rng.random() < (e0 if b == 0 else e1) else b) for b in s]
            k = "".join(str(b) for b in m[::-1])
            out[k] = out.get(k, 0) + 1
        return out

    rng = np.random.default_rng(1)
    pops = noisy({"000": 0.5, "111": 0.5}, shots=20000)
    cal0 = noisy({"000": 1.0}, shots=20000)
    cal1 = noisy({"111": 1.0}, shots=20000)
    par = [{"000": 1000} if (j % 2 == 0) else {"001": 1000} for j in range(2 * n + 2)]
    raw = analyze(n, pops, par)
    mit = analyze(n, pops, par, cal0, cal1, mitigate=True)
    assert raw["populations"] < 0.85
    assert mit["populations"] == pytest.approx(1.0, abs=0.03)


# ------------------------------------------------------------------ QPU path (dry run, no submit)
def test_qpu_dry_run_pubs_and_placement(backend, cal):
    spec = {"n_qubits": 12, "mode": "qpu", "readout_mitigation": True}
    prep = prepare(spec, backend, cal, scheduled=False)
    pubs, _ = build_qpu_pubs(prep)
    assert len(pubs) == 2 * 12 + 3 + 2  # populations + 2N+2 parity + 2 readout calibrations
    phys = prep["plan"]["qubits"]
    for circ, params, shots in pubs:
        assert params is None and shots > 0
        assert not circ.parameters  # fully bound
        assert set(circ.count_ops()) <= {"rz", "sx", "x", "cz", "measure", "barrier"}
        final = circ.layout.final_index_layout()[:12]
        assert list(final) == phys
        used = {circ.find_bit(q).index for inst in circ.data for q in inst.qubits if inst.operation.name != "barrier"}
        assert used <= set(phys)
        assert circ.cregs[0].name == "c" and circ.num_clbits == 12
    assert prep["two_qubit_gate_count"] == 11
    assert prep["circuit_depth"] == prep["plan"]["depth"]
    o = sampler_options({**spec, "dynamical_decoupling": True, "measure_twirling": True})
    assert o["dynamical_decoupling"]["enable"] and o["twirling"]["enable_measure"]


def test_estimate_qpu_seconds_monotone_and_conservative():
    a = estimate_qpu_seconds({"n_qubits": 8, "shots": 2000})
    b = estimate_qpu_seconds({"n_qubits": 32, "shots": 2000})
    assert 0 < a < b
    # >= 250us per shot (Heron default rep delay) for every shot submitted
    total_shots = 2000 * (1 + 66) + 2 * 1000
    assert b >= total_shots * 2.5e-4


def test_run_experiment_never_raises(backend, cal):
    r = run_experiment({"n_qubits": 500, "mode": "sim"}, backend, cal, "bad")
    assert r["error"] and r["fidelity"] is None and r["exp_id"] == "bad"
    r = run_experiment({"n_qubits": 4, "layout_strategy": "nope"}, backend, cal, "bad2")
    assert "layout_strategy" in r["error"]


# ------------------------------------------------------------------ slow
@pytest.mark.slow
def test_noisy_sim_small(backend, cal):
    r = run_experiment({"n_qubits": 8, "shots": 1000, "mode": "sim"}, backend, cal, "e1")
    assert r["error"] is None, r.get("traceback")
    assert 0.6 < r["fidelity"] <= 1.0 and r["certified"]
    assert r["qpu_seconds"] == 0 and len(r["parity_curve"]) == 18
    json.dumps(r)


def test_linear_variant_is_sequential(backend, cal):
    prep = prepare({"n_qubits": 10, "circuit_variant": "linear"}, backend, cal)
    assert prep["circuit_depth"] == 9
    prep = prepare({"n_qubits": 10, "circuit_variant": "tree"}, backend, cal)
    assert prep["circuit_depth"] == prep["plan"]["depth"] < 9


def test_frame_sim_fast_and_sane(backend, cal):
    import time

    t = time.time()
    r = run_experiment({"n_qubits": 24, "shots": 2000, "mode": "sim"}, backend, cal, "f24")
    assert r["error"] is None, r.get("traceback")
    assert time.time() - t < 30
    assert 0.6 < r["fidelity"] < 0.98 and r["ci_low"] <= r["fidelity"] <= r["ci_high"]
    lin = run_experiment({"n_qubits": 24, "shots": 2000, "circuit_variant": "linear"}, backend, cal, "l24")
    assert lin["fidelity"] < r["fidelity"]  # idle dephasing punishes the deep schedule


@pytest.mark.slow
def test_frame_sim_agrees_with_aer_mps(backend, cal):
    spec = {"n_qubits": 8, "shots": 2000, "mode": "sim", "root": 136}
    a = run_experiment({**spec, "sim_engine": "aer_mps"}, backend, cal, "a")
    f = run_experiment({**spec, "sim_engine": "pauli_frame"}, backend, cal, "f")
    assert a["error"] is None and f["error"] is None
    assert abs(a["fidelity"] - f["fidelity"]) < 0.05


@pytest.mark.slow
def test_qpu_code_path_locally_on_fake_backend(backend, cal):
    """Runs the real run_qpu code (SamplerV2 job mode) in qiskit-ibm-runtime local mode."""
    r = run_experiment({"n_qubits": 4, "shots": 500, "mode": "qpu"}, backend, cal, "q1")
    assert r["error"] is None, r.get("traceback")
    assert r["fidelity"] > 0.6 and len(r["job_ids"]) == 1

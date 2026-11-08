"""Running a GHZ experiment end to end (sim via Aer MPS, or QPU via one SamplerV2 job)."""
from __future__ import annotations

import time
import traceback

from .circuits import build_logical_circuits, compact, phis, transpile_on_layout, two_qubit_stats
from .fidelity import analyze
from .layout import plan_layout

DEFAULT_SPEC = {
    "n_qubits": 8,
    "layout_strategy": "weighted_tree",
    "root": None,
    "exclude_qubits": [],
    "circuit_variant": "tree",
    "dynamical_decoupling": True,
    "measure_twirling": True,
    "readout_mitigation": True,
    "shots": 2000,
    "mode": "sim",
    "hypothesis": "",
    "sim_engine": "pauli_frame",  # sim only: "pauli_frame" (fast, default) | "aer_mps" (slow reference)
}
MAX_SHOTS = 20000
SIM_MAX_PARITY_SHOTS = 1000  # sim-only speed cap for each of the 2N+2 parity circuits (see run_sim)
CAL_SHOTS_CAP = 1000

# --- QPU time model (deliberately pessimistic) -----------------------------------------------
QPU_FIXED_S = 8.0  # job overhead billed as quantum time (compile/load), generous
QPU_PER_SHOT_S = 4.0e-4  # Heron rep_delay ~250 us + circuit + readout; 400 us is conservative
QPU_TWIRL_FACTOR = 1.25  # measurement twirling -> extra randomized circuit loads


def normalize_spec(spec: dict) -> dict:
    s = dict(DEFAULT_SPEC)
    s.update({k: v for k, v in (spec or {}).items() if v is not None or k == "root"})
    s["n_qubits"] = int(s["n_qubits"])
    s["shots"] = max(100, min(int(s["shots"]), MAX_SHOTS))
    s["exclude_qubits"] = [int(q) for q in (s.get("exclude_qubits") or [])]
    s["root"] = None if s.get("root") in (None, "", "null") else int(s["root"])
    for k in ("dynamical_decoupling", "measure_twirling", "readout_mitigation"):
        s[k] = bool(s[k])
    return s


def _shot_plan(s: dict) -> dict:
    n, shots = s["n_qubits"], s["shots"]
    cal = min(shots, CAL_SHOTS_CAP) if s["readout_mitigation"] else 0
    return {"populations": shots, "parity": shots, "cal": cal, "n_parity": 2 * n + 2}


def estimate_qpu_seconds(spec: dict) -> float:
    """Conservative QPU-seconds estimate for one experiment on the QPU path.
    total shots = shots * (1 + (2N+2)) + 2 * cal_shots; billed ~ fixed + shots * 400us (x1.25 twirl)."""
    s = normalize_spec(spec)
    p = _shot_plan(s)
    total = p["populations"] + p["n_parity"] * p["parity"] + 2 * p["cal"]
    est = QPU_FIXED_S + total * QPU_PER_SHOT_S
    if s["measure_twirling"]:
        est *= QPU_TWIRL_FACTOR
    return round(est, 2)


# ------------------------------------------------------------------------------------------
def prepare(spec: dict, backend, calibration: dict, scheduled: bool = False) -> dict:
    """Plan the layout, build and transpile every circuit (exactly as submitted on QPU).
    Returns the plan, transpiled circuits (populations, parity_0..parity_{2N+1}, [cal0, cal1])."""
    s = normalize_spec(spec)
    n = s["n_qubits"]
    plan = plan_layout(calibration, n, s["layout_strategy"], s["root"], s["exclude_qubits"])
    logical = build_logical_circuits(plan, s["circuit_variant"])
    templates = [logical["populations"], logical["parity"]]
    if s["readout_mitigation"]:
        templates += [logical["cal0"], logical["cal1"]]
    tc = transpile_on_layout(templates, backend, plan["qubits"], scheduled=scheduled)
    pops_t, par_t = tc[0], tc[1]
    phi = next(iter(par_t.parameters))
    parity = []
    for j, f in enumerate(phis(n)):
        c = par_t.assign_parameters({phi: f}, inplace=False)
        c.name = f"parity_{j}"
        parity.append(c)
    cals = tc[2:4] if s["readout_mitigation"] else []
    count2q, depth2q = two_qubit_stats(pops_t)
    return {
        "spec": s,
        "plan": plan,
        "populations": pops_t,
        "parity": parity,
        "cal": cals,
        "two_qubit_gate_count": count2q,
        "circuit_depth": depth2q,
        "transpiled_depth": pops_t.depth(),
    }


def build_qpu_pubs(prep: dict) -> tuple[list, int]:
    """PUB list for the single SamplerV2 job: [populations, parity_0..2N+1, cal0, cal1]."""
    p = _shot_plan(prep["spec"])
    pubs = [(prep["populations"], None, p["populations"])]
    pubs += [(c, None, p["parity"]) for c in prep["parity"]]
    pubs += [(c, None, p["cal"]) for c in prep["cal"]]
    return pubs, p["populations"]


def sampler_options(spec: dict) -> dict:
    s = normalize_spec(spec)
    opts = {"default_shots": s["shots"]}
    if s["dynamical_decoupling"]:
        opts["dynamical_decoupling"] = {"enable": True, "sequence_type": "XpXm"}
    if s["measure_twirling"]:
        opts["twirling"] = {"enable_gates": False, "enable_measure": True}
    return opts


# ------------------------------------------------------------------------------------------ sim
_NOISE_CACHE: dict = {}


def _device_errors(backend):
    key = (backend.name, id(backend))
    if key not in _NOISE_CACHE:
        from qiskit_aer.noise.device.models import basic_device_gate_errors, basic_device_readout_errors

        gate = basic_device_gate_errors(target=backend.target)
        ro = basic_device_readout_errors(target=backend.target)
        gate_map = {}
        for name, qargs, err in gate:
            gate_map[(name, tuple(qargs))] = err
        ro_map = {tuple(q)[0]: err for q, err in ro}
        _NOISE_CACHE[key] = (gate_map, ro_map)
    return _NOISE_CACHE[key]


def compact_noise_model(backend, physical: list[int]):
    """The fake backend's own noise model (depolarizing + thermal relaxation per gate, readout
    error) restricted to the active physical qubits and re-indexed to compact 0..n-1."""
    from qiskit_aer.noise import NoiseModel

    gate_map, ro_map = _device_errors(backend)
    pos = {p: i for i, p in enumerate(physical)}
    nm = NoiseModel(basis_gates=["id", "rz", "sx", "x", "cz", "reset", "delay", "measure"])
    for (name, qargs), err in gate_map.items():
        if all(q in pos for q in qargs):
            nm.add_quantum_error(err, name, [pos[q] for q in qargs])
    for q, err in ro_map.items():
        if q in pos:
            nm.add_readout_error(err, [pos[q]])
    return nm


def _idle_relaxation(circuits, backend, physical):
    """Insert T1/T2 thermal relaxation on the scheduled idle periods (delays) of active qubits."""
    from qiskit.circuit import Delay
    from qiskit_aer.noise import RelaxationNoisePass

    qp = backend.target.qubit_properties
    t1s = [qp[p].t1 if qp[p] is not None and qp[p].t1 else 1e-4 for p in physical]
    t2s = [min(qp[p].t2, 2 * t1s[i]) if qp[p] is not None and qp[p].t2 else 1e-4 for i, p in enumerate(physical)]
    rp = RelaxationNoisePass(t1s=t1s, t2s=t2s, dt=backend.dt, op_types=[Delay])
    return [rp(c) for c in circuits]


def run_sim(prep: dict, backend, calibration: dict | None = None) -> tuple[dict, dict]:
    """Dispatch to the configured simulator engine. Returns (counts by role, notes)."""
    s = prep["spec"]
    if s.get("sim_engine", "pauli_frame") == "aer_mps":
        return run_sim_aer(prep, backend)
    from .frame_sim import simulate_counts

    p = _shot_plan(s)
    seed = _seed(s)
    counts = simulate_counts(
        prep["plan"], s["circuit_variant"], calibration, backend, p["n_parity"],
        p["populations"], p["parity"], p["cal"], seed=seed,
    )
    notes = {
        "simulator": "pauli_frame: exact Pauli-frame propagation through the CNOT tree + exact sampling of the "
        "Rz/H measurement, noise from calibration (depolarizing gates, Pauli-twirled idle T1/T2, readout)",
        "simulated_qubits": len(prep["plan"]["qubits"]),
        "seed": seed,
        "shots": {"populations": p["populations"], "parity_each": p["parity"], "cal_each": p["cal"]},
        "dynamical_decoupling": "not applied on sim (no-op)" if s["dynamical_decoupling"] else "off",
        "measure_twirling": "not applied on sim (no-op)" if s["measure_twirling"] else "off",
    }
    return counts, notes


def _seed(s: dict) -> int:
    import secrets

    return int(s["seed"]) if s.get("seed") is not None else secrets.randbits(32)


def run_sim_aer(prep: dict, backend) -> tuple[dict, dict]:
    """Reference engine: noisy Aer MPS simulation of the compacted, scheduled circuits.
    Accurate but slow (~20 ms/shot at N=40); parity circuits capped at SIM_MAX_PARITY_SHOTS."""
    from qiskit_aer import AerSimulator

    s = prep["spec"]
    physical = prep["plan"]["qubits"]
    n = len(physical)
    p = _shot_plan(s)
    parity_shots = min(p["parity"], SIM_MAX_PARITY_SHOTS)
    circs = [prep["populations"]] + prep["parity"] + list(prep["cal"])
    comp = [compact(c, physical) for c in circs]
    comp = _idle_relaxation(comp, backend, physical)
    nm = compact_noise_model(backend, physical)
    sim = AerSimulator(method="matrix_product_state", noise_model=nm, max_parallel_experiments=0)
    pops_job = sim.run(comp[:1], shots=p["populations"], seed_simulator=11)
    par_job = sim.run(comp[1 : 1 + p["n_parity"]], shots=parity_shots, seed_simulator=12)
    cal_res = None
    if prep["cal"]:
        cal_res = sim.run(comp[1 + p["n_parity"] :], shots=p["cal"], seed_simulator=13).result()
    pr, rr = pops_job.result(), par_job.result()
    counts = {
        "populations": pr.get_counts(0),
        "parity": [rr.get_counts(i) for i in range(p["n_parity"])],
        "cal0": cal_res.get_counts(0) if cal_res else None,
        "cal1": cal_res.get_counts(1) if cal_res else None,
    }
    notes = {
        "simulator": "AerSimulator(matrix_product_state) + compact FakeBackend noise + idle T1/T2 relaxation",
        "simulated_qubits": n,
        "shots": {"populations": p["populations"], "parity_each": parity_shots, "cal_each": p["cal"]},
        "dynamical_decoupling": "not applied on sim (no-op; idle noise is Markovian T1/T2)" if s["dynamical_decoupling"] else "off",
        "measure_twirling": "not applied on sim (no-op)" if s["measure_twirling"] else "off",
    }
    return counts, notes


# ------------------------------------------------------------------------------------------ qpu
def _qpu_seconds(job) -> float | None:
    try:
        u = job.usage()
        if u is not None:
            return float(u)
    except Exception:
        pass
    try:
        m = job.metrics() or {}
        q = (m.get("usage") or {}).get("quantum_seconds") or (m.get("usage") or {}).get("seconds")
        if q is not None:
            return float(q)
    except Exception:
        pass
    return None


def submit_qpu(prep: dict, backend):
    """Submit ONE SamplerV2 job (job mode) with every circuit as its own PUB. Returns the job
    without waiting (so many experiments can sit in the IBM queue at once)."""
    from qiskit_ibm_runtime import SamplerV2

    s = prep["spec"]
    pubs, _ = build_qpu_pubs(prep)
    sampler = SamplerV2(mode=backend)
    o = sampler_options(s)
    sampler.options.default_shots = o["default_shots"]
    if "dynamical_decoupling" in o:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    if "twirling" in o:
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = True
    return sampler.run(pubs)


def counts_from_result(res, n_qubits: int, has_cal: bool) -> dict:
    n_par = 2 * n_qubits + 2

    def counts_of(i):
        return getattr(res[i].data, "c").get_counts()

    return {
        "populations": counts_of(0),
        "parity": [counts_of(1 + j) for j in range(n_par)],
        "cal0": counts_of(1 + n_par) if has_cal else None,
        "cal1": counts_of(2 + n_par) if has_cal else None,
    }


def qpu_notes(spec: dict, num_pubs: int) -> dict:
    s = normalize_spec(spec)
    return {
        "sampler_options": sampler_options(s),
        "num_pubs": num_pubs,
        "dynamical_decoupling": "SamplerV2 dynamical_decoupling XpXm" if s["dynamical_decoupling"] else "off",
        "measure_twirling": "SamplerV2 twirling.enable_measure" if s["measure_twirling"] else "off",
    }


def run_qpu(prep: dict, backend, state: dict | None = None) -> tuple[dict, dict, list[str], float | None]:
    """Submit and wait (blocking path used by the agent loop)."""
    job = submit_qpu(prep, backend)
    job_id = job.job_id()
    if state is not None:
        state["job_ids"] = [job_id]  # recorded BEFORE waiting, so a failure still reports/charges it
    res = job.result()
    counts = counts_from_result(res, prep["spec"]["n_qubits"], bool(prep["cal"]))
    notes = qpu_notes(prep["spec"], len(build_qpu_pubs(prep)[0]))
    return counts, notes, [job_id], _qpu_seconds(job)


# ------------------------------------------------------------------------------------------ api
def _empty_result(exp_id, spec, backend, err, wall, plan=None):
    return {
        "exp_id": exp_id,
        "spec": spec,
        "physical_qubits": plan["qubits"] if plan else None,
        "tree_edges": plan["tree_edges"] if plan else None,
        "root": plan["root"] if plan else None,
        "circuit_depth": None,
        "two_qubit_gate_count": None,
        "fidelity": None,
        "ci_low": None,
        "ci_high": None,
        "populations": None,
        "coherence": None,
        "parity_curve": None,
        "certified": False,
        "qpu_seconds": 0.0,
        "wall_seconds": round(wall, 2),
        "backend": getattr(backend, "name", None),
        "job_ids": [],
        "error": err,
    }


def run_experiment(spec: dict, backend, calibration: dict, exp_id: str) -> dict:
    """Plan, build, run and analyse one GHZ experiment. Never raises: failures come back as a
    result with "error" set."""
    t0 = time.time()
    plan = None
    s = spec
    state: dict = {}
    try:
        s = normalize_spec(spec)
        if s["mode"] not in ("sim", "qpu"):
            raise ValueError(f"unknown mode {s['mode']!r}")
        aer = s["mode"] == "sim" and s.get("sim_engine") == "aer_mps"
        prep = prepare(s, backend, calibration, scheduled=aer)
        plan = prep["plan"]
        job_ids: list[str] = []
        qpu_s = 0.0
        if s["mode"] == "sim":
            counts, notes = run_sim(prep, backend, calibration)
        else:
            counts, notes, job_ids, used = run_qpu(prep, backend, state)
            qpu_s = used if used is not None else estimate_qpu_seconds(s)
            notes["qpu_seconds_source"] = "job.usage()" if used is not None else "estimate (usage unavailable)"
        n = s["n_qubits"]
        fit = analyze(
            n,
            counts["populations"],
            counts["parity"],
            counts.get("cal0"),
            counts.get("cal1"),
            mitigate=s["readout_mitigation"],
        )
        result = _empty_result(exp_id, s, backend, None, time.time() - t0, plan)
        result.update(
            {
                "circuit_depth": prep["circuit_depth"],
                "two_qubit_gate_count": prep["two_qubit_gate_count"],
                "transpiled_depth": prep["transpiled_depth"],
                "plan_depth": plan["depth"],
                "sim_method": (s.get("sim_engine") if s["mode"] == "sim" else None),
                **fit,
                "qpu_seconds": round(float(qpu_s), 3),
                "job_ids": job_ids,
                "notes": notes,
                "wall_seconds": round(time.time() - t0, 2),
            }
        )
        return result
    except Exception as e:  # never raise to the agent
        r = _empty_result(exp_id, s, backend, f"{type(e).__name__}: {e}", time.time() - t0, plan)
        if state.get("job_ids"):  # a job was submitted: report it and charge the conservative estimate
            r["job_ids"] = state["job_ids"]
            try:
                r["qpu_seconds"] = estimate_qpu_seconds(s)
            except Exception:
                r["qpu_seconds"] = 60.0
        r["traceback"] = traceback.format_exc(limit=6)
        return r

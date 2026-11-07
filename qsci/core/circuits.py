"""GHZ preparation, parity-oscillation and readout-calibration circuits.

Circuits are built on n *logical* qubits where logical i == physical plan["qubits"][i]; they are then
transpiled with initial_layout=plan["qubits"] so the placement is exactly the planned one.
"""
from __future__ import annotations

import math

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Parameter

from .layout import linear_schedule, schedule_tree

CREG = "c"


def phis(n: int) -> list[float]:
    """phi_j = j*pi/(N+1), j = 0..2N+1 (2N+2 settings, uniformly covering [0, 2*pi))."""
    return [j * math.pi / (n + 1) for j in range(2 * n + 2)]


def ghz_layers(plan: dict, variant: str) -> list[list[tuple[int, int]]]:
    if variant == "tree":
        return schedule_tree(plan["root"], plan["tree_edges"])
    if variant == "linear":
        return linear_schedule(plan["root"], plan["tree_edges"])
    raise ValueError(f"unknown circuit_variant {variant!r}; expected 'tree' or 'linear'")


def _base(n: int, name: str) -> QuantumCircuit:
    return QuantumCircuit(QuantumRegister(n, "q"), ClassicalRegister(n, CREG), name=name)


def ghz_prep(plan: dict, variant: str, name: str = "ghz") -> QuantumCircuit:
    idx = {q: i for i, q in enumerate(plan["qubits"])}
    n = len(plan["qubits"])
    qc = _base(n, name)
    qc.h(idx[plan["root"]])
    for k, layer in enumerate(ghz_layers(plan, variant)):
        if variant == "linear" and k:
            qc.barrier()  # forbid the scheduler from parallelising: one CNOT per time step
        for p, c in layer:
            qc.cx(idx[p], idx[c])
    return qc


def build_logical_circuits(plan: dict, variant: str) -> dict:
    """Returns {"populations": qc, "parity": parameterized qc (Parameter 'phi'), "phi": Parameter,
    "cal0": qc, "cal1": qc}."""
    n = len(plan["qubits"])
    pops = ghz_prep(plan, variant, "populations")
    pops.barrier()
    pops.measure(range(n), range(n))

    phi = Parameter("phi")
    par = ghz_prep(plan, variant, "parity")
    par.barrier()
    for i in range(n):
        par.rz(phi, i)
        par.h(i)
    par.measure(range(n), range(n))

    cal0 = _base(n, "cal0")
    cal0.measure(range(n), range(n))
    cal1 = _base(n, "cal1")
    cal1.x(range(n))
    cal1.measure(range(n), range(n))
    return {"populations": pops, "parity": par, "phi": phi, "cal0": cal0, "cal1": cal1}


def transpile_on_layout(circuits: list[QuantumCircuit], backend, physical: list[int], scheduled: bool = False):
    """Transpile keeping logical i on physical[i] (initial_layout + optimization_level=1; every
    tree edge is a native coupler so no routing/SWAPs are needed). Raises if placement moved."""
    kwargs = dict(backend=backend, initial_layout=list(physical), optimization_level=1, seed_transpiler=1234)
    if scheduled:
        kwargs["scheduling_method"] = "alap"
    out = transpile(circuits, **kwargs)
    out = out if isinstance(out, list) else [out]
    for tc in out:
        verify_placement(tc, physical)
    return out


def verify_placement(tc: QuantumCircuit, physical: list[int]) -> None:
    allowed = set(physical)
    if tc.layout is not None:
        final = tc.layout.final_index_layout()
        if list(final[: len(physical)]) != list(physical):
            raise RuntimeError(f"transpiler moved qubits: wanted {physical}, got {final[: len(physical)]}")
    for inst in tc.data:
        if inst.operation.name in ("barrier", "delay"):
            continue
        for q in inst.qubits:
            pq = tc.find_bit(q).index
            if pq not in allowed:
                raise RuntimeError(f"gate {inst.operation.name} on unplanned physical qubit {pq}")
    for name in ("swap",):
        if name in tc.count_ops():
            raise RuntimeError("transpiler inserted SWAPs")


def two_qubit_stats(tc: QuantumCircuit) -> tuple[int, int]:
    """(2q-gate count, 2q-gate layer depth) of a transpiled circuit. Barriers synchronise their
    qubits (so the barrier-separated 'linear' variant counts one layer per CNOT)."""
    d: dict[int, int] = {}
    count = 0
    for inst in tc.data:
        qs = [tc.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "barrier":
            m = max((d.get(q, 0) for q in qs), default=0)
            for q in qs:
                d[q] = m
        elif inst.operation.num_qubits == 2:
            count += 1
            m = max(d.get(q, 0) for q in qs) + 1
            for q in qs:
                d[q] = m
    return count, max(d.values(), default=0)


def compact(tc: QuantumCircuit, physical: list[int]) -> QuantumCircuit:
    """Drop the idle device qubits: physical[i] -> compact qubit i (keeps delays on active qubits)."""
    pos = {p: i for i, p in enumerate(physical)}
    n = len(physical)
    out = QuantumCircuit(QuantumRegister(n, "q"), ClassicalRegister(tc.num_clbits, CREG), name=tc.name)
    for inst in tc.data:
        pq = [tc.find_bit(q).index for q in inst.qubits]
        if any(p not in pos for p in pq):
            if inst.operation.name in ("delay", "barrier"):
                pq = [p for p in pq if p in pos]
                if not pq:
                    continue
            else:
                raise RuntimeError(f"{inst.operation.name} on inactive qubit")
        cl = [tc.find_bit(c).index for c in inst.clbits]
        if inst.operation.name == "barrier":
            out.barrier([out.qubits[pos[p]] for p in pq])
            continue
        out.append(inst.operation, [out.qubits[pos[p]] for p in pq], [out.clbits[c] for c in cl])
    return out

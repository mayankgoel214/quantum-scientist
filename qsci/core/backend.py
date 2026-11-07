"""Backend access + calibration snapshots (calibration.json schema in CONTRACTS.md)."""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path

TWO_Q_GATES = ("cz", "ecr", "cx")


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        load_dotenv()  # also honour a .env in the cwd, without overriding
    except Exception:
        pass


def get_backend(mode: str, name: str | None = None):
    """'sim' -> FakeFez (156q Heron, offline). 'qpu' -> live IBM backend via QiskitRuntimeService.

    For 'qpu', `name` (or IBM_BACKEND env) selects the device; otherwise the least busy operational
    real device with >= 100 qubits. Secrets are read from .env and never printed.
    """
    if mode == "sim":
        from qiskit_ibm_runtime.fake_provider import FakeFez

        return FakeFez()
    if mode != "qpu":
        raise ValueError(f"unknown mode {mode!r} (expected 'sim' or 'qpu')")
    _load_env()
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("IBM_QUANTUM_TOKEN") or None
    instance = os.environ.get("IBM_QUANTUM_INSTANCE") or None
    if not token:
        raise RuntimeError("IBM_QUANTUM_TOKEN is not set (.env)")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
    name = name or (os.environ.get("IBM_BACKEND") or "").split("#")[0].strip() or None
    if name:
        return service.backend(name)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=100)


def _is_fake(backend) -> bool:
    return type(backend).__module__.startswith("qiskit_ibm_runtime.fake_provider")


def _f(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(x) or math.isinf(x)) else x


def _undirected_edges(backend) -> list[tuple[int, int]]:
    cmap = backend.coupling_map
    edges = set()
    for a, b in cmap.get_edges():
        edges.add((min(a, b), max(a, b)))
    return sorted(edges)


def chip_coordinates(num_qubits: int, edges: list[tuple[int, int]]) -> dict[int, tuple[float, float]]:
    """Deterministic 2-D coordinates for a chip map.

    IBM heavy-hex devices number qubits row by row: each row is a run of consecutively indexed,
    consecutively coupled qubits, and the rows are joined by short 'bridge' qubits. We recover the
    rows, align them using the bridges (bridges sit vertically between their two row neighbours) and
    lay the picture out on a grid. Falls back to a fixed-seed spring layout for other topologies.
    """
    adj: dict[int, set[int]] = {q: set() for q in range(num_qubits)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    # segments of consecutive indices joined by (i, i+1) edges
    seg_of: dict[int, int] = {}
    segments: list[list[int]] = []
    for q in range(num_qubits):
        if q - 1 in adj[q] and (q - 1) in seg_of:
            seg_of[q] = seg_of[q - 1]
            segments[seg_of[q]].append(q)
        else:
            seg_of[q] = len(segments)
            segments.append([q])
    rows = [s for s in segments if len(s) >= 5]
    coords: dict[int, tuple[float, float]] = {}
    ok = len(rows) >= 2
    if ok:
        row_idx = {}
        for r, s in enumerate(rows):
            for i, q in enumerate(s):
                row_idx[q] = (r, i)
        offsets = {0: 0.0}
        bridges = [q for q in range(num_qubits) if q not in row_idx]
        # resolve row offsets via bridges joining row r and r+1
        for _ in range(len(rows)):
            for q in bridges:
                nb = [row_idx[n] for n in adj[q] if n in row_idx]
                if len(nb) != 2:
                    continue
                (r1, i1), (r2, i2) = sorted(nb)
                if r1 in offsets and r2 not in offsets:
                    offsets[r2] = offsets[r1] + i1 - i2
                elif r2 in offsets and r1 not in offsets:
                    offsets[r1] = offsets[r2] + i2 - i1
        if len(offsets) != len(rows):
            ok = False
        else:
            for q, (r, i) in row_idx.items():
                coords[q] = (float(i + offsets[r]), float(2 * r))
            for q in bridges:
                nb = [n for n in adj[q] if n in coords]
                if not nb:
                    ok = False
                    break
                xs = [coords[n][0] for n in nb]
                ys = [coords[n][1] for n in nb]
                y = sum(ys) / len(ys) if len(nb) > 1 else ys[0] + 1.0
                coords[q] = (sum(xs) / len(xs), y)
    if ok and len(set(coords.values())) == num_qubits:
        minx = min(x for x, _ in coords.values())
        return {q: (x - minx, y) for q, (x, y) in coords.items()}
    # fallback: deterministic spring layout
    import rustworkx as rx

    g = rx.PyGraph()
    g.add_nodes_from(range(num_qubits))
    g.add_edges_from_no_data(edges)
    pos = rx.spring_layout(g, seed=7)
    return {q: (round(10 * pos[q][0], 3), round(10 * pos[q][1], 3)) for q in range(num_qubits)}


def get_calibration(backend) -> dict:
    """Snapshot of the backend calibration in the calibration.json schema. Missing values -> None."""
    target = backend.target
    n = backend.num_qubits
    edges = _undirected_edges(backend)
    coords = chip_coordinates(n, edges)

    def inst_err(name: str, qargs: tuple) -> float | None:
        try:
            if name not in target.operation_names:
                return None
            props = target[name].get(qargs)
            return _f(props.error) if props is not None else None
        except Exception:
            return None

    qubits = []
    qprops = getattr(target, "qubit_properties", None) or [None] * n
    for q in range(n):
        qp = qprops[q] if q < len(qprops) else None
        t1 = _f(getattr(qp, "t1", None))
        t2 = _f(getattr(qp, "t2", None))
        x, y = coords.get(q, (None, None))
        qubits.append(
            {
                "id": q,
                "t1_us": round(t1 * 1e6, 3) if t1 is not None else None,
                "t2_us": round(t2 * 1e6, 3) if t2 is not None else None,
                "readout_error": inst_err("measure", (q,)),
                "sx_error": inst_err("sx", (q,)),
                "x": x,
                "y": y,
            }
        )
    gate = next((g for g in TWO_Q_GATES if g in target.operation_names), None)
    edge_rows = []
    for a, b in edges:
        errs = [e for e in (inst_err(gate, (a, b)), inst_err(gate, (b, a))) if e is not None] if gate else []
        edge_rows.append({"a": a, "b": b, "error": min(errs) if errs else None, "gate": gate})
    return {
        "backend": backend.name,
        "num_qubits": n,
        "source": "fake" if _is_fake(backend) else "live",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "qubits": qubits,
        "edges": edge_rows,
    }

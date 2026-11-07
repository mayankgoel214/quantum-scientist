"""Fast noisy GHZ simulator (Pauli-frame / Pauli-twirling approximation), vectorised over shots.

Why: a noisy matrix-product-state trajectory simulation of the 2N+3 circuits costs ~20 ms per
shot at N=40 (~30 min per experiment), far too slow for the agent loop. GHZ preparation is
Clifford, so under Pauli noise every shot ends in P|GHZ> for a Pauli P = X^fx Z^fz that we
track exactly by propagating error frames through the CNOT tree:

    P|GHZ> = (|fx> + s |~fx>)/sqrt(2),  s = (-1)^{|fz|}

The non-Clifford measurement rotations are then sampled EXACTLY from that state:
  * Z basis: outcome fx or ~fx with prob 1/2 each;
  * Rz(phi)^N then H^N: uniformly random bits with P(even parity) = (1 + s cos(m phi))/2,
    m = N - 2|fx|.
followed by per-qubit readout flips.

Noise model (from the calibration snapshot, i.e. the same numbers the layout planner sees):
  * each CNOT (= H.CZ.H on the target): 2-qubit depolarizing with p = 5/4 * cz_error
    (+ 1q depolarizing p = 3/2 * sx_error for each of the 2 target 1q gates);
  * the root H and every basis-change 1q gate: 1q depolarizing p = 3/2 * sx_error;
  * idle qubits that already hold the GHZ state: Pauli-twirled T1/T2 relaxation for the
    duration of each layer they sit out (qubits still in |0> are immune to T1/T2, so no noise);
  * readout: symmetric flip with the calibrated readout_error.
Approximations: thermal relaxation is Pauli-twirled (PTA); gate errors are modelled as
depolarizing; no crosstalk / leakage / coherent errors; DD and measurement twirling are no-ops.
"""
from __future__ import annotations

import math

import numpy as np

from .layout import linear_schedule, schedule_tree

DEFAULT_CZ_NS = 84.0
DEFAULT_SX_NS = 24.0


def _durations(backend) -> tuple[dict, float]:
    cz, sx = {}, DEFAULT_SX_NS
    try:
        t = backend.target
        gate = next(g for g in ("cz", "ecr", "cx") if g in t.operation_names)
        for qargs, props in t[gate].items():
            if props is not None and props.duration:
                cz[tuple(qargs)] = props.duration * 1e9
        sxd = [p.duration for p in t["sx"].values() if p is not None and p.duration]
        if sxd:
            sx = float(np.median(sxd)) * 1e9
    except Exception:
        pass
    return cz, sx


def _pta_probs(t_ns: float, t1_us, t2_us) -> tuple[float, float, float]:
    t1 = (t1_us or 100.0) * 1e3
    t2 = min((t2_us or 100.0) * 1e3, 2 * t1)
    px = py = (1 - math.exp(-t_ns / t1)) / 4
    pz = max(0.0, (1 - math.exp(-t_ns / t2)) / 2 - px)
    return px, py, pz


class _Frames:
    def __init__(self, shots: int, n: int, rng):
        self.fx = np.zeros((shots, n), dtype=bool)
        self.fz = np.zeros((shots, n), dtype=bool)
        self.rng = rng
        self.shots = shots

    def depol1(self, q: int, p: float):
        if p <= 0:
            return
        hit = self.rng.random(self.shots) < p
        k = self.rng.integers(1, 4, self.shots)  # 1=X 2=Y 3=Z
        self.fx[:, q] ^= hit & (k != 3)
        self.fz[:, q] ^= hit & (k != 1)

    def depol2(self, a: int, b: int, p: float):
        if p <= 0:
            return
        hit = self.rng.random(self.shots) < p
        k = self.rng.integers(1, 16, self.shots)  # non-identity 2q Pauli, 2 bits per qubit
        ka, kb = k & 3, k >> 2
        for q, kk in ((a, ka), (b, kb)):
            self.fx[:, q] ^= hit & ((kk == 1) | (kk == 2))
            self.fz[:, q] ^= hit & ((kk == 2) | (kk == 3))

    def pauli(self, q: int, px: float, py: float, pz: float):
        if px + py + pz <= 0:
            return
        r = self.rng.random(self.shots)
        x = r < px
        y = (r >= px) & (r < px + py)
        z = (r >= px + py) & (r < px + py + pz)
        self.fx[:, q] ^= x | y
        self.fz[:, q] ^= y | z

    def cx(self, c: int, t: int):
        self.fx[:, t] ^= self.fx[:, c]
        self.fz[:, c] ^= self.fz[:, t]


def simulate_counts(plan: dict, variant: str, calibration: dict, backend, n_parity: int, shots_pop: int,
                    shots_par: int, shots_cal: int, seed: int = 0) -> dict:
    """Counts for populations, parity_0..n_parity-1 and the two readout calibration circuits."""
    rng = np.random.default_rng(seed)
    phys = plan["qubits"]
    n = len(phys)
    idx = {q: i for i, q in enumerate(phys)}
    qcal = {int(q["id"]): q for q in calibration["qubits"]}
    ecal = {}
    for e in calibration["edges"]:
        ecal[(e["a"], e["b"])] = ecal[(e["b"], e["a"])] = e.get("error")
    ro = np.array([qcal[p].get("readout_error") or 0.02 for p in phys])
    sxe = np.array([qcal[p].get("sx_error") or 3e-4 for p in phys])
    cz_ns, sx_ns = _durations(backend)
    layers = schedule_tree(plan["root"], plan["tree_edges"]) if variant == "tree" else linear_schedule(plan["root"], plan["tree_edges"])

    total = shots_pop + n_parity * shots_par
    F = _Frames(total, n, rng)
    # root H
    r0 = idx[plan["root"]]
    F.depol1(r0, 1.5 * sxe[r0])
    joined = np.zeros(n, dtype=bool)
    joined[r0] = True
    for layer in layers:
        dur = max(cz_ns.get((p, c), cz_ns.get((c, p), DEFAULT_CZ_NS)) for p, c in layer) + 2 * sx_ns
        busy = set()
        for p, c in layer:
            a, b = idx[p], idx[c]
            F.cx(a, b)
            err = ecal.get((p, c))
            F.depol2(a, b, min(1.0, 1.25 * (0.02 if err is None else err)))
            F.depol1(b, 1.5 * sxe[b])
            F.depol1(b, 1.5 * sxe[b])
            busy.update((a, b))
        for q in np.nonzero(joined)[0]:
            if q not in busy:
                q_phys = phys[q]
                F.pauli(q, *_pta_probs(dur, qcal[q_phys].get("t1_us"), qcal[q_phys].get("t2_us")))
        for p, c in layer:
            joined[idx[c]] = True
    # basis-change 1q gates (H, or Rz.H): depolarizing before measurement
    for q in range(n):
        F.depol1(q, 1.5 * sxe[q])

    fx, s = F.fx, np.where(F.fz.sum(1) % 2 == 0, 1.0, -1.0)
    out = {}
    # populations: fx or ~fx with prob 1/2
    pz = fx[:shots_pop] ^ (rng.random((shots_pop, 1)) < 0.5)
    out["populations"] = _to_counts(_readout(pz, ro, rng))
    m = n - 2 * fx[shots_pop:].sum(1)
    sp = s[shots_pop:]
    out["parity"] = []
    for j in range(n_parity):
        phi = j * math.pi / (n + 1)
        sl = slice(j * shots_par, (j + 1) * shots_par)
        p_even = (1 + sp[sl] * np.cos(m[sl] * phi)) / 2
        bits = rng.random((shots_par, n)) < 0.5
        want_odd = rng.random(shots_par) >= p_even
        fix = (bits.sum(1) % 2 == 1) != want_odd
        bits[fix, 0] ^= True
        out["parity"].append(_to_counts(_readout(bits, ro, rng)))
    if shots_cal:
        z = np.zeros((shots_cal, n), dtype=bool)
        out["cal0"] = _to_counts(_readout(z, ro, rng))
        out["cal1"] = _to_counts(_readout(~z, ro, rng))
    else:
        out["cal0"] = out["cal1"] = None
    return out


def _readout(bits: np.ndarray, ro: np.ndarray, rng) -> np.ndarray:
    return bits ^ (rng.random(bits.shape) < ro[None, :])


def _to_counts(bits: np.ndarray) -> dict:
    """bool[shots, n] (column i = clbit i) -> qiskit-style counts (clbit 0 rightmost)."""
    rows, k = np.unique(bits[:, ::-1].astype(np.uint8) + ord("0"), axis=0, return_counts=True)
    return {r.tobytes().decode(): int(c) for r, c in zip(rows, k)}

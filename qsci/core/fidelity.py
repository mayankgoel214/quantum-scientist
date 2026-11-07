"""GHZ fidelity witness from measured counts.

F = (P(0...0) + P(1...1)) / 2 + C / 2

Coherence C comes from the parity oscillation: after Rz(phi) on every qubit and an X-basis
measurement, the ideal GHZ state gives <P(phi)> = cos(N*phi + theta), and in general
<P(phi)> = C * cos(N*phi + theta) + (terms at other frequencies), with C = 2|rho_{0..0,1..1}|.
With phi_j = j*pi/(N+1), j = 0..2N+1 (2N+2 equally spaced points on [0, 2*pi)), the discrete
Fourier component at frequency N isolates C exactly, for ANY unknown phase offset theta:

    C = | sum_j exp(-i N phi_j) <P_j> | / (N + 1)

(debiased for shot noise, see analyze()). CI = basic bootstrap 95% interval over shots.

NOTE (deviation from the CONTRACTS.md formula): with these phi_j the contract's
|sum_j (-1)^j <P_j>| / (2N+2) evaluates to 0 for a perfect GHZ state, because (-1)^j is the
frequency-(N+1) component, not frequency N. The Fourier form above is what the 2N+2-point
protocol is designed for (it is also insensitive to the device's phase offset theta).

Readout mitigation (optional): per-qubit ("tensored") confusion matrices estimated from two
calibration circuits (all-|0> and all-|1>). The inverse of a tensor product is the tensor product
of inverses, and both the GHZ populations and the parity are product-form functionals of the
bitstring, so the mitigated values are computed EXACTLY under the tensored model as
sum_s p(s) * prod_q w_q(s_q) over observed bitstrings. Approximations: readout errors are assumed
uncorrelated across qubits and state-independent of neighbours (no crosstalk), and the per-qubit
rates are estimated from the 2 calibration circuits (their shot noise is NOT bootstrapped).
"""
from __future__ import annotations

import math

import numpy as np


def counts_to_arrays(counts: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """counts {'bitstring': k} (qiskit order: clbit 0 is the RIGHTMOST char) -> (bits[k, n], c[k])."""
    keys = list(counts.keys())
    bits = np.zeros((len(keys), n), dtype=np.int8)
    for r, key in enumerate(keys):
        key = key.replace(" ", "")
        if len(key) < n:
            key = key.zfill(n)
        # clbit i is key[-1 - i]
        bits[r] = np.frombuffer(key[::-1][:n].encode(), dtype=np.uint8) - ord("0")
    c = np.array([counts[k] for k in keys], dtype=np.int64)
    return bits, c


def readout_rates(cal0: dict, cal1: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """e0[q] = P(read 1 | prepared 0), e1[q] = P(read 0 | prepared 1)."""
    b0, c0 = counts_to_arrays(cal0, n)
    b1, c1 = counts_to_arrays(cal1, n)
    e0 = (b0 * c0[:, None]).sum(0) / c0.sum()
    e1 = ((1 - b1) * c1[:, None]).sum(0) / c1.sum()
    return e0.astype(float), e1.astype(float)


def _inverse_rows(e0, e1):
    """Per-qubit inverse confusion matrix rows. A = [[1-e0, e1], [e0, 1-e1]] (rows measured,
    cols true) -> Ainv = [[1-e1, -e1], [-e0, 1-e0]] / det."""
    det = 1.0 - e0 - e1
    det = np.where(np.abs(det) < 1e-3, 1e-3, det)
    row0 = np.stack([(1 - e1) / det, -e1 / det], axis=1)  # Ainv[0, s]
    row1 = np.stack([-e0 / det, (1 - e0) / det], axis=1)  # Ainv[1, s]
    return row0, row1


def _prod_weight(bits: np.ndarray, table: np.ndarray) -> np.ndarray:
    """prod_q table[q, bits[:, q]] for every row (log-free; values are O(1))."""
    q = np.arange(bits.shape[1])
    return np.prod(table[q[None, :], bits], axis=1)


def weights_populations(bits, n, e0=None, e1=None):
    if e0 is None:
        w0 = (bits.sum(1) == 0).astype(float)
        w1 = (bits.sum(1) == n).astype(float)
        return w0, w1
    r0, r1 = _inverse_rows(e0, e1)
    return _prod_weight(bits, r0), _prod_weight(bits, r1)


def weights_parity(bits, e0=None, e1=None):
    if e0 is None:
        return 1.0 - 2.0 * (bits.sum(1) % 2)
    r0, r1 = _inverse_rows(e0, e1)
    return _prod_weight(bits, r0 - r1)  # (1, -1) . Ainv[:, s]


def coherence_from_parities(parities, n: int) -> float:
    ph = np.array([j * math.pi / (n + 1) for j in range(2 * n + 2)])
    z = np.sum(np.exp(-1j * n * ph) * np.asarray(parities, dtype=float))
    return float(np.abs(z) / (n + 1))


def analyze(
    n: int,
    pop_counts: dict,
    parity_counts: list[dict],
    cal0: dict | None = None,
    cal1: dict | None = None,
    mitigate: bool = False,
    n_boot: int = 400,
    seed: int = 0,
) -> dict:
    """Fidelity + bootstrap 95% CI (resampling shots of every circuit independently)."""
    if len(parity_counts) != 2 * n + 2:
        raise ValueError(f"need {2 * n + 2} parity circuits, got {len(parity_counts)}")
    e0 = e1 = None
    if mitigate:
        if cal0 is None or cal1 is None:
            raise ValueError("readout mitigation needs cal0/cal1 counts")
        e0, e1 = readout_rates(cal0, cal1, n)
    rng = np.random.default_rng(seed)

    bits, c = counts_to_arrays(pop_counts, n)
    w0, w1 = weights_populations(bits, n, e0, e1)
    shots = c.sum()
    p0 = float(c @ w0 / shots)
    p1 = float(c @ w1 / shots)
    boot_c = rng.multinomial(shots, c / shots, size=n_boot)
    bp = (boot_c @ w0 + boot_c @ w1) / shots

    par, bpar = [], []
    for counts in parity_counts:
        b, k = counts_to_arrays(counts, n)
        w = weights_parity(b, e0, e1)
        s = k.sum()
        par.append(float(k @ w / s))
        bk = rng.multinomial(s, k / s, size=n_boot)
        bpar.append(bk @ w / s)
    bpar = np.stack(bpar, axis=1)  # (n_boot, 2N+2)

    ph = np.array([j * math.pi / (n + 1) for j in range(2 * n + 2)])
    phase = np.exp(-1j * n * ph)
    z = complex(np.dot(phase, par)) / (n + 1)
    zb = (bpar @ phase) / (n + 1)
    # |z| is biased upward by shot noise (a classical mixture would show C ~ noise > 0).
    # Debias: C = sqrt(max(0, |z|^2 - E|dz|^2)), with E|dz|^2 estimated from the bootstrap.
    noise2 = float(np.mean(np.abs(zb - z) ** 2))

    def coh_of(zz):
        return np.sqrt(np.clip(np.abs(zz) ** 2 - noise2, 0.0, None))

    pops = float(np.clip(p0 + p1, 0.0, 1.0))
    coh = float(np.clip(coh_of(z), 0.0, 1.0))
    fid = pops / 2 + coh / 2
    bfid = np.clip(bp, 0, 1) / 2 + np.clip(coh_of(zb), 0, 1) / 2
    # basic (reverse-percentile) bootstrap interval: robust to the estimator's residual bias
    q_lo, q_hi = np.percentile(bfid, [2.5, 97.5])
    lo, hi = float(np.clip(2 * fid - q_hi, 0, 1)), float(np.clip(2 * fid - q_lo, 0, 1))
    lo, hi = min(lo, fid), max(hi, fid)
    out = {
        "fidelity": round(fid, 5),
        "ci_low": round(float(lo), 5),
        "ci_high": round(float(hi), 5),
        "populations": round(pops, 5),
        "p_all0": round(p0, 5),
        "p_all1": round(p1, 5),
        "coherence": round(coh, 5),
        "parity_curve": [{"phi": round(float(f), 6), "parity": round(p, 5)} for f, p in zip(ph, par)],
        "certified": bool(lo > 0.5),
    }
    if mitigate:
        out["readout_rates"] = {"mean_e0": round(float(e0.mean()), 5), "mean_e1": round(float(e1.mean()), 5)}
    return out

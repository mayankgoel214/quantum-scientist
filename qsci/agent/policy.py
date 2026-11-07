"""Scripted experimental policy.

Used two ways:
  * by FakeLLM, so the whole loop runs end-to-end with no API key (`--llm fake`);
  * as the autopilot fallback when Gemini fails, so a flaky API never wastes the night.

It is a pure function of the observed state (history of results), so it can take over at any
point mid-run. Strategy: geometric scale-up on the simulator, binary search at the frontier,
try variants that attack the dominant error at the first failing N, then (qpu mode) confirm the
largest sim-certified configs on hardware, largest first, stepping down on failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

DEFAULT_CONFIG = {
    "layout_strategy": "weighted_tree",
    "root": None,
    "exclude_qubits": [],
    "circuit_variant": "tree",
    "dynamical_decoupling": True,
    "measure_twirling": True,
    "readout_mitigation": True,
    "shots": 2000,
}

CONFIG_KEYS = ("n_qubits", "layout_strategy", "root", "exclude_qubits", "circuit_variant",
               "dynamical_decoupling", "measure_twirling", "readout_mitigation")


def config_key(spec: dict) -> tuple:
    """Identity of a physical configuration (ignores shots / hypothesis / mode)."""
    out = []
    for k in CONFIG_KEYS:
        v = spec.get(k)
        if k == "exclude_qubits":
            v = tuple(sorted(int(q) for q in (v or [])))
        out.append(v)
    return tuple(out)


def policy_key(spec: dict) -> tuple:
    return config_key(spec) + (int(spec.get("shots") or 0),)


@dataclass
class AgentState:
    mode: str                                  # "sim" | "qpu"
    num_qubits: int
    history: list[dict] = field(default_factory=list)   # ExperimentResults, oldest first
    bad_qubits: list[int] = field(default_factory=list)
    calibration_seen: bool = False
    qpu_remaining_s: float = 0.0
    llm_calls_left: int = 0
    estimate_qpu: Callable[[dict], float] | None = None
    hw_available: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _ok(r: dict) -> bool:
    return not r.get("error") and r.get("fidelity") is not None


class ScriptedPolicy:
    def __init__(self, *, start_n: int = 8, max_sim_experiments: int = 24,
                 max_hw_experiments: int = 4, hw_shots: int = 4000) -> None:
        self.start_n = start_n
        self.max_sim = max_sim_experiments
        self.max_hw = max_hw_experiments
        self.hw_shots = hw_shots

    # ------------------------------------------------------------------ public
    def next_action(self, st: AgentState) -> tuple[str, str, dict]:
        if not st.calibration_seen:
            return ("First I need to know the chip: which qubits and couplers are good, and "
                    "which to avoid.", "get_calibration_summary", {})
        sim = [r for r in st.history if (r.get("spec") or {}).get("mode") == "sim"]
        if len(sim) < self.max_sim:
            step = self._sim_step(st, sim)
            if step is not None:
                return step
        if st.mode == "qpu" and st.hw_available:
            step = self._hw_step(st, sim)
            if step is not None:
                return step
        return ("The frontier is mapped as well as the budget allows; writing up the results.",
                "finish", {"summary": self._summary(st)})

    # ------------------------------------------------------------------ sim phase
    def _sim_step(self, st: AgentState, sim: list[dict]):
        tried = {policy_key(r["spec"]) for r in sim}
        good = [r for r in sim if _ok(r)]
        cert = [r for r in good if r.get("certified")]
        lo_r = max(cert, key=lambda r: (r["spec"]["n_qubits"], r["fidelity"]), default=None)
        lo = lo_r["spec"]["n_qubits"] if lo_r else 0
        base = {k: lo_r["spec"].get(k, v) for k, v in DEFAULT_CONFIG.items()} if lo_r else dict(DEFAULT_CONFIG)
        base["shots"] = DEFAULT_CONFIG["shots"]
        failed_above = [r["spec"]["n_qubits"] for r in good
                        if not r.get("certified") and r["spec"]["n_qubits"] > lo]
        hi = min(failed_above) if failed_above else None
        nmax = st.num_qubits

        def spec_for(n, cfg, hyp):
            s = dict(cfg)
            s["n_qubits"] = int(n)
            s["hypothesis"] = hyp
            return s

        def fresh(s):
            return policy_key(s) not in tried

        # A) no failure above the record yet -> scale up geometrically
        if hi is None:
            if lo >= nmax:
                return None
            n = self.start_n if lo == 0 else min(nmax, max(lo + 2, int(lo * 1.6)))
            s = spec_for(n, base, f"N={n} should still certify if errors compound slowly.")
            if fresh(s):
                if lo == 0:
                    text = (f"Starting at N={n} with a calibration-weighted tree, DD, twirling and "
                            "readout mitigation all on, to get a baseline.")
                else:
                    text = (f"N={lo} certified (record so far), so scale up aggressively to "
                            f"N={n} to find where fidelity crosses 0.5.")
                return text, "run_simulation", s
            # this config was tried and errored -> fall through to the frontier logic
            hi = n
        # B) binary search between the record and the first failure
        if hi - lo > 2:
            mid = (lo + hi) // 2
            s = spec_for(mid, base, f"Bisecting the frontier between N={lo} and N={hi}.")
            if fresh(s):
                return (f"N={hi} failed but N={lo} certified; bisect at N={mid} rather than "
                        "crawling one qubit at a time.", "run_simulation", s)
        # C) at the frontier: attack the dominant error at N=hi with variants
        hi_runs = [r for r in good if r["spec"]["n_qubits"] == hi]
        best_hi = max(hi_runs, key=lambda r: r["fidelity"], default=None)
        variants = []
        if base["layout_strategy"] != "bfs_tree":
            variants.append(({"layout_strategy": "bfs_tree"},
                             "a BFS tree from a central root may be shallower, cutting T2 idle time"))
        if st.bad_qubits and not base["exclude_qubits"]:
            variants.append(({"exclude_qubits": sorted(st.bad_qubits)[:12]},
                             "excluding the worst-calibrated qubits removes the weakest links"))
        if hi <= 12 and base["circuit_variant"] != "linear":
            variants.append(({"circuit_variant": "linear", "layout_strategy": "best_chain"},
                             "at small N a low-error chain might beat a tree despite its depth"))
        if best_hi is not None and best_hi["fidelity"] > 0.5:
            variants.append(({"shots": 6000},
                             f"F={best_hi['fidelity']:.3f} is above 0.5 but the CI is too wide; "
                             "more shots shrink it"))
        for change, why in variants:
            cfg = dict(base)
            cfg.update(change)
            s = spec_for(hi, cfg, why)
            if fresh(s):
                return (f"N={hi} is the frontier (record N={lo}). Variant test: {why}.",
                        "run_simulation", s)
        return None

    # ------------------------------------------------------------------ hardware phase
    def _hw_step(self, st: AgentState, sim: list[dict]):
        hw = [r for r in st.history if (r.get("spec") or {}).get("mode") == "qpu"]
        if len(hw) >= self.max_hw:
            return None
        hw_keys = {config_key(r["spec"]) for r in hw}
        if any(r.get("certified") for r in hw):
            return None  # largest-first ordering means we're done once one certifies
        hw_fail_n = min((r["spec"]["n_qubits"] for r in hw), default=None)
        cands = sorted((r for r in sim if _ok(r) and r.get("certified")),
                       key=lambda r: (-r["spec"]["n_qubits"], -r["fidelity"]))
        seen = set()
        for r in cands:
            k = config_key(r["spec"])
            if k in seen or k in hw_keys or k in st.extra.get("hw_refused", set()):
                continue
            seen.add(k)
            n = r["spec"]["n_qubits"]
            # step down decisively after a hardware failure (hardware time is precious)
            if hw_fail_n is not None and n > hw_fail_n - max(2, round(0.12 * hw_fail_n)):
                continue
            s = dict(r["spec"])
            s["shots"] = self.hw_shots
            s.pop("mode", None)
            s["hypothesis"] = (f"Sim certified N={n} ({r['exp_id']}, F={r['fidelity']:.3f}); "
                               "confirming on real hardware.")
            est = st.estimate_qpu(dict(s, mode="qpu")) if st.estimate_qpu else 0.0
            if est > st.qpu_remaining_s:
                continue
            if hw:
                text = (f"Hardware failed at N={hw_fail_n}; step down to the next sim-certified "
                        f"config, N={n} (~{est:.0f}s of QPU).")
            else:
                text = (f"Simulator frontier is N={n}. Spend ~{est:.0f}s of the QPU budget to "
                        f"confirm it on real hardware with {self.hw_shots} shots.")
            return text, "run_on_hardware", s
        return None

    # ------------------------------------------------------------------ summary
    @staticmethod
    def _summary(st: AgentState) -> str:
        parts = []
        for mode, label in (("sim", "simulator"), ("qpu", "hardware")):
            rs = [r for r in st.history if (r.get("spec") or {}).get("mode") == mode
                  and r.get("certified")]
            if rs:
                b = max(rs, key=lambda r: (r["spec"]["n_qubits"], r["fidelity"]))
                parts.append(f"Best certified on {label}: N={b['spec']['n_qubits']} "
                             f"({b['exp_id']}, F={b['fidelity']:.3f}, CI low {b['ci_low']:.3f}).")
            elif mode == "sim" or st.mode == "qpu":
                parts.append(f"No certified GHZ state on {label}.")
        parts.append(f"{len(st.history)} experiments run by the scripted policy.")
        return " ".join(parts)

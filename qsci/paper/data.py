"""Load a run directory and compute every number the paper quotes.

Nothing in the paper text is typed by hand: all figures quoted in prose come from `RunFacts`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from qsci.events import read_events


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class Experiment:
    exp_id: str
    mode: str
    n: int
    raw: dict
    started: datetime | None = None
    finished: datetime | None = None

    @property
    def spec(self) -> dict:
        return self.raw.get("spec") or {}

    @property
    def fidelity(self) -> float | None:
        return self.raw.get("fidelity")

    @property
    def ci_low(self) -> float | None:
        return self.raw.get("ci_low")

    @property
    def ci_high(self) -> float | None:
        return self.raw.get("ci_high")

    @property
    def certified(self) -> bool:
        lo = self.ci_low
        return bool(self.raw.get("certified")) if lo is None else bool(lo > 0.5)

    @property
    def error(self) -> str | None:
        return self.raw.get("error")

    @property
    def ok(self) -> bool:
        return self.error is None and self.fidelity is not None


@dataclass
class RunFacts:
    run_dir: Path
    run_id: str
    calibration: dict
    events: list[dict]
    experiments: list[Experiment]
    budget: dict | None
    run_started: dict = field(default_factory=dict)
    run_finished: dict = field(default_factory=dict)
    thoughts: list[dict] = field(default_factory=list)
    batch: bool = False                      # queue-aware batch run (scripts/qpu_batch.py)
    pending: list[dict] = field(default_factory=list)

    # ---- convenience selectors -------------------------------------------------
    def by_mode(self, mode: str) -> list[Experiment]:
        return [e for e in self.experiments if e.mode == mode]

    def record(self, mode: str) -> Experiment | None:
        """Largest certified N in a mode; ties broken by higher CI lower bound."""
        cands = [e for e in self.by_mode(mode) if e.ok and e.certified]
        if not cands:
            return None
        return max(cands, key=lambda e: (e.n, e.ci_low or 0))

    def record_overall(self) -> Experiment | None:
        return self.record("qpu") or self.record("sim")

    @property
    def backend(self) -> str:
        return self.run_started.get("backend") or self.calibration.get("backend") or "unknown backend"

    @property
    def run_mode(self) -> str:
        return self.run_started.get("mode") or ("qpu" if self.by_mode("qpu") else "sim")

    @property
    def started_at(self) -> datetime | None:
        return _parse_ts(self.events[0]["ts"]) if self.events else None

    @property
    def finished_at(self) -> datetime | None:
        for ev in reversed(self.events):
            if ev.get("type") == "run_finished":
                return _parse_ts(ev.get("ts"))
        return _parse_ts(self.events[-1]["ts"]) if self.events else None

    @property
    def duration_hours(self) -> float | None:
        a, b = self.started_at, self.finished_at
        return (b - a).total_seconds() / 3600 if a and b else None

    @property
    def qpu_seconds_used(self) -> float:
        if self.budget and self.budget.get("qpu_seconds") is not None:
            return float(self.budget["qpu_seconds"])
        return float(sum((e.raw.get("qpu_seconds") or 0) for e in self.by_mode("qpu")))

    @property
    def max_qpu_seconds(self) -> float | None:
        b = self.budget or {}
        return b.get("max_qpu_seconds") or self.run_started.get("max_qpu_seconds")

    @property
    def llm_calls(self) -> int | None:
        b = self.budget or {}
        if b.get("llm_calls") is not None:
            return int(b["llm_calls"])
        last = [e for e in self.events if e.get("type") == "budget"]
        return int(last[-1]["data"].get("llm_calls", 0)) if last else None

    @property
    def max_llm_calls(self) -> int | None:
        b = self.budget or {}
        return b.get("max_llm_calls") or self.run_started.get("max_llm_calls")

    # ---- derived statistics --------------------------------------------------
    def calib_stats(self) -> dict:
        qs = self.calibration.get("qubits") or []
        es = self.calibration.get("edges") or []

        def med(vals):
            vals = sorted(v for v in vals if v is not None and not math.isnan(v))
            if not vals:
                return None
            m = len(vals) // 2
            return vals[m] if len(vals) % 2 else 0.5 * (vals[m - 1] + vals[m])

        two_q = [e.get("error") for e in es if e.get("error") is not None and e.get("error") < 0.5]
        return {
            "n_qubits": self.calibration.get("num_qubits") or len(qs),
            "n_edges": len(es),
            "median_t1": med([q.get("t1_us") for q in qs]),
            "median_t2": med([q.get("t2_us") for q in qs]),
            "median_ro": med([q.get("readout_error") for q in qs]),
            "median_sx": med([q.get("sx_error") for q in qs]),
            "median_2q": med(two_q),
            "gate": (es[0].get("gate") if es else None) or "cz",
            "source": self.calibration.get("source"),
            "timestamp": self.calibration.get("timestamp"),
        }

    def record_calib(self, exp: Experiment) -> dict:
        """Calibration numbers restricted to the qubits/edges used by one experiment."""
        qs = {q["id"]: q for q in self.calibration.get("qubits") or []}
        emap = {}
        for e in self.calibration.get("edges") or []:
            emap[(e["a"], e["b"])] = e.get("error")
            emap[(e["b"], e["a"])] = e.get("error")
        used = exp.raw.get("physical_qubits") or []
        ro = [qs[q].get("readout_error") for q in used if q in qs and qs[q].get("readout_error") is not None]
        tw = [emap.get((a, b)) for a, b in (exp.raw.get("tree_edges") or []) if emap.get((a, b)) is not None]
        return {
            "mean_ro": sum(ro) / len(ro) if ro else None,
            "mean_2q": sum(tw) / len(tw) if tw else None,
            "sum_2q": sum(tw) if tw else None,
        }

    def mitigation_summary(self) -> dict:
        ok = [e for e in self.experiments if e.spec]
        n = len(ok) or 1
        hw = [e for e in ok if e.mode == "qpu"]
        nh = len(hw)
        return {
            "hw_dd": (sum(bool(e.spec.get("dynamical_decoupling")) for e in hw) / nh) if nh else None,
            "hw_twirl": (sum(bool(e.spec.get("measure_twirling")) for e in hw) / nh) if nh else None,
            "dd": sum(bool(e.spec.get("dynamical_decoupling")) for e in ok) / n,
            "twirl": sum(bool(e.spec.get("measure_twirling")) for e in ok) / n,
            "rem": sum(bool(e.spec.get("readout_mitigation")) for e in ok) / n,
            "shots": sorted({e.spec.get("shots") for e in ok if e.spec.get("shots")}),
            "strategies": sorted({e.spec.get("layout_strategy") for e in ok if e.spec.get("layout_strategy")}),
            "variants": sorted({e.spec.get("circuit_variant") for e in ok if e.spec.get("circuit_variant")}),
        }

    def sim_hw_gap(self) -> list[tuple[int, float, float]]:
        """(N, best sim F, best hw F) for every N attempted successfully in both modes."""
        out = []
        sims = {}
        for e in self.by_mode("sim"):
            if e.ok:
                sims[e.n] = max(sims.get(e.n, -1), e.fidelity)
        hws = {}
        for e in self.by_mode("qpu"):
            if e.ok:
                hws[e.n] = max(hws.get(e.n, -1), e.fidelity)
        for n in sorted(set(sims) & set(hws)):
            out.append((n, sims[n], hws[n]))
        return out

    def twin_pairs(self) -> list[tuple[Experiment, Experiment]]:
        """(hardware experiment, its digital-twin prediction) for every successful hardware run."""
        twins = {e.raw.get("twin_of"): e for e in self.experiments if e.raw.get("twin_of")}
        return [(e, twins[e.exp_id]) for e in self.by_mode("qpu") if e.ok and e.exp_id in twins]

    def layout_pairs(self) -> list[dict]:
        """Matched-N hardware comparisons of the error-aware tree vs the textbook chain."""
        twins = {e.raw.get("twin_of"): e for e in self.experiments if e.raw.get("twin_of")}
        by = {}
        for e in self.by_mode("qpu"):
            if e.ok:
                by.setdefault(e.n, {})[e.spec.get("layout_strategy")] = e
        out = []
        for n in sorted(by):
            t, c = by[n].get("weighted_tree"), by[n].get("best_chain")
            if t and c:
                out.append({"n": n, "tree": t, "chain": c, "tree_twin": twins.get(t.exp_id), "chain_twin": twins.get(c.exp_id)})
        return out

    def gated_fraction(self) -> tuple[int, int]:
        """How many hardware experiments were preceded by a simulation at >= the same N."""
        gated = total = 0
        sim_max = 0
        for e in self.experiments:  # chronological
            if e.mode == "sim" and e.ok:
                sim_max = max(sim_max, e.n)
            elif e.mode == "qpu":
                total += 1
                gated += int(sim_max >= e.n)
        return gated, total

    def decay_fit(self, mode: str) -> tuple[float, float] | None:
        """Fit F(N) = F0 * exp(-N / N_e) to successful runs (log-linear least squares)."""
        pts = [(e.n, e.fidelity) for e in self.by_mode(mode) if e.ok and e.fidelity and e.fidelity > 0]
        if len({n for n, _ in pts}) < 2:
            return None
        xs = [n for n, _ in pts]
        ys = [math.log(f) for _, f in pts]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        if slope >= 0:
            return None
        f0 = math.exp(my - slope * mx)
        return f0, -1.0 / slope

    def predicted_threshold_n(self, mode: str) -> float | None:
        fit = self.decay_fit(mode)
        if not fit:
            return None
        f0, ne = fit
        return ne * math.log(f0 / 0.5) if f0 > 0.5 else None


def load_run(run_dir: Path) -> RunFacts:
    run_dir = Path(run_dir)
    cal_path = run_dir / "calibration.json"
    calibration = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    events = read_events(run_dir)
    budget = None
    if (run_dir / "budget.json").exists():
        try:
            budget = json.loads((run_dir / "budget.json").read_text())
        except json.JSONDecodeError:
            budget = None

    started_ts, finished_ts = {}, {}
    for ev in events:
        d = ev.get("data") or {}
        if ev.get("type") == "experiment_started" and d.get("exp_id"):
            started_ts[d["exp_id"]] = _parse_ts(ev.get("ts"))
        elif ev.get("type") == "experiment_result" and d.get("exp_id"):
            finished_ts[d["exp_id"]] = _parse_ts(ev.get("ts"))

    raw_by_id: dict[str, dict] = {}
    exp_dir = run_dir / "experiments"
    if exp_dir.exists():
        for p in sorted(exp_dir.glob("*.json")):
            try:
                d = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            raw_by_id[d.get("exp_id") or p.stem] = d
    # experiments that only exist in the event log (e.g. file not written yet)
    for ev in events:
        d = ev.get("data") or {}
        if ev.get("type") == "experiment_result" and d.get("exp_id") and d["exp_id"] not in raw_by_id:
            raw_by_id[d["exp_id"]] = {**d, "spec": {"n_qubits": d.get("n_qubits"), "mode": d.get("mode")}}

    pending = []
    if (run_dir / "pending.json").exists():
        try:
            pending = json.loads((run_dir / "pending.json").read_text())
        except json.JSONDecodeError:
            pending = []
        for entry in pending:  # digital-twin predictions become simulator experiments "<id>-twin"
            pred = entry.get("predicted") or {}
            if pred.get("fidelity") is None:
                continue
            tid = f"{entry['exp_id']}-twin"
            raw_by_id[tid] = {"exp_id": tid, "twin_of": entry["exp_id"],
                              "spec": {**entry["spec"], "mode": "sim"}, "fidelity": pred["fidelity"],
                              "ci_low": pred.get("ci_low"), "ci_high": pred.get("ci_high"),
                              "certified": pred.get("certified"), "physical_qubits": (entry.get("plan") or {}).get("qubits"),
                              "circuit_depth": entry.get("circuit_depth"), "sim_method": "pauli_frame (digital twin, live calibration)"}
            started_ts[tid] = started_ts.get(entry["exp_id"])

    exps = []
    for exp_id, d in raw_by_id.items():
        spec = d.get("spec") or {}
        mode = spec.get("mode") or d.get("mode") or "sim"
        n = spec.get("n_qubits") or d.get("n_qubits") or len(d.get("physical_qubits") or [])
        exps.append(Experiment(exp_id, mode, int(n), d, started_ts.get(exp_id), finished_ts.get(exp_id)))
    far = datetime.max.replace(tzinfo=None)
    exps.sort(key=lambda e: ((e.started or e.finished).replace(tzinfo=None) if (e.started or e.finished) else far, e.exp_id))

    def first(t):
        for ev in events:
            if ev.get("type") == t:
                return ev.get("data") or {}
        return {}

    def last(t):
        for ev in reversed(events):
            if ev.get("type") == t:
                return ev.get("data") or {}
        return {}

    rs = first("run_started")
    run_id = rs.get("run_id") or run_dir.name
    return RunFacts(
        run_dir=run_dir, run_id=run_id, calibration=calibration, events=events,
        experiments=exps, budget=budget, run_started=rs, run_finished=last("run_finished"),
        thoughts=[ev for ev in events if ev.get("type") == "thought" and (ev.get("data") or {}).get("text")],
        batch=bool(pending), pending=pending,
    )

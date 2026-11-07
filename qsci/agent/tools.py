"""Tools exposed to the agent + their executor (ToolBox).

ToolBox owns everything with side effects: running experiments through qsci.core, writing
experiments/<exp_id>.json, emitting experiment/budget/milestone events, charging the QPU budget
and enforcing the hardware gate. Tool errors never raise out of ToolBox.call(); they come back
as {"error": "..."} so the model can read them and adapt.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from .policy import DEFAULT_CONFIG, config_key
from .prompts import format_ci, history_row

STRATEGIES = ["weighted_tree", "bfs_tree", "best_chain"]
VARIANTS = ["tree", "linear"]

_SPEC_PROPS = {
    "n_qubits": {"type": "integer", "description": "Number of GHZ qubits N (>= 2)."},
    "layout_strategy": {"type": "string", "enum": STRATEGIES,
                        "description": "How physical qubits are chosen. weighted_tree = "
                        "calibration-weighted low-error tree (default); bfs_tree = BFS from root; "
                        "best_chain = lowest-error linear chain."},
    "root": {"type": "integer", "description": "Optional physical root qubit; omit for auto."},
    "exclude_qubits": {"type": "array", "items": {"type": "integer"},
                       "description": "Physical qubits to avoid."},
    "circuit_variant": {"type": "string", "enum": VARIANTS,
                        "description": "tree = fan-out from root (shallow); linear = CNOT chain."},
    "dynamical_decoupling": {"type": "boolean", "description": "DD on idle qubits (default true)."},
    "measure_twirling": {"type": "boolean", "description": "Measurement twirling (default true)."},
    "readout_mitigation": {"type": "boolean", "description": "Readout mitigation (default true)."},
    "shots": {"type": "integer", "description": "Shots per circuit (sim 500-8000, hw 1000-10000)."},
    "hypothesis": {"type": "string", "description": "What you expect and why (one sentence)."},
}


def tool_declarations(mode: str) -> list[dict]:
    spec_schema = {"type": "object", "properties": _SPEC_PROPS, "required": ["n_qubits", "hypothesis"]}
    decls = [
        {"name": "get_calibration_summary",
         "description": "Compact calibration summary of the chip: medians, best qubits/edges, "
                        "suggested roots and qubits to exclude."},
        {"name": "preview_layout",
         "description": "Free (no experiment): plan a layout and report its depth, qubits and "
                        "error stats, to compare strategies/roots before running.",
         "parameters": {"type": "object", "properties": {
             k: _SPEC_PROPS[k] for k in ("n_qubits", "layout_strategy", "root", "exclude_qubits")},
             "required": ["n_qubits"]}},
        {"name": "run_simulation",
         "description": "Run one GHZ fidelity experiment on the noisy simulator of this chip.",
         "parameters": spec_schema},
        {"name": "get_history", "description": "Compact table of all experiments so far."},
        {"name": "finish",
         "description": "End the run with an honest summary of what was certified (sim vs hardware).",
         "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},
                        "required": ["summary"]}},
    ]
    if mode == "qpu":
        decls.insert(3, {
            "name": "run_on_hardware",
            "description": "Run one GHZ experiment on REAL IBM hardware. Costs QPU seconds from a "
                           "hard budget. Refused unless the same config (N, layout, root, "
                           "exclusions, variant, mitigation flags) was validated on the simulator "
                           "with certified=true or fidelity >= 0.5.",
            "parameters": spec_schema})
    return decls


# ------------------------------------------------------------------ calibration summary

def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def summarize_calibration(cal: dict) -> dict:
    qubits = cal.get("qubits") or []
    edges = cal.get("edges") or []
    m_ro = _med(q.get("readout_error") for q in qubits)
    m_t2 = _med(q.get("t2_us") for q in qubits)
    m_t1 = _med(q.get("t1_us") for q in qubits)
    m_sx = _med(q.get("sx_error") for q in qubits)
    m_e = _med(e.get("error") for e in edges)
    edge_err: dict[int, list[float]] = {}
    for e in edges:
        err = e.get("error")
        if isinstance(err, (int, float)):
            edge_err.setdefault(e["a"], []).append(err)
            edge_err.setdefault(e["b"], []).append(err)
    bad = set()
    for q in qubits:
        ro, t2 = q.get("readout_error"), q.get("t2_us")
        if ro is None or t2 is None:
            bad.add(q["id"])
        elif m_ro and ro > max(0.05, 3 * m_ro):
            bad.add(q["id"])
        elif m_t2 and t2 < 0.3 * m_t2:
            bad.add(q["id"])
    bad_edges = []
    for e in edges:
        err = e.get("error")
        if err is None or err >= 0.5 or (m_e and err > max(0.02, 3 * m_e)):
            bad_edges.append([e["a"], e["b"], None if err is None else round(err, 4)])
    for q, errs in edge_err.items():
        if errs and min(errs) > max(0.02, 3 * (m_e or 0.01)):
            bad.add(q)

    def score(q):
        errs = edge_err.get(q["id"], [1.0])
        return (q.get("readout_error") or 1) + statistics.mean(errs) * 2 - min((q.get("t2_us") or 0), 300) / 3e4

    ranked = sorted((q for q in qubits if q["id"] not in bad), key=score)
    roots = [q["id"] for q in ranked if len(edge_err.get(q["id"], [])) >= 3][:6]
    best_edges = sorted((e for e in edges if isinstance(e.get("error"), (int, float))),
                        key=lambda e: e["error"])[:10]
    r = lambda x, d=4: None if x is None else round(x, d)
    return {
        "backend": cal.get("backend"), "num_qubits": cal.get("num_qubits") or len(qubits),
        "source": cal.get("source"),
        "median": {"t1_us": r(m_t1, 1), "t2_us": r(m_t2, 1), "readout_error": r(m_ro),
                   "sx_error": r(m_sx, 5), "cz_error": r(m_e)},
        "bad_qubits": sorted(bad)[:40],
        "bad_edges": bad_edges[:20],
        "best_qubits": [q["id"] for q in ranked[:15]],
        "suggested_roots": roots,
        "best_edges": [[e["a"], e["b"], round(e["error"], 4)] for e in best_edges],
    }


# ------------------------------------------------------------------ spec normalisation

def normalize_spec(args: dict, mode: str, num_qubits: int) -> tuple[dict | None, str | None]:
    if not isinstance(args, dict):
        return None, "arguments must be an object"
    if "__malformed__" in args:
        return None, "malformed function-call arguments"
    s = dict(DEFAULT_CONFIG)
    try:
        n = int(args.get("n_qubits"))
    except (TypeError, ValueError):
        return None, "n_qubits must be an integer"
    if not 2 <= n <= num_qubits:
        return None, f"n_qubits must be between 2 and {num_qubits}"
    s["n_qubits"] = n
    strat = args.get("layout_strategy", s["layout_strategy"]) or s["layout_strategy"]
    if strat not in STRATEGIES:
        return None, f"layout_strategy must be one of {STRATEGIES}"
    s["layout_strategy"] = strat
    var = args.get("circuit_variant", s["circuit_variant"]) or s["circuit_variant"]
    if var not in VARIANTS:
        return None, f"circuit_variant must be one of {VARIANTS}"
    s["circuit_variant"] = var
    root = args.get("root")
    if root is not None:
        try:
            root = int(root)
        except (TypeError, ValueError):
            return None, "root must be an integer or omitted"
        if not 0 <= root < num_qubits:
            return None, "root out of range"
    s["root"] = root
    excl = args.get("exclude_qubits") or []
    try:
        excl = sorted({int(q) for q in excl})
    except (TypeError, ValueError):
        return None, "exclude_qubits must be a list of integers"
    if root is not None and root in excl:
        return None, "root is in exclude_qubits"
    s["exclude_qubits"] = excl
    for k in ("dynamical_decoupling", "measure_twirling", "readout_mitigation"):
        if k in args and args[k] is not None:
            s[k] = bool(args[k])
    lo, hi = (500, 8000) if mode == "sim" else (1000, 10000)
    try:
        shots = int(args.get("shots") or DEFAULT_CONFIG["shots"])
    except (TypeError, ValueError):
        shots = DEFAULT_CONFIG["shots"]
    s["shots"] = max(lo, min(hi, shots))
    s["mode"] = mode
    s["hypothesis"] = str(args.get("hypothesis") or "")[:400]
    return s, None


def _compact_result(r: dict) -> dict:
    keys = ("exp_id", "fidelity", "ci_low", "ci_high", "certified", "populations", "coherence",
            "circuit_depth", "two_qubit_gate_count", "root", "qpu_seconds", "error")
    out = {k: r.get(k) for k in keys}
    for k in ("fidelity", "ci_low", "ci_high", "populations", "coherence"):
        if isinstance(out[k], float):
            out[k] = round(out[k], 4)
    out["n_qubits"] = (r.get("spec") or {}).get("n_qubits")
    out["mode"] = (r.get("spec") or {}).get("mode")
    out["physical_qubits"] = r.get("physical_qubits")
    return out


# ------------------------------------------------------------------ executor

class ToolBox:
    def __init__(self, *, core, backend, calibration: dict, mode: str, budget, events,
                 run_dir: Path, max_experiments: int = 80,
                 log: Callable[[str], None] = print) -> None:
        self.core = core
        self.backend = backend
        self.cal = calibration
        self.mode = mode
        self.budget = budget
        self.events = events
        self.run_dir = Path(run_dir)
        self.max_experiments = max_experiments
        self.log = log
        self.num_qubits = int(calibration.get("num_qubits") or len(calibration.get("qubits") or []) or 156)
        self.history: list[dict] = []
        self.cal_summary: dict | None = None
        self.calibration_seen = False
        self.best: dict[str, dict | None] = {"sim": None, "qpu": None}
        self.hw_refused: set = set()
        self.finished_summary: str | None = None
        self._finish_refusals = 0

    # -------------------------------------------------- helpers
    def emit_budget(self) -> None:
        self.events.emit("budget", **{k: self.budget.snapshot()[k] for k in
                                      ("llm_calls", "max_llm_calls", "qpu_seconds", "max_qpu_seconds")})

    def experiments_left(self) -> int:
        return self.max_experiments - len(self.history)

    def calibration_summary(self) -> dict:
        if self.cal_summary is None:
            try:
                self.cal_summary = summarize_calibration(self.cal)
            except Exception as e:  # odd calibration shape -> minimal summary
                self.cal_summary = {"num_qubits": self.num_qubits, "error": f"summary failed: {e}"}
        return self.cal_summary

    def bad_qubits(self) -> list[int]:
        return list(self.calibration_summary().get("bad_qubits") or [])

    def estimate_qpu(self, spec: dict) -> float:
        return float(self.core.estimate_qpu_seconds(spec))

    def sim_validated(self, spec: dict) -> dict | None:
        k = config_key(spec)
        for r in self.history:
            s = r.get("spec") or {}
            if s.get("mode") == "sim" and not r.get("error") and config_key(s) == k:
                if r.get("certified") or (r.get("fidelity") or 0) >= 0.5:
                    return r
        return None

    def next_exp_id(self) -> str:
        return f"e{len(self.history) + 1:03d}"

    # -------------------------------------------------- dispatch
    def call(self, name: str, args: dict) -> dict:
        args = args if isinstance(args, dict) else {}
        try:
            if name == "get_calibration_summary":
                self.calibration_seen = True
                return self.calibration_summary()
            if name == "preview_layout":
                return self._preview(args)
            if name == "get_history":
                return {"experiments": [history_row(r) for r in self.history[-40:]],
                        "total": len(self.history)}
            if name == "run_simulation":
                return self._run(args, "sim")
            if name == "run_on_hardware":
                if self.mode != "qpu":
                    return {"error": "run_on_hardware is not available in sim mode"}
                return self._run(args, "qpu")
            if name == "finish":
                return self._finish(args)
            return {"error": f"unknown tool '{name}'"}
        except Exception as e:  # a tool must never take the night down
            msg = f"{name} failed: {type(e).__name__}: {str(e)[:300]}"
            self.events.emit("error", text=msg)
            return {"error": msg}

    def _preview(self, args: dict) -> dict:
        a = dict(args)
        a.setdefault("hypothesis", "preview")
        spec, err = normalize_spec(a, "sim", self.num_qubits)
        if err:
            return {"error": err}
        lay = self.core.plan_layout(self.cal, spec["n_qubits"], spec["layout_strategy"],
                                    root=spec["root"], exclude=spec["exclude_qubits"])
        qs = set(lay.get("qubits") or [])
        qinfo = {q["id"]: q for q in self.cal.get("qubits") or []}
        ro = [qinfo[q].get("readout_error") for q in qs if q in qinfo and qinfo[q].get("readout_error") is not None]
        emap = {}
        for e in self.cal.get("edges") or []:
            emap[(e["a"], e["b"])] = e.get("error")
            emap[(e["b"], e["a"])] = e.get("error")
        tree_err = [emap.get(tuple(t)) for t in lay.get("tree_edges") or []]
        tree_err = [x for x in tree_err if isinstance(x, (int, float))]
        return {"n_qubits": spec["n_qubits"], "layout_strategy": spec["layout_strategy"],
                "root": lay.get("root"), "depth": lay.get("depth"), "qubits": sorted(qs),
                "mean_readout_error": round(statistics.mean(ro), 4) if ro else None,
                "max_readout_error": round(max(ro), 4) if ro else None,
                "mean_cz_error": round(statistics.mean(tree_err), 4) if tree_err else None,
                "max_cz_error": round(max(tree_err), 4) if tree_err else None}

    def _finish(self, args: dict) -> dict:
        summary = str(args.get("summary") or "").strip() or "Run finished."
        if self.mode == "qpu" and self._finish_refusals == 0:
            hw = [r for r in self.history if (r.get("spec") or {}).get("mode") == "qpu"]
            if not hw and self.budget.max_qpu_seconds - self.budget.qpu_seconds > 30 and self.sim_any_validated():
                self._finish_refusals += 1
                return {"error": "finish declined once: no hardware run yet while QPU budget "
                                 "remains and you have sim-validated configs. Confirm on hardware "
                                 "with run_on_hardware, or call finish again to end anyway."}
        self.finished_summary = summary[:2000]
        return {"ok": True}

    def sim_any_validated(self) -> bool:
        return any((r.get("spec") or {}).get("mode") == "sim" and not r.get("error")
                   and (r.get("certified") or (r.get("fidelity") or 0) >= 0.5) for r in self.history)

    # -------------------------------------------------- experiments
    def _run(self, args: dict, mode: str) -> dict:
        if self.experiments_left() <= 0:
            return {"error": f"experiment cap reached ({self.max_experiments}); call finish"}
        spec, err = normalize_spec(args, mode, self.num_qubits)
        if err:
            return {"error": f"invalid spec: {err}"}
        est = 0.0
        if mode == "qpu":
            if self.sim_validated(spec) is None:
                self.hw_refused.add(config_key(spec))
                return {"error": "refused: this exact config (N, layout_strategy, root, "
                                 "exclude_qubits, circuit_variant, mitigation flags) has not been "
                                 "validated on the simulator with certified=true or fidelity>=0.5. "
                                 "Run it with run_simulation first."}
            try:
                est = self.estimate_qpu(spec)
            except Exception as e:
                return {"error": f"refused: could not estimate QPU time ({e})"}
            if not self.budget.can_spend_qpu(est):
                self.hw_refused.add(config_key(spec))
                snap = self.budget.snapshot()
                left = snap["max_qpu_seconds"] - snap["qpu_seconds"]
                return {"error": f"refused: estimated {est:.1f}s exceeds remaining QPU budget "
                                 f"({left:.1f}s). Use fewer shots, smaller N, or finish."}
        if not spec["hypothesis"]:
            spec["hypothesis"] = f"{mode} test at N={spec['n_qubits']}"
        exp_id = self.next_exp_id()
        self.events.emit("experiment_started", exp_id=exp_id, spec=spec)
        tag = "QPU" if mode == "qpu" else "sim"
        self.log(f"  [{exp_id}] {tag} N={spec['n_qubits']} {spec['layout_strategy']}/"
                 f"{spec['circuit_variant']} shots={spec['shots']}"
                 + (f" (est {est:.0f}s QPU)" if mode == "qpu" else "") + " ...")
        t0 = time.time()
        try:
            result = self.core.run_experiment(spec, self.backend, self.cal, exp_id)
            if not isinstance(result, dict):
                raise TypeError(f"run_experiment returned {type(result).__name__}")
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
        result = self._complete_result(result, spec, exp_id, mode, time.time() - t0)
        if mode == "qpu":
            # charge actual usage; if unknown (failed/odd job) charge the conservative estimate
            used = result.get("qpu_seconds")
            if not isinstance(used, (int, float)) or (result.get("error") and not used):
                used = est
            self.budget.charge_qpu(float(used))
        self.history.append(result)
        self._write_result(result)
        self.events.emit("experiment_result", exp_id=exp_id, n_qubits=spec["n_qubits"], mode=mode,
                         fidelity=result.get("fidelity"), ci_low=result.get("ci_low"),
                         ci_high=result.get("ci_high"), certified=bool(result.get("certified")),
                         physical_qubits=result.get("physical_qubits"),
                         tree_edges=result.get("tree_edges"), root=result.get("root"),
                         qpu_seconds=result.get("qpu_seconds"), error=result.get("error"))
        if result.get("error"):
            self.events.emit("error", text=f"{exp_id} failed: {str(result['error'])[:300]}")
            self.log(f"  [{exp_id}] ERROR {str(result['error'])[:120]}")
        else:
            self.log(f"  [{exp_id}] {format_ci(result)} "
                     f"{'CERTIFIED' if result.get('certified') else 'not certified'}")
        self._check_milestone(result, mode)
        self.emit_budget()
        return _compact_result(result)

    def _complete_result(self, r: dict, spec: dict, exp_id: str, mode: str, wall: float) -> dict:
        r = dict(r)
        r["exp_id"] = exp_id
        r["spec"] = spec
        for k in ("physical_qubits", "tree_edges", "root", "circuit_depth", "two_qubit_gate_count",
                  "fidelity", "ci_low", "ci_high", "populations", "coherence", "parity_curve",
                  "backend", "error"):
            r.setdefault(k, None)
        r.setdefault("job_ids", [])
        r.setdefault("qpu_seconds", 0.0 if mode == "sim" else None)
        r.setdefault("wall_seconds", round(wall, 2))
        # Certification is recomputed here: ci_low > 0.5, nothing else counts.
        ci_low = r.get("ci_low")
        r["certified"] = bool(not r.get("error") and isinstance(ci_low, (int, float)) and ci_low > 0.5)
        return r

    def _write_result(self, r: dict) -> None:
        path = self.run_dir / "experiments" / f"{r['exp_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(r, indent=2, default=str))
        except Exception as e:
            self.events.emit("error", text=f"could not write {path.name}: {e}")

    def _check_milestone(self, r: dict, mode: str) -> None:
        if not r.get("certified"):
            return
        n = r["spec"]["n_qubits"]
        prev = self.best[mode]
        if prev is None or n > prev["spec"]["n_qubits"]:
            self.best[mode] = r
            where = "real hardware" if mode == "qpu" else "simulator"
            text = (f"New {where} record: {n}-qubit GHZ entanglement certified "
                    f"({format_ci(r)}, {r['exp_id']})")
            self.events.emit("milestone", text=text, n_qubits=n, fidelity=r.get("fidelity"))
            self.log(f"  *** {text}")
        elif n == prev["spec"]["n_qubits"] and (r.get("fidelity") or 0) > (prev.get("fidelity") or 0):
            self.best[mode] = r  # same N, better fidelity: keep as best, no new milestone

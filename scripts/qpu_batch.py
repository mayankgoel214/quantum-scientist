"""Queue-aware hardware run: the agent designs a batch, every job is submitted at once, results are
collected whenever IBM finishes them. Same run-directory / event format as the agent loop.

Why: the open-plan queue on ibm_fez can hold 20k+ jobs, so the blocking one-experiment-at-a-time
loop could finish only one or two experiments before the deadline. Submitting the whole batch lets
every experiment queue in parallel.

    uv run python -m scripts.qpu_batch submit [--backend ibm_fez] [--no-llm] [--dry-run]
    uv run python -m scripts.qpu_batch collect runs/<run_id> [--watch] [--max-hours 9]

Each spec is also run on the Pauli-frame simulator driven by the LIVE calibration (a "digital twin")
before submission, so the paper can compare prediction vs. hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

from qsci.budget import Budget
from qsci.core import estimate_qpu_seconds, get_backend, get_calibration
from qsci.core.execute import (
    _empty_result,
    _qpu_seconds,
    build_qpu_pubs,
    counts_from_result,
    normalize_spec,
    prepare,
    qpu_notes,
    run_sim,
    submit_qpu,
)
from qsci.core.fidelity import analyze
from qsci.events import EventLog, new_run_dir, read_events

GOAL = ("Certify genuine multipartite entanglement (GHZ fidelity CI lower bound > 0.5) on as many qubits "
        "as possible on real IBM hardware, and test whether an error-aware layout beats the textbook "
        "linear-chain GHZ circuit at equal size.")
STRATEGIES = {"weighted_tree", "bfs_tree", "best_chain"}
MAX_EXPERIMENTS = 8
DEFAULT_PLAN = [
    {"n_qubits": 16, "layout_strategy": "weighted_tree", "shots": 1000, "hypothesis": "warm-up: should certify easily"},
    {"n_qubits": 24, "layout_strategy": "best_chain", "shots": 1000, "hypothesis": "textbook linear GHZ baseline at N=24"},
    {"n_qubits": 24, "layout_strategy": "weighted_tree", "shots": 1000, "hypothesis": "error-aware tree should beat the chain at N=24"},
    {"n_qubits": 32, "layout_strategy": "best_chain", "shots": 1000, "hypothesis": "textbook linear GHZ baseline at N=32"},
    {"n_qubits": 32, "layout_strategy": "weighted_tree", "shots": 1000, "hypothesis": "tree advantage should grow with N"},
    {"n_qubits": 40, "layout_strategy": "weighted_tree", "shots": 1000, "hypothesis": "probe the certification frontier"},
    {"n_qubits": 48, "layout_strategy": "weighted_tree", "shots": 1000, "hypothesis": "stretch goal near the frontier"},
]


def _load_env():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def _service():
    from qiskit_ibm_runtime import QiskitRuntimeService

    return QiskitRuntimeService(channel="ibm_quantum_platform",
                                token=(os.environ.get("IBM_QUANTUM_TOKEN") or "").strip(),
                                instance=(os.environ.get("IBM_QUANTUM_INSTANCE") or "").strip() or None)


def _full_spec(e: dict) -> dict:
    chain = e["layout_strategy"] == "best_chain"
    return normalize_spec({
        "n_qubits": int(e["n_qubits"]),
        "layout_strategy": e["layout_strategy"],
        "root": None,
        "exclude_qubits": [],
        "circuit_variant": "linear" if chain else "tree",
        "dynamical_decoupling": True,
        "measure_twirling": True,
        "readout_mitigation": True,
        "shots": int(e.get("shots", 1000)),
        "mode": "qpu",
        "hypothesis": str(e.get("hypothesis", ""))[:300],
    })


def _sim_evidence(root: Path) -> list[dict]:
    """Certified/uncertified simulator results from earlier runs (context for the planner)."""
    out = []
    for f in sorted(root.glob("*-sim/experiments/*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if r.get("fidelity") is None:
            continue
        s = r["spec"]
        out.append({"n": s["n_qubits"], "layout": s["layout_strategy"], "variant": s.get("circuit_variant"),
                    "F": round(r["fidelity"], 3), "ci_low": round(r["ci_low"], 3), "certified": r["certified"]})
    out.sort(key=lambda x: (x["layout"], x["n"]))
    return out[-40:]


def _cal_summary(cal: dict) -> dict:
    import numpy as np

    def stats(xs):
        xs = [x for x in xs if x is not None]
        return {"median": round(float(np.median(xs)), 5), "p90": round(float(np.percentile(xs, 90)), 5)} if xs else None

    return {
        "backend": cal["backend"], "num_qubits": cal["num_qubits"],
        "readout_error": stats([q.get("readout_error") for q in cal["qubits"]]),
        "t2_us": stats([q.get("t2_us") for q in cal["qubits"]]),
        "two_qubit_error": stats([e.get("error") for e in cal["edges"]]),
    }


def gemini_plan(cal: dict, evidence: list[dict], budget_s: float) -> tuple[list[dict], str, str] | None:
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        return None
    model = (os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash").split("#")[0].strip() or "gemini-2.5-flash"
    costs = {f"N={n},shots={sh}": estimate_qpu_seconds({"n_qubits": n, "shots": sh, "mode": "qpu"})
             for n in (16, 24, 32, 40, 48, 56) for sh in (500, 1000)}
    prompt = f"""You are Q-Scientist, an autonomous experimental physicist. Tonight you get ONE batch of jobs on
real IBM quantum hardware ({cal['backend']}). The queue is very long, so every job is submitted at once and you
cannot adapt between jobs: design the batch carefully.

GOAL: {GOAL}

Live calibration summary: {json.dumps(_cal_summary(cal))}
Your earlier simulator results (noise model of the same chip family): {json.dumps(evidence)}
Estimated QPU seconds per experiment: {json.dumps(costs)}
Budget for the whole batch: {budget_s:.0f} QPU seconds (hard cap; the free plan has 600 s total).

Rules: at most {MAX_EXPERIMENTS} experiments. layout_strategy is one of weighted_tree (error-weighted
shallow tree, your method), bfs_tree, best_chain (the textbook linear CNOT ladder baseline). shots 500 or 1000.
Hard requirements (a validator enforces them):
- at least one SAFE anchor with N <= 16 (if every job fails, the night produced nothing);
- a matched pair (same N, between 20 and 32) of best_chain vs weighted_tree, at a size where the tree should
  still certify, so the comparison is informative;
- the rest of the budget brackets the hardware frontier.
IMPORTANT: real hardware is MUCH noisier than this simulator (crosstalk, leakage, drift, non-Markovian noise
are not modelled). Published simple GHZ experiments on Heron-class chips without post-selection typically
certify a few tens of qubits, not 60. Expect the hardware frontier well below your simulator frontier.

Return JSON: {{"rationale": "3-5 sentences on why this batch", "experiments": [{{"n_qubits": int,
"layout_strategy": str, "shots": int, "hypothesis": "one sentence"}}]}}"""
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                  "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4,
                                       "maxOutputTokens": 16384,
                                       "thinkingConfig": {"thinkingBudget": 4096}}},
            timeout=120,
        )
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(text)
        return list(data["experiments"]), str(data.get("rationale", "")), f"gemini:{model}"
    except Exception as e:
        print(f"Gemini planning failed ({type(e).__name__}: {str(e)[:200]}); using default plan")
        return None


def validate_plan(exps: list[dict], budget_s: float) -> tuple[list[dict], list[str]]:
    notes, clean = [], []
    for e in exps:
        try:
            n = int(e["n_qubits"])
            strat = str(e["layout_strategy"])
            shots = int(e.get("shots", 1000))
        except Exception:
            notes.append(f"dropped malformed entry {e!r}"[:200])
            continue
        if strat not in STRATEGIES or not 4 <= n <= 100:
            notes.append(f"dropped invalid entry N={n} {strat}")
            continue
        shots = 500 if shots < 750 else 1000
        clean.append({**e, "n_qubits": n, "layout_strategy": strat, "shots": shots})
    clean = clean[:MAX_EXPERIMENTS]
    if not any(e["n_qubits"] <= 16 for e in clean):
        clean.insert(0, {"n_qubits": 12, "layout_strategy": "weighted_tree", "shots": 1000, "hypothesis": "added: safe anchor"})
        notes.append("added a safe N=12 anchor")
    ns_chain = {e["n_qubits"] for e in clean if e["layout_strategy"] == "best_chain"}
    ns_tree = {e["n_qubits"] for e in clean if e["layout_strategy"] == "weighted_tree"}
    if not {n for n in ns_chain & ns_tree if 20 <= n <= 32}:
        clean.append({"n_qubits": 24, "layout_strategy": "best_chain", "shots": 1000, "hypothesis": "added: chain baseline"})
        if 24 not in ns_tree:
            clean.append({"n_qubits": 24, "layout_strategy": "weighted_tree", "shots": 1000, "hypothesis": "added: tree at N=24"})
        notes.append("added a matched chain-vs-tree pair at N=24")

    def total(xs):
        return sum(estimate_qpu_seconds(_full_spec(x)) for x in xs)

    while clean and total(clean) > budget_s:
        protected = lambda x: x["n_qubits"] <= 16 or (20 <= x["n_qubits"] <= 32 and x["n_qubits"] in ns_chain & ns_tree)
        ns_chain = {e["n_qubits"] for e in clean if e["layout_strategy"] == "best_chain"} | {24}
        ns_tree = {e["n_qubits"] for e in clean if e["layout_strategy"] == "weighted_tree"}
        big = max(clean, key=lambda x: (not protected(x), x["n_qubits"]))
        clean.remove(big)
        notes.append(f"dropped N={big['n_qubits']} {big['layout_strategy']} to fit the budget")
    return clean, notes


# ------------------------------------------------------------------------------------------ submit
def cmd_submit(args) -> Path:
    _load_env()
    root = Path(args.runs_root)
    backend = get_backend("qpu", args.backend)
    cal = get_calibration(backend)
    budget_s = float(args.max_qpu_seconds)

    planned = None if args.no_llm else gemini_plan(cal, _sim_evidence(root), budget_s)
    if planned:
        exps, rationale, planner = planned
    else:
        exps, rationale, planner = DEFAULT_PLAN, "Default batch (no language model).", "fixed-plan"
    exps, vnotes = validate_plan(exps, budget_s)
    specs = [_full_spec(e) for e in exps]
    est_total = sum(estimate_qpu_seconds(s) for s in specs)
    print(f"planner={planner}\n{rationale}\n")
    for i, s in enumerate(specs, 1):
        print(f"  e{i:03d}: N={s['n_qubits']:>3} {s['layout_strategy']:<13} shots={s['shots']} "
              f"est {estimate_qpu_seconds(s):.0f}s  | {s['hypothesis']}")
    print(f"estimated total {est_total:.0f}s of {budget_s:.0f}s budget; notes: {vnotes}")
    if args.dry_run:
        for s in specs:
            try:
                prep = prepare(s, backend, cal)
                counts, _ = run_sim(prep, backend, cal)
                fit = analyze(s["n_qubits"], counts["populations"], counts["parity"], counts.get("cal0"),
                              counts.get("cal1"), mitigate=True)
                print(f"  twin N={s['n_qubits']} {s['layout_strategy']}: F={fit['fidelity']:.3f} ci_low={fit['ci_low']:.3f}")
            except Exception as e:
                print(f"  twin N={s['n_qubits']} {s['layout_strategy']}: {type(e).__name__}: {e}"[:200])
        return root

    run_dir = new_run_dir("qpu", root)
    (run_dir / "calibration.json").write_text(json.dumps(cal, indent=1))
    log = EventLog(run_dir)
    budget = Budget(run_dir, max_llm_calls=1, max_qpu_seconds=budget_s)
    if planner.startswith("gemini"):
        budget.charge_llm()
    log.emit("run_started", run_id=run_dir.name, mode="qpu", backend=cal["backend"], goal=GOAL, planner=planner,
             max_llm_calls=1, max_qpu_seconds=budget_s)
    log.emit("thought", text=("[batch plan] " + rationale)[:1200])
    for n in vnotes:
        log.emit("thought", text=f"[validator] {n}")
    log.emit("budget", **budget.snapshot())

    pending = []
    for i, s in enumerate(specs, 1):
        exp_id = f"e{i:03d}"
        entry = {"exp_id": exp_id, "spec": s, "job_id": None, "status": "new", "submitted_at": None}
        try:
            prep = prepare(s, backend, cal)
            entry["plan"] = prep["plan"]
            entry["circuit_depth"] = prep["circuit_depth"]
            entry["two_qubit_gate_count"] = prep["two_qubit_gate_count"]
            entry["transpiled_depth"] = prep["transpiled_depth"]
            entry["num_pubs"] = len(build_qpu_pubs(prep)[0])
            try:  # digital-twin prediction from the live calibration
                counts, _ = run_sim(prep, backend, cal)
                fit = analyze(s["n_qubits"], counts["populations"], counts["parity"], counts.get("cal0"),
                              counts.get("cal1"), mitigate=True)
                entry["predicted"] = {k: fit[k] for k in ("fidelity", "ci_low", "ci_high", "certified")}
            except Exception as e:
                entry["predicted"] = {"error": f"{type(e).__name__}: {e}"[:200]}
            job = submit_qpu(prep, backend)
            entry["job_id"] = job.job_id()
            entry["status"] = "submitted"
            entry["submitted_at"] = time.time()
            log.emit("experiment_started", exp_id=exp_id, spec=s)
            pred = entry["predicted"].get("fidelity")
            print(f"  submitted {exp_id} N={s['n_qubits']} {s['layout_strategy']} job={entry['job_id']} "
                  f"(twin predicts F={pred if pred is None else round(pred, 3)})")
        except Exception as e:
            entry["status"] = "failed_submit"
            entry["error"] = f"{type(e).__name__}: {e}"[:500]
            log.emit("error", text=f"{exp_id} submit failed: {entry['error']}")
            print(f"  {exp_id} submit FAILED: {entry['error']}")
        pending.append(entry)
        (run_dir / "pending.json").write_text(json.dumps(pending, indent=1, default=str))
    print(f"\nrun dir: {run_dir}")
    return run_dir


# ------------------------------------------------------------------------------------------ collect
def _withdrawn(rows: list[dict]) -> set[str]:
    """Certified results whose pre-registered confirmation re-run(s) all failed."""
    failed = {r["spec"].get("confirmation_of") for r in rows if r["spec"].get("confirmation_of") and r.get("fidelity") is not None
              and not r.get("certified")}
    passed = {r["spec"].get("confirmation_of") for r in rows if r["spec"].get("confirmation_of") and r.get("certified")}
    return {x for x in failed if x not in passed}


def _best_certified(run_dir: Path):
    rows = [json.loads(f.read_text()) for f in (run_dir / "experiments").glob("*.json")]
    gone = _withdrawn(rows)
    best = None
    for r in rows:
        if r.get("certified") and r["exp_id"] not in gone and (best is None or r["spec"]["n_qubits"] > best["spec"]["n_qubits"]):
            best = r
    return best


def collect_once(run_dir: Path, service=None) -> bool:
    """Collect every finished job. Returns True when nothing is left pending."""
    pend_path = run_dir / "pending.json"
    pending = json.loads(pend_path.read_text())
    log = EventLog(run_dir)
    _bj = json.loads((run_dir / "budget.json").read_text())
    budget = Budget(run_dir, max_llm_calls=int(_bj.get("max_llm_calls", 1)), max_qpu_seconds=float(_bj["max_qpu_seconds"]))
    service = service or _service()
    (run_dir / "experiments").mkdir(exist_ok=True)
    left = 0
    for entry in pending:
        if entry["status"] != "submitted":
            continue
        job = service.job(entry["job_id"])
        status = job.status()
        status = getattr(status, "name", str(status))
        entry["last_status"] = status
        if status in ("QUEUED", "RUNNING", "INITIALIZING", "VALIDATING"):
            left += 1
            continue
        s, exp_id = entry["spec"], entry["exp_id"]
        wall = time.time() - (entry.get("submitted_at") or time.time())
        result = _empty_result(exp_id, s, type("B", (), {"name": json.loads((run_dir / "calibration.json").read_text())["backend"]}), None, wall, entry.get("plan"))
        result["job_ids"] = [entry["job_id"]]
        result["predicted"] = entry.get("predicted")
        try:
            if status != "DONE":
                raise RuntimeError(f"job ended with status {status}")
            res = job.result()
            counts = counts_from_result(res, s["n_qubits"], s["readout_mitigation"])
            fit = analyze(s["n_qubits"], counts["populations"], counts["parity"], counts.get("cal0"),
                          counts.get("cal1"), mitigate=s["readout_mitigation"])
            used = _qpu_seconds(job)
            qpu_s = used if used is not None else estimate_qpu_seconds(s)
            notes = qpu_notes(s, entry.get("num_pubs", 0))
            notes["qpu_seconds_source"] = "job.usage()" if used is not None else "estimate (usage unavailable)"
            notes["execution"] = "batch submission (all jobs queued at once)"
            result.update({"circuit_depth": entry.get("circuit_depth"), "two_qubit_gate_count": entry.get("two_qubit_gate_count"),
                           "transpiled_depth": entry.get("transpiled_depth"), "plan_depth": (entry.get("plan") or {}).get("depth"),
                           "sim_method": None, **fit, "qpu_seconds": round(float(qpu_s), 3), "notes": notes})
            entry["status"] = "collected"
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"[:500]
            result["qpu_seconds"] = 0.0 if status in ("CANCELLED",) else estimate_qpu_seconds(s)
            entry["status"] = "failed"
        prev_best = _best_certified(run_dir)
        (run_dir / "experiments" / f"{exp_id}.json").write_text(json.dumps(result, indent=1, default=str))
        budget.charge_qpu(result["qpu_seconds"] or 0.0)
        log.emit("experiment_result", exp_id=exp_id, n_qubits=s["n_qubits"], mode="qpu", fidelity=result.get("fidelity"),
                 ci_low=result.get("ci_low"), ci_high=result.get("ci_high"), certified=result.get("certified"),
                 physical_qubits=result.get("physical_qubits"), tree_edges=result.get("tree_edges"),
                 root=result.get("root"), qpu_seconds=result.get("qpu_seconds"), error=result.get("error"))
        log.emit("budget", **budget.snapshot())
        if result.get("error"):
            log.emit("error", text=f"{exp_id} failed: {result['error'][:300]}")
        elif result.get("certified") and (prev_best is None or s["n_qubits"] > prev_best["spec"]["n_qubits"]):
            log.emit("milestone", text=f"New hardware record: {s['n_qubits']}-qubit GHZ entanglement certified "
                     f"(F={result['fidelity']:.3f} [{result['ci_low']:.3f},{result['ci_high']:.3f}], {exp_id})",
                     n_qubits=s["n_qubits"], fidelity=result["fidelity"])
        orig = s.get("confirmation_of")
        if orig and not result.get("error"):
            ok = bool(result.get("certified"))
            log.emit("milestone" if not ok else "thought",
                     text=(f"[confirmation] {exp_id} re-ran {orig} on the same qubits with {s.get('shots')} shots: "
                           f"F={result['fidelity']:.3f} [{result['ci_low']:.3f},{result['ci_high']:.3f}]. "
                           + (f"Confirmed: {orig} stands." if ok else
                              f"Not certified, so I withdraw {orig}'s N={s['n_qubits']} result as a likely upward fluctuation.")),
                     n_qubits=s["n_qubits"], fidelity=result["fidelity"])
        f = result.get("fidelity")
        pred = (entry.get("predicted") or {}).get("fidelity")
        if f is not None and pred is not None:
            log.emit("thought", text=(f"[prediction check] {exp_id} (N={s['n_qubits']}, {s['layout_strategy']}): my digital twin "
                                      f"predicted F={pred:.3f} before submission; hardware measured F={f:.3f} "
                                      f"({f - pred:+.3f}). " + ("Certified." if result.get("certified") else "Not certified.")))
        print(f"  {exp_id} N={s['n_qubits']} {s['layout_strategy']}: "
              + (f"F={f:.3f} [{result['ci_low']:.3f},{result['ci_high']:.3f}] certified={result['certified']}" if f is not None
                 else f"ERROR {result['error']}"))
    pend_path.write_text(json.dumps(pending, indent=1, default=str))
    evs = read_events(run_dir)
    last_fin = max((e["seq"] for e in evs if e["type"] == "run_finished"), default=-1)
    last_start = max((e["seq"] for e in evs if e["type"] == "experiment_started"), default=-1)
    if left == 0 and last_fin < last_start:
        best = _best_certified(run_dir)
        log.emit("run_finished", best_certified_n=best["spec"]["n_qubits"] if best else None,
                 best_exp_id=best["exp_id"] if best else None,
                 summary=(f"Largest certified GHZ on hardware: N={best['spec']['n_qubits']} (F={best['fidelity']:.3f})"
                          if best else "No GHZ size certified on hardware in this batch."))
    return left == 0


def _round2_plan(run_dir: Path, budget_left: float) -> tuple[list[dict], str, str] | None:
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        return None
    model = (os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash").split("#")[0].strip() or "gemini-2.5-flash"
    pend = json.loads((run_dir / "pending.json").read_text())
    rows = []
    for e in pend:
        f = run_dir / "experiments" / f"{e['exp_id']}.json"
        r = json.loads(f.read_text()) if f.exists() else {}
        rows.append({"exp_id": e["exp_id"], "N": e["spec"]["n_qubits"], "layout": e["spec"]["layout_strategy"],
                     "shots": e["spec"]["shots"], "twin_predicted_F": round((e.get("predicted") or {}).get("fidelity") or 0, 3),
                     "hardware_F": None if r.get("fidelity") is None else round(r["fidelity"], 3),
                     "hardware_ci": None if r.get("ci_low") is None else [round(r["ci_low"], 3), round(r["ci_high"], 3)],
                     "certified": r.get("certified"), "error": r.get("error"), "qpu_seconds_billed": r.get("qpu_seconds")})
    costs = {f"N={n},shots={sh}": estimate_qpu_seconds({"n_qubits": n, "shots": sh, "mode": "qpu"})
             for n in (16, 24, 28, 32, 36, 40) for sh in (1000, 2000)}
    prompt = f"""You are Q-Scientist, an autonomous experimental physicist. Your first hardware batch on real IBM
hardware has finished. Results (your digital-twin prediction was frozen BEFORE each job ran):
{json.dumps(rows, indent=1)}

You have {budget_left:.0f} QPU seconds left. Estimated cost table (conservative; actual billing has been about half):
{json.dumps(costs)}

Design ROUND 2 (at most 4 experiments). Hard requirements (a validator enforces them):
- a CONFIRMATION re-run of your largest certified hardware result: identical N and layout, 2000 shots. A record chosen
  as the best of several noisy runs is biased upward; an independent repeat is the honest test.
- use what you learned: e.g. probe sizes between your largest certified and smallest uncertified N, and/or a second
  tree-vs-chain matched pair if the first comparison was inconclusive.
layout_strategy: weighted_tree | bfs_tree | best_chain. shots: 1000 or 2000.

Return JSON: {{"analysis": "3-5 sentences: what the hardware results and the twin's errors taught you",
"rationale": "2-3 sentences on why this round-2 design", "experiments": [{{"n_qubits": int, "layout_strategy": str,
"shots": int, "hypothesis": "one sentence"}}]}}"""
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                  "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4,
                                       "maxOutputTokens": 16384, "thinkingConfig": {"thinkingBudget": 4096}}},
            timeout=180,
        )
        r.raise_for_status()
        data = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        text = (str(data.get("analysis", "")) + " " + str(data.get("rationale", ""))).strip()
        return list(data["experiments"]), text, f"gemini:{model}"
    except Exception as e:
        print(f"Gemini round-2 planning failed ({type(e).__name__}: {str(e)[:200]})")
        return None


def cmd_extend(args) -> None:
    """Round 2: the agent reads the hardware results and designs a follow-up inside the same run."""
    _load_env()
    run_dir = Path(args.run_dir)
    cal = json.loads((run_dir / "calibration.json").read_text())
    backend = get_backend("qpu", cal["backend"])
    bj = json.loads((run_dir / "budget.json").read_text())
    budget_left = float(bj["max_qpu_seconds"]) - float(bj["qpu_seconds"])
    pend = json.loads((run_dir / "pending.json").read_text())
    if any(e["status"] == "submitted" for e in pend):
        print("round 1 still running; extend refused")
        return
    best = _best_certified(run_dir)
    planned = _round2_plan(run_dir, budget_left)
    exps, text, planner = planned if planned else ([], "Round 2 (no language model): confirmation of the record.", "fixed-plan")
    # validator: confirmation of the record is mandatory
    exps = [e for e in exps if isinstance(e, dict)]
    if best:
        bs = best["spec"]
        if not any(int(e.get("n_qubits", 0)) == bs["n_qubits"] and e.get("layout_strategy") == bs["layout_strategy"] for e in exps):
            exps.insert(0, {"n_qubits": bs["n_qubits"], "layout_strategy": bs["layout_strategy"], "shots": 2000,
                            "hypothesis": "added by validator: confirmation re-run of the record"})
        for e in exps:  # the confirmation runs at 2000 shots
            if int(e.get("n_qubits", 0)) == bs["n_qubits"] and e.get("layout_strategy") == bs["layout_strategy"]:
                e["shots"] = 2000
                e["confirmation_of"] = best["exp_id"]
    clean = []
    for e in exps[:4]:
        try:
            n, strat, shots = int(e["n_qubits"]), str(e["layout_strategy"]), int(e.get("shots", 1000))
        except Exception:
            continue
        if strat in STRATEGIES and 4 <= n <= 100:
            clean.append({**e, "n_qubits": n, "layout_strategy": strat, "shots": 2000 if shots >= 1500 else 1000})
    specs = []
    for e in clean:
        sp = _full_spec(e)
        sp["shots"] = e["shots"]
        if e.get("confirmation_of"):
            sp["confirmation_of"] = e["confirmation_of"]
        specs.append(sp)
    while specs and sum(estimate_qpu_seconds(x) for x in specs) > budget_left:
        drop = max((x for x in specs if not x.get("confirmation_of")), key=lambda x: x["n_qubits"], default=None)
        if drop is None:
            specs[0]["shots"] = 1000
            if sum(estimate_qpu_seconds(x) for x in specs) > budget_left:
                specs = []
            break
        specs.remove(drop)
    print(f"planner={planner}\n{text}\n")
    for sp in specs:
        print(f"  N={sp['n_qubits']} {sp['layout_strategy']} shots={sp['shots']} est {estimate_qpu_seconds(sp):.0f}s | {sp['hypothesis']}")
    print(f"budget left {budget_left:.0f}s")
    if args.dry_run or not specs:
        return
    log = EventLog(run_dir)
    log.emit("thought", text=("[round 2] " + text)[:1500])
    k = len(pend)
    for sp in specs:
        k += 1
        exp_id = f"e{k:03d}"
        entry = {"exp_id": exp_id, "spec": sp, "job_id": None, "status": "new", "submitted_at": None, "round": 2}
        try:
            prep = prepare(sp, backend, cal)
            entry.update({"plan": prep["plan"], "circuit_depth": prep["circuit_depth"],
                          "two_qubit_gate_count": prep["two_qubit_gate_count"], "transpiled_depth": prep["transpiled_depth"],
                          "num_pubs": len(build_qpu_pubs(prep)[0])})
            try:
                counts, _ = run_sim(prep, backend, cal)
                fit = analyze(sp["n_qubits"], counts["populations"], counts["parity"], counts.get("cal0"), counts.get("cal1"), mitigate=True)
                entry["predicted"] = {kk: fit[kk] for kk in ("fidelity", "ci_low", "ci_high", "certified")}
            except Exception as ex:
                entry["predicted"] = {"error": str(ex)[:200]}
            job = submit_qpu(prep, backend)
            entry.update({"job_id": job.job_id(), "status": "submitted", "submitted_at": time.time()})
            log.emit("experiment_started", exp_id=exp_id, spec=sp)
            print(f"  submitted {exp_id} job={entry['job_id']}")
        except Exception as ex:
            entry.update({"status": "failed_submit", "error": f"{type(ex).__name__}: {ex}"[:500]})
            log.emit("error", text=f"{exp_id} submit failed: {entry['error']}")
        pend.append(entry)
        (run_dir / "pending.json").write_text(json.dumps(pend, indent=1, default=str))
    if planner.startswith("gemini"):
        bj2 = json.loads((run_dir / "budget.json").read_text())
        bj2["llm_calls"] = int(bj2.get("llm_calls", 0)) + 1
        bj2["max_llm_calls"] = int(bj2.get("max_llm_calls", 1)) + 1
        (run_dir / "budget.json").write_text(json.dumps(bj2))
        log.emit("budget", **bj2)


def cmd_collect(args) -> None:
    _load_env()
    run_dir = Path(args.run_dir)
    service = _service()
    deadline = time.time() + args.max_hours * 3600
    while True:
        done = collect_once(run_dir, service)
        pending = json.loads((run_dir / "pending.json").read_text())
        states = {}
        for e in pending:
            st = e.get("last_status") or e["status"]
            states[st] = states.get(st, 0) + 1
        print(time.strftime("%H:%M:%S"), "status:", states, flush=True)
        if done or not args.watch or time.time() > deadline:
            break
        time.sleep(args.interval)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("submit")
    a.add_argument("--backend", default=None)
    a.add_argument("--no-llm", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--max-qpu-seconds", type=float, default=float((os.environ.get("MAX_QPU_SECONDS") or "480").split("#")[0]))
    a.add_argument("--runs-root", default="runs")
    c = sub.add_parser("collect")
    c.add_argument("run_dir")
    c.add_argument("--watch", action="store_true")
    c.add_argument("--interval", type=int, default=120)
    c.add_argument("--max-hours", type=float, default=9.0)
    x = sub.add_parser("extend")
    x.add_argument("run_dir")
    x.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return {"submit": cmd_submit, "collect": cmd_collect, "extend": cmd_extend}[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])

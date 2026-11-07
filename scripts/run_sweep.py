"""NO-AI fallback: a fixed sweep over GHZ sizes, same run-directory/event format as the agent.

    uv run python -m scripts.run_sweep --mode sim --ns 8,16,24,32
    uv run python -m scripts.run_sweep --mode qpu --ns 8,16,24 [--backend ibm_fez] [--shots 2000]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from qsci.budget import Budget
from qsci.core import estimate_qpu_seconds, get_backend, get_calibration, run_experiment
from qsci.events import EventLog, new_run_dir


def _summary(res: dict) -> dict:
    keys = ["exp_id", "fidelity", "ci_low", "ci_high", "certified", "physical_qubits", "tree_edges", "root", "qpu_seconds", "error"]
    out = {k: res.get(k) for k in keys}
    out["n_qubits"] = res["spec"]["n_qubits"] if isinstance(res.get("spec"), dict) else None
    out["mode"] = res["spec"].get("mode") if isinstance(res.get("spec"), dict) else None
    return out


def main(argv=None) -> Path:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["sim", "qpu"], default="sim")
    ap.add_argument("--ns", default="8,16,24,32")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--strategy", default="weighted_tree", choices=["weighted_tree", "bfs_tree", "best_chain"])
    ap.add_argument("--variant", default="tree", choices=["tree", "linear"])
    ap.add_argument("--no-mitigation", action="store_true")
    ap.add_argument("--max-qpu-seconds", type=float, default=float(os.environ.get("MAX_QPU_SECONDS") or 480))
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args(argv)
    ns = [int(x) for x in args.ns.split(",") if x.strip()]

    run_dir = new_run_dir(args.mode, Path(args.runs_root))
    log = EventLog(run_dir)
    budget = Budget(run_dir, max_llm_calls=0, max_qpu_seconds=args.max_qpu_seconds)
    try:
        backend = get_backend(args.mode, args.backend)
        cal = get_calibration(backend)
    except Exception as e:
        log.emit("error", text=f"backend setup failed: {type(e).__name__}: {e}")
        log.emit("run_finished", best_certified_n=None, best_exp_id=None, summary="aborted: no backend")
        print(f"backend setup failed: {e}")
        return run_dir
    (run_dir / "calibration.json").write_text(json.dumps(cal, indent=1))
    log.emit(
        "run_started",
        run_id=run_dir.name,
        mode=args.mode,
        backend=cal["backend"],
        goal=f"NO-AI sweep over N={ns}: certify GHZ fidelity > 0.5",
        max_llm_calls=0,
        max_qpu_seconds=args.max_qpu_seconds,
    )
    log.emit("budget", **budget.snapshot())

    best_n, best_id, lines = None, None, []
    for i, n in enumerate(ns, start=1):
        exp_id = f"e{i:03d}"
        spec = {
            "n_qubits": n,
            "layout_strategy": args.strategy,
            "root": None,
            "exclude_qubits": [],
            "circuit_variant": args.variant,
            "dynamical_decoupling": True,
            "measure_twirling": True,
            "readout_mitigation": not args.no_mitigation,
            "shots": args.shots,
            "mode": args.mode,
            "hypothesis": f"sweep point N={n}",
        }
        if args.mode == "qpu":
            est = estimate_qpu_seconds(spec)
            if not budget.can_spend_qpu(est):
                log.emit("error", text=f"{exp_id} N={n} skipped: estimated {est:.0f}s QPU exceeds remaining budget")
                print(f"N={n}: skipped (budget)")
                continue
        log.emit("experiment_started", exp_id=exp_id, spec=spec)
        res = run_experiment(spec, backend, cal, exp_id)
        if args.mode == "qpu":
            budget.charge_qpu(res.get("qpu_seconds") or 0.0)
        (run_dir / "experiments" / f"{exp_id}.json").write_text(json.dumps(res, indent=1, default=str))
        log.emit("experiment_result", **_summary(res))
        log.emit("budget", **budget.snapshot())
        if res.get("error"):
            log.emit("error", text=f"{exp_id} N={n}: {res['error']}")
            lines.append(f"N={n}: error")
            print(f"N={n}: ERROR {res['error']}")
            continue
        line = f"N={n}: F={res['fidelity']:.3f} [{res['ci_low']:.3f}, {res['ci_high']:.3f}] {'CERTIFIED' if res['certified'] else 'not certified'} ({res['wall_seconds']}s)"
        print(line)
        lines.append(line)
        if res["certified"] and (best_n is None or n > best_n):
            best_n, best_id = n, exp_id
            log.emit("milestone", text=f"New record: {n}-qubit GHZ certified (F={res['fidelity']:.3f})", n_qubits=n, fidelity=res["fidelity"])
    summary = (f"Largest certified GHZ: N={best_n} ({best_id}). " if best_n else "No certified GHZ state. ") + "; ".join(lines)
    log.emit("run_finished", best_certified_n=best_n, best_exp_id=best_id, summary=summary)
    print(f"run dir: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()

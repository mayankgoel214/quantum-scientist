"""Run the Q-Scientist agent.

    uv run python -m scripts.run_agent --mode sim --llm fake
    uv run python -m scripts.run_agent --mode qpu --llm gemini --max-qpu-seconds 480
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Autonomous GHZ-entanglement scientist")
    p.add_argument("--mode", choices=["sim", "qpu"], required=True)
    p.add_argument("--llm", choices=["gemini", "fake"], default="gemini")
    p.add_argument("--max-llm-calls", type=int, default=int(os.environ.get("MAX_LLM_CALLS") or 300))
    p.add_argument("--max-qpu-seconds", type=float, default=float(os.environ.get("MAX_QPU_SECONDS") or 480))
    p.add_argument("--backend", default=os.environ.get("IBM_BACKEND") or None,
                   help="IBM backend name for --mode qpu (default: IBM_BACKEND env / core default)")
    p.add_argument("--max-hours", type=float, default=7.0)
    p.add_argument("--max-experiments", type=int, default=80)
    p.add_argument("--runs-root", default="runs")
    a = p.parse_args(argv)

    if a.llm == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set (add it to .env) -- or run with --llm fake.", file=sys.stderr)
        return 2
    if a.mode == "qpu" and a.max_qpu_seconds > 600:
        print("Refusing --max-qpu-seconds > 600 (the account only has 10 hardware minutes).", file=sys.stderr)
        return 2

    from qsci.agent.loop import run_agent

    run_dir = run_agent(mode=a.mode, llm=a.llm, max_llm_calls=a.max_llm_calls,
                        max_qpu_seconds=a.max_qpu_seconds,
                        backend_name=a.backend if a.mode == "qpu" else None,
                        max_hours=a.max_hours, max_experiments=a.max_experiments,
                        runs_root=Path(a.runs_root),
                        log=lambda msg: print(msg, flush=True))
    print(f"Run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

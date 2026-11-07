"""Write the Q-Scientist paper for a finished run.

    uv run python -m scripts.write_paper runs/<run_id> [--no-llm] [--no-pdf] [--operator "Name"]

Outputs runs/<run_id>/paper/{paper.html, paper.pdf, fig_*.png}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from qsci.paper import write_paper


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="runs/<run_id>")
    ap.add_argument("--operator", default="Mayank Goel", help="human co-author name")
    ap.add_argument("--no-llm", action="store_true", help="skip the optional Gemini polish")
    ap.add_argument("--no-pdf", action="store_true", help="write HTML + figures only")
    args = ap.parse_args(argv)
    if not args.run_dir.is_dir():
        print(f"error: {args.run_dir} is not a directory", file=sys.stderr)
        return 2
    load_dotenv()
    res = write_paper(args.run_dir, operator=args.operator, use_llm=not args.no_llm, pdf=not args.no_pdf)
    print(f"HTML: {res['html']}")
    print(f"PDF:  {res['pdf'] or '(not generated)'}")
    print(f"Figures: {', '.join(p.name for p in res['figures'])}")
    print(f"LLM polish: {'yes' if res['llm_used'] else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

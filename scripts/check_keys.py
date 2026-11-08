"""Report which required keys are set in .env (never prints values).

Exit 0 when Gemini + IBM token + IBM instance are all set, else 1.
Usage: uv run python -m scripts.check_keys
"""

import sys
from pathlib import Path

from dotenv import dotenv_values

REQUIRED = ["GEMINI_API_KEY", "IBM_QUANTUM_TOKEN", "IBM_QUANTUM_INSTANCE"]


def main() -> int:
    values = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
    ok = True
    for k in REQUIRED:
        v = (values.get(k) or "").strip()
        is_set = len(v) >= 10
        ok &= is_set
        print(f"{k}: {'set' if is_set else 'MISSING'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

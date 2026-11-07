"""Hard spending caps. Charged BEFORE spending so a cap can never be overrun."""
from __future__ import annotations

import json
import threading
from pathlib import Path


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, run_dir: Path, max_llm_calls: int, max_qpu_seconds: float):
        self.path = Path(run_dir) / "budget.json"
        self._lock = threading.Lock()
        state = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.llm_calls = int(state.get("llm_calls", 0))
        self.qpu_seconds = float(state.get("qpu_seconds", 0.0))
        self.max_llm_calls = int(max_llm_calls)
        self.max_qpu_seconds = float(max_qpu_seconds)
        self._save()

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.snapshot(), indent=2))

    def charge_llm(self) -> None:
        with self._lock:
            if self.llm_calls >= self.max_llm_calls:
                raise BudgetExceeded(f"LLM call cap reached ({self.max_llm_calls})")
            self.llm_calls += 1
            self._save()

    def can_spend_qpu(self, est_seconds: float) -> bool:
        return self.qpu_seconds + est_seconds <= self.max_qpu_seconds

    def charge_qpu(self, seconds: float) -> None:
        with self._lock:
            self.qpu_seconds += float(seconds)
            self._save()

    def snapshot(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "max_llm_calls": self.max_llm_calls,
            "qpu_seconds": round(self.qpu_seconds, 3),
            "max_qpu_seconds": self.max_qpu_seconds,
        }

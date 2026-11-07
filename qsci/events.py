"""Append-only event log shared by the agent, the sweep script, the UI and the paper writer."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_dir(mode: str, root: Path = Path("runs")) -> Path:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{mode}"
    run_dir = Path(root) / run_id
    (run_dir / "experiments").mkdir(parents=True, exist_ok=False)
    return run_dir


def read_events(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "events.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # tolerate a partially written last line during live tailing
    return out


class EventLog:
    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = len(read_events(run_dir))

    def emit(self, type: str, **data) -> dict:
        with self._lock:
            event = {"ts": _now(), "seq": self._seq, "type": type, "data": data}
            self._seq += 1
            with self.path.open("a") as f:
                f.write(json.dumps(event, default=str) + "\n")
                f.flush()
        return event

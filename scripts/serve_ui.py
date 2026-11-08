"""Mission-control web UI server (stdlib only).

    uv run python -m scripts.serve_ui [--port 8765] [--runs-dir runs] [--host 127.0.0.1]

Serves ui/ as static files plus a small read-only JSON API over run directories:

    GET /api/runs                                   newest first, with a summary per run
    GET /api/runs/<id>/events?since=<seq>           events with seq > since (default: all)
    GET /api/runs/<id>/calibration
    GET /api/runs/<id>/budget
    GET /api/runs/<id>/experiments/<exp_id>
    GET /api/runs/<id>/paper/<file>                 paper.html, paper.pdf, fig_*.png, ...
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")  # no slashes, no leading dot

RUNS_DIR = ROOT / "runs"


def _read_events(run_dir: Path) -> list[dict]:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # partially written last line while the agent is still writing
    return out


def _run_dir(run_id: str) -> Path | None:
    if not SAFE.match(run_id):
        return None
    p = (RUNS_DIR / run_id).resolve()
    if p.parent != RUNS_DIR.resolve() or not p.is_dir():
        return None
    return p


def _child(base: Path, name: str) -> Path | None:
    if not SAFE.match(name):
        return None
    p = (base / name).resolve()
    if p.parent != base.resolve() or not p.is_file():
        return None
    return p


def _summary(run_dir: Path) -> dict:
    events = _read_events(run_dir)
    started = next((e for e in events if e.get("type") == "run_started"), None)
    finished = next((e for e in events if e.get("type") == "run_finished"), None)
    best = {"sim": 0, "qpu": 0}
    n_exp = 0
    conf_of = {(e.get("data") or {}).get("exp_id"): ((e.get("data") or {}).get("spec") or {}).get("confirmation_of")
               for e in events if e.get("type") == "experiment_started"}
    results = []
    for e in events:
        if e.get("type") != "experiment_result":
            continue
        n_exp += 1
        d = e.get("data") or {}
        lo = d.get("ci_low")
        results.append((d, d.get("error") is None and isinstance(lo, (int, float)) and lo > 0.5))
    # a result whose pre-registered confirmation re-run failed is withdrawn (same rule as the paper and the UI)
    failed = {conf_of.get(d.get("exp_id")) for d, ok in results if conf_of.get(d.get("exp_id")) and d.get("error") is None and not ok}
    passed = {conf_of.get(d.get("exp_id")) for d, ok in results if conf_of.get(d.get("exp_id")) and ok}
    for d, ok in results:
        if ok and not (d.get("exp_id") in failed and d.get("exp_id") not in passed):
            m = d.get("mode", "sim")
            best[m] = max(best.get(m, 0), int(d.get("n_qubits") or 0))
    sd = (started or {}).get("data", {})
    mode = sd.get("mode") or run_dir.name.rsplit("-", 1)[-1]
    try:
        mtime = (run_dir / "events.jsonl").stat().st_mtime
    except OSError:
        mtime = run_dir.stat().st_mtime
    return {
        "run_id": run_dir.name,
        "mode": mode,
        "backend": sd.get("backend"),
        "best_certified_n": max(best.values()) if best else 0,
        "best_certified_sim": best.get("sim", 0),
        "best_certified_qpu": best.get("qpu", 0),
        "n_experiments": n_exp,
        "n_events": len(events),
        "finished": finished is not None,
        "started_at": (started or {}).get("ts"),
        "updated_at": mtime,
        "has_paper": (run_dir / "paper" / "paper.html").exists(),
    }


def _sort_key(run_dir: Path):
    # run ids embed YYYYMMDD-HHMMSS (optionally after a "demo-" prefix); fall back to mtime
    # real runs sort ahead of demo-* fixtures so the default pick is tonight's real run
    m = re.search(r"(\d{8}-\d{6})", run_dir.name)
    return (not run_dir.name.startswith("demo-"), m.group(1) if m else "", run_dir.stat().st_mtime)


class Handler(BaseHTTPRequestHandler):
    server_version = "QScientist/1.0"

    def log_message(self, fmt, *args):  # quieter: skip polling noise
        if "/events" in (args[0] if args else ""):
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ------------------------------------------------------------ helpers
    def _send(self, status: int, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=60" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def _err(self, status: int, msg: str) -> None:
        self._json({"error": msg}, status)

    def _file(self, path: Path) -> None:
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        self._send(200, path.read_bytes(), ctype)

    # ------------------------------------------------------------ routing
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.split("/") if p]
        try:
            if parts[:1] == ["api"]:
                return self._api(parts[1:], parse_qs(url.query))
            return self._static(parts)
        except BrokenPipeError:
            pass
        except Exception as exc:  # never crash the server mid-demo
            self._err(500, f"{type(exc).__name__}: {exc}")

    def _static(self, parts: list[str]) -> None:
        if not parts:
            parts = ["index.html"]
        if any(not SAFE.match(p) for p in parts):
            return self._err(404, "not found")
        p = (UI_DIR.joinpath(*parts)).resolve()
        if UI_DIR.resolve() not in p.parents or not p.is_file():
            return self._err(404, "not found")
        self._file(p)

    def _api(self, parts: list[str], q: dict) -> None:
        if parts == ["runs"]:
            if not RUNS_DIR.is_dir():
                return self._json({"runs": []})
            dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and SAFE.match(d.name)
                    and (d / "events.jsonl").exists()]
            dirs.sort(key=_sort_key, reverse=True)
            return self._json({"runs": [_summary(d) for d in dirs]})

        if len(parts) < 3 or parts[0] != "runs":
            return self._err(404, "unknown endpoint")
        run_dir = _run_dir(parts[1])
        if run_dir is None:
            return self._err(404, "run not found")
        what, rest = parts[2], parts[3:]

        if what == "events" and not rest:
            try:
                since = int(q.get("since", ["-1"])[0])
            except ValueError:
                return self._err(400, "since must be an integer")
            events = _read_events(run_dir)
            new = [e for e in events if isinstance(e.get("seq"), int) and e["seq"] > since]
            finished = any(e.get("type") == "run_finished" for e in events)
            return self._json({"run_id": run_dir.name, "events": new, "finished": finished,
                               "last_seq": events[-1]["seq"] if events else -1})
        if what in ("calibration", "budget") and not rest:
            p = run_dir / f"{what}.json"
            if not p.is_file():
                return self._err(404, f"{what}.json not found")
            return self._file(p)
        if what == "experiments" and len(rest) == 1:
            p = _child(run_dir / "experiments", rest[0] + ".json") if not rest[0].endswith(".json") \
                else _child(run_dir / "experiments", rest[0])
            if p is None:
                return self._err(404, "experiment not found")
            return self._file(p)
        if what == "paper" and len(rest) <= 1:
            p = _child(run_dir / "paper", rest[0] if rest else "paper.html")
            if p is None:
                return self._err(404, "paper file not found")
            return self._file(p)
        return self._err(404, "unknown endpoint")


def main() -> None:
    global RUNS_DIR
    ap = argparse.ArgumentParser(description="Q-Scientist mission control")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    args = ap.parse_args()
    RUNS_DIR = Path(args.runs_dir).resolve()
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"Q-Scientist mission control → http://localhost:{args.port}  (runs: {RUNS_DIR})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

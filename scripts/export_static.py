"""Export mission control + finished runs as a static site (for Vercel or any static host).

    uv run python -m scripts.export_static [--out site] [--runs RUN_ID ...]

Writes the same JSON the local server's /api returns, as files under <out>/data/,
and flags the page with window.QS_STATIC so ui/app.js reads those files instead.
Only run data is exported: no .env, no keys, no source.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scripts.serve_ui import ROOT, RUNS_DIR, UI_DIR, _read_events, _sort_key, _summary


def export(out: Path, run_ids: list[str] | None) -> list[str]:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(UI_DIR, out)
    idx = out / "index.html"
    html = idx.read_text()
    html = html.replace('<script src="app.js"></script>',
                        '<script>window.QS_STATIC = true;</script>\n  <script src="app.js"></script>')
    idx.write_text(html)

    dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "events.jsonl").exists()
            and not d.name.startswith("demo-")]
    if run_ids:
        dirs = [d for d in dirs if d.name in run_ids]
    dirs.sort(key=_sort_key, reverse=True)
    summaries = []
    for d in dirs:
        dst = out / "data" / d.name
        (dst / "experiments").mkdir(parents=True)
        events = _read_events(d)
        (dst / "events.json").write_text(json.dumps({
            "run_id": d.name, "events": events,
            "finished": any(e.get("type") == "run_finished" for e in events),
            "last_seq": events[-1]["seq"] if events else -1}))
        for name in ("calibration.json", "budget.json"):
            if (d / name).is_file():
                shutil.copy(d / name, dst / name)
        for f in (d / "experiments").glob("*.json"):
            shutil.copy(f, dst / "experiments" / f.name)
        if (d / "paper").is_dir():
            shutil.copytree(d / "paper", dst / "paper")
        summaries.append(_summary(d))
    (out / "data" / "runs.json").write_text(json.dumps({"runs": summaries}))
    (out / "vercel.json").write_text(json.dumps({"cleanUrls": False, "headers": [
        {"source": "/data/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=300"}]}]}, indent=2))
    return [d.name for d in dirs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "site"))
    ap.add_argument("--runs", nargs="*")
    a = ap.parse_args()
    print("exported:", export(Path(a.out), a.runs), "->", a.out)


if __name__ == "__main__":
    main()

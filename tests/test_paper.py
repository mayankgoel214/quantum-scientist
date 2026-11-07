"""Paper generation from the hand-built fixture run (no network, no LLM)."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from qsci.paper.data import load_run
from qsci.paper.llm import numbers_preserved
from scripts.write_paper import main

FIXTURE = Path(__file__).parent / "fixtures" / "sample_run"


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> Path:
    run = tmp_path_factory.mktemp("runs") / FIXTURE.name
    shutil.copytree(FIXTURE, run, ignore=shutil.ignore_patterns("paper"))
    assert main([str(run), "--no-llm", "--operator", "Ada Lovelace"]) == 0
    return run


def _expected(run: Path) -> dict:
    exps = [json.loads(p.read_text()) for p in sorted((run / "experiments").glob("*.json"))]
    ok = [e for e in exps if e.get("error") is None and e.get("ci_low") is not None]
    hw = max((e for e in ok if e["spec"]["mode"] == "qpu" and e["ci_low"] > 0.5), key=lambda e: e["spec"]["n_qubits"])
    sim = max((e for e in ok if e["spec"]["mode"] == "sim" and e["ci_low"] > 0.5), key=lambda e: e["spec"]["n_qubits"])
    return {"exps": exps, "hw": hw, "sim": sim}


def test_files_exist(generated: Path):
    paper = generated / "paper"
    assert (paper / "paper.html").stat().st_size > 5000
    for name in ("fig_fidelity_vs_n.png", "fig_parity.png", "fig_chip_map.png", "fig_timeline.png"):
        assert (paper / name).stat().st_size > 10000, name
    if shutil.which("google-chrome") or Path("/Applications/Google Chrome.app").exists():
        assert (paper / "paper.pdf").read_bytes()[:5] == b"%PDF-"


def test_key_numbers_in_html(generated: Path):
    html = (generated / "paper" / "paper.html").read_text()
    exp = _expected(generated)
    hw, sim = exp["hw"], exp["sim"]
    assert f"N = {hw['spec']['n_qubits']}" in html
    assert f"{hw['fidelity']:.3f}" in html
    assert f"[{hw['ci_low']:.3f}, {hw['ci_high']:.3f}]" in html
    assert f"N = {sim['spec']['n_qubits']}" in html
    assert f"{len(exp['exps'])} experiments" in html
    assert "Q-Scientist (autonomous agent, Gemini) and Ada Lovelace" in html
    run_id = json.loads((generated / "events.jsonl").read_text().splitlines()[0])["data"]["run_id"]
    assert run_id in html
    for e in exp["exps"]:  # every experiment in the table, every hardware job id listed
        assert f"<td>{e['exp_id']}</td>" in html
        for j in e.get("job_ids") or []:
            assert j in html


def test_sections_and_thoughts(generated: Path):
    html = (generated / "paper" / "paper.html").read_text()
    for h in ("Introduction", "Methods", "The autonomous agent", "Results", "Discussion and limitations", "Reproducibility", "References"):
        assert h in html
    quotes = re.findall(r"<blockquote>", html)
    assert 3 <= len(quotes) <= 5
    thoughts = [json.loads(l)["data"]["text"] for l in (generated / "events.jsonl").read_text().splitlines() if '"thought"' in l]
    assert sum(t[:60] in html for t in thoughts) >= 3
    assert "Gemini under a numbers-must-not-change check" not in html  # --no-llm


def test_budget_not_charged_without_llm(generated: Path):
    before = json.loads((FIXTURE / "budget.json").read_text())
    after = json.loads((generated / "budget.json").read_text())
    assert before["llm_calls"] == after["llm_calls"]


def test_record_selection():
    facts = load_run(FIXTURE)
    exp = _expected(FIXTURE)
    assert facts.record("qpu").exp_id == exp["hw"]["exp_id"]
    assert facts.record("sim").exp_id == exp["sim"]["exp_id"]
    assert all(e.n for e in facts.experiments)


def test_llm_number_guard():
    draft = "We certified N = 36 with F = 0.541 (CI 0.511–0.571)."
    assert numbers_preserved(draft, "A 36-qubit state was certified: F = 0.541, CI 0.511–0.571.")
    assert not numbers_preserved(draft, "We certified N = 38 with F = 0.541.")


def test_llm_polish_mocked(tmp_path, monkeypatch):
    """Polish path without network: header-only key, <=2 calls, budget charged, bad rewrites rejected."""
    import qsci.paper.llm as llm

    run = tmp_path / FIXTURE.name
    shutil.copytree(FIXTURE, run, ignore=shutil.ignore_patterns("paper"))
    calls = []

    class Resp:
        status_code = 200

        def __init__(self, text):
            self._text = text

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": self._text}]}}]}

    def fake_post(url, headers, json, timeout):
        calls.append((url, headers))
        assert "key=" not in url and headers["x-goog-api-key"] == "test-key"
        draft = json["contents"][0]["parts"][0]["text"].split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0]
        # first call: faithful rewrite; second call: tries to inflate a number -> must be rejected
        return Resp(draft.replace("We report", "We present") if len(calls) == 1 else draft + " We also reached N = 99.")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm.requests, "post", fake_post)
    res = __import__("qsci.paper", fromlist=["write_paper"]).write_paper(run, use_llm=True, pdf=False, log=lambda m: None)
    html = (run / "paper" / "paper.html").read_text()
    assert len(calls) == 2
    assert "We present an autonomous search" in html
    assert "N = 99" not in html
    assert res["llm_used"] is True
    before = json.loads((FIXTURE / "budget.json").read_text())["llm_calls"]
    assert json.loads((run / "budget.json").read_text())["llm_calls"] == before + 2

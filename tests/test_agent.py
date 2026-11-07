"""Agent-loop tests. No network, no qiskit: uses a stand-in fake core and FakeLLM / fake HTTP."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from qsci.agent.llm import (GeminiLLM, LLMError, LLMRequest, build_gemini_body,
                            parse_gemini_response)
from qsci.agent.loop import run_agent
from qsci.agent.policy import config_key
from qsci.agent.tools import ToolBox, normalize_spec, summarize_calibration, tool_declarations
from qsci.budget import Budget, BudgetExceeded
from qsci.events import EventLog, read_events

EVENT_KEYS = {
    "run_started": {"run_id", "mode", "backend", "goal", "planner", "max_llm_calls", "max_qpu_seconds"},
    "thought": {"text"},
    "experiment_started": {"exp_id", "spec"},
    "experiment_result": {"exp_id", "n_qubits", "mode", "fidelity", "ci_low", "ci_high", "certified",
                          "physical_qubits", "tree_edges", "root", "qpu_seconds", "error"},
    "budget": {"llm_calls", "max_llm_calls", "qpu_seconds", "max_qpu_seconds"},
    "milestone": {"text", "n_qubits", "fidelity"},
    "error": {"text"},
    "run_finished": {"best_certified_n", "best_exp_id", "summary"},
}
SPEC_KEYS = {"n_qubits", "layout_strategy", "root", "exclude_qubits", "circuit_variant",
             "dynamical_decoupling", "measure_twirling", "readout_mitigation", "shots", "mode",
             "hypothesis"}


# ----------------------------------------------------------------------------- fake core

class FakeCore:
    """Stand-in for qsci.core following CONTRACTS.md. F decays ~exp(-N/45) (frontier ~N=28)."""

    def __init__(self, n_qubits: int = 156, fail_exp: bool = False, interrupt_at: int | None = None):
        self.n = n_qubits
        self.fail_exp = fail_exp
        self.interrupt_at = interrupt_at
        self.calls: list[dict] = []

    def get_backend(self, mode, name=None):
        return SimpleNamespace(name="fake_fez" if mode == "sim" else (name or "ibm_fez"))

    def get_calibration(self, backend):
        rng = random.Random(1)
        qubits = [{"id": i, "t1_us": rng.uniform(80, 250), "t2_us": rng.uniform(40, 200),
                   "readout_error": 0.2 if i in (5, 17) else rng.uniform(0.005, 0.03),
                   "sx_error": 3e-4, "x": float(i % 16), "y": float(i // 16)} for i in range(self.n)]
        edges = [{"a": i, "b": i + 1, "error": rng.uniform(0.002, 0.01), "gate": "cz"}
                 for i in range(self.n - 1)]
        return {"backend": backend.name, "num_qubits": self.n, "source": "fake",
                "timestamp": "2026-11-07T00:00:00+00:00", "qubits": qubits, "edges": edges}

    def plan_layout(self, calibration, n, strategy, root=None, exclude=()):
        qs = [q for q in range(self.n) if q not in set(exclude)][:n]
        return {"root": qs[0], "qubits": qs, "tree_edges": [[qs[i], qs[i + 1]] for i in range(n - 1)],
                "depth": n if strategy == "best_chain" else int(math.log2(n)) + 2}

    def estimate_qpu_seconds(self, spec):
        return 2.0 + spec["shots"] * (2 * spec["n_qubits"] + 3) / 10000.0

    def run_experiment(self, spec, backend, calibration, exp_id):
        self.calls.append(spec)
        if self.interrupt_at is not None and len(self.calls) >= self.interrupt_at:
            raise KeyboardInterrupt
        if self.fail_exp:
            raise RuntimeError("simulated core crash")
        n = spec["n_qubits"]
        f = math.exp(-n / 45)
        if spec["layout_strategy"] == "bfs_tree":
            f += 0.02
        if spec["exclude_qubits"]:
            f += 0.01
        if spec["circuit_variant"] == "linear":
            f -= 0.05
        if spec["mode"] == "qpu":
            f *= 0.93
        half = 1.3 / math.sqrt(spec["shots"])
        lay = self.plan_layout(calibration, n, spec["layout_strategy"], spec["root"], spec["exclude_qubits"])
        return {"exp_id": exp_id, "spec": spec, "physical_qubits": lay["qubits"],
                "tree_edges": lay["tree_edges"], "root": lay["root"], "circuit_depth": lay["depth"],
                "two_qubit_gate_count": n - 1, "fidelity": f, "ci_low": f - half, "ci_high": f + half,
                "populations": f + 0.1, "coherence": f - 0.1,
                "parity_curve": [{"phi": 0.0, "parity": f}], "certified": f - half > 0.5,
                "qpu_seconds": 0.8 * self.estimate_qpu_seconds(spec) if spec["mode"] == "qpu" else 0.0,
                "wall_seconds": 0.01, "backend": backend.name, "job_ids": [], "error": None}


def quiet(*_a, **_k):
    pass


def check_events(run_dir: Path, max_llm: int, max_qpu: float) -> list[dict]:
    evs = read_events(run_dir)
    assert evs, "no events written"
    assert [e["seq"] for e in evs] == list(range(len(evs)))
    assert evs[0]["type"] == "run_started"
    assert evs[-1]["type"] == "run_finished"
    assert sum(e["type"] == "run_finished" for e in evs) == 1
    for e in evs:
        assert set(e) == {"ts", "seq", "type", "data"}
        assert e["type"] in EVENT_KEYS, e["type"]
        assert set(e["data"]) == EVENT_KEYS[e["type"]], (e["type"], set(e["data"]))
        if e["type"] == "budget":
            assert e["data"]["llm_calls"] <= max_llm
            assert e["data"]["qpu_seconds"] <= max_qpu + 1e-9
        if e["type"] == "experiment_started":
            assert set(e["data"]["spec"]) == SPEC_KEYS
        if e["type"] == "thought":
            assert e["data"]["text"].strip()
    budget = json.loads((run_dir / "budget.json").read_text())
    assert budget["llm_calls"] <= max_llm and budget["qpu_seconds"] <= max_qpu + 1e-9
    results = [e for e in evs if e["type"] == "experiment_result"]
    for i, e in enumerate(results, 1):
        assert e["data"]["exp_id"] == f"e{i:03d}"
        saved = json.loads((run_dir / "experiments" / f"e{i:03d}.json").read_text())
        assert saved["exp_id"] == f"e{i:03d}"
        assert saved["certified"] == (saved["error"] is None and saved["ci_low"] is not None
                                      and saved["ci_low"] > 0.5)
    assert (run_dir / "calibration.json").exists()
    return evs


# ----------------------------------------------------------------------------- full loops

def test_full_loop_sim_fake(tmp_path):
    core = FakeCore()
    rd = run_agent(mode="sim", llm="fake", max_llm_calls=300, core=core, log=quiet,
                   run_dir=tmp_path / "run-sim")
    evs = check_events(rd, 300, 480)
    res = [e["data"] for e in evs if e["type"] == "experiment_result"]
    assert len(res) >= 5
    assert all(r["mode"] == "sim" for r in res)
    ns = [r["n_qubits"] for r in res]
    assert max(ns) > 16                       # it scaled up
    miles = [e for e in evs if e["type"] == "milestone"]
    assert miles and all("simulator" in m["data"]["text"] for m in miles)
    fin = evs[-1]["data"]
    best_cert = max(r["n_qubits"] for r in res if r["certified"])
    assert fin["best_certified_n"] == best_cert
    assert 24 <= best_cert <= 32              # found the frontier of the fake model (~28)
    assert any(e["type"] == "thought" for e in evs)


def test_full_loop_qpu_fake_hardware_gating(tmp_path):
    core = FakeCore()
    rd = run_agent(mode="qpu", llm="fake", max_llm_calls=300, max_qpu_seconds=480, core=core,
                   log=quiet, run_dir=tmp_path / "run-qpu")
    evs = check_events(rd, 300, 480)
    starts = [e["data"]["spec"] for e in evs if e["type"] == "experiment_started"]
    sim_ok = [s for s, r in zip(starts, [e["data"] for e in evs if e["type"] == "experiment_result"])
              if s["mode"] == "sim" and (r["certified"] or (r["fidelity"] or 0) >= 0.5)]
    hw = [s for s in starts if s["mode"] == "qpu"]
    assert hw, "expected at least one hardware confirmation"
    first_hw = starts.index(hw[0])
    assert all(s["mode"] == "sim" for s in starts[:first_hw])   # sim exploration first
    validated = {config_key(s) for s in sim_ok}
    assert all(config_key(s) in validated for s in hw)        # gate: every hw config sim-validated
    hw_res = [e["data"] for e in evs if e["type"] == "experiment_result" and e["data"]["mode"] == "qpu"]
    assert all(r["qpu_seconds"] > 0 for r in hw_res)
    hw_miles = [e for e in evs if e["type"] == "milestone" and "hardware" in e["data"]["text"]]
    fin = evs[-1]["data"]
    if any(r["certified"] for r in hw_res):
        assert hw_miles
        assert fin["best_certified_n"] == max(r["n_qubits"] for r in hw_res if r["certified"])
    else:
        assert fin["best_certified_n"] is None


def test_tiny_qpu_budget_refuses_hardware(tmp_path):
    rd = run_agent(mode="qpu", llm="fake", max_llm_calls=300, max_qpu_seconds=3.0, core=FakeCore(),
                   log=quiet, run_dir=tmp_path / "r")
    evs = check_events(rd, 300, 3.0)
    assert not [e for e in evs if e["type"] == "experiment_started" and e["data"]["spec"]["mode"] == "qpu"]
    assert evs[-1]["data"]["best_certified_n"] is None


def test_llm_cap_is_hard(tmp_path):
    rd = run_agent(mode="sim", llm="fake", max_llm_calls=4, core=FakeCore(), log=quiet,
                   run_dir=tmp_path / "r")
    evs = check_events(rd, 4, 480)
    assert json.loads((rd / "budget.json").read_text())["llm_calls"] == 4
    assert "LLM call budget exhausted" in evs[-1]["data"]["summary"]


def test_experiment_cap_and_wall_time(tmp_path):
    rd = run_agent(mode="sim", llm="fake", max_experiments=3, core=FakeCore(), log=quiet,
                   run_dir=tmp_path / "a")
    evs = check_events(rd, 300, 480)
    assert sum(e["type"] == "experiment_result" for e in evs) == 3
    rd2 = run_agent(mode="sim", llm="fake", max_hours=0, core=FakeCore(), log=quiet,
                    run_dir=tmp_path / "b")
    evs2 = check_events(rd2, 300, 480)
    assert "wall time" in evs2[-1]["data"]["summary"]


def test_core_crash_never_kills_run(tmp_path):
    rd = run_agent(mode="sim", llm="fake", core=FakeCore(fail_exp=True), log=quiet,
                   run_dir=tmp_path / "r")
    evs = check_events(rd, 300, 480)
    res = [e["data"] for e in evs if e["type"] == "experiment_result"]
    assert res and all(r["error"] and not r["certified"] for r in res)
    assert any(e["type"] == "error" for e in evs)
    assert evs[-1]["data"]["best_certified_n"] is None


def test_core_missing_backend_still_finishes(tmp_path):
    class Broken(FakeCore):
        def get_backend(self, mode, name=None):
            raise RuntimeError("no IBM token")
    rd = run_agent(mode="qpu", llm="fake", core=Broken(), log=quiet, run_dir=tmp_path / "r")
    evs = read_events(rd)
    assert evs[0]["type"] == "run_started" and evs[-1]["type"] == "run_finished"
    assert "setup" in evs[-1]["data"]["summary"]


def test_ctrl_c_writes_run_finished(tmp_path):
    rd = run_agent(mode="sim", llm="fake", core=FakeCore(interrupt_at=2), log=quiet,
                   run_dir=tmp_path / "r")
    evs = read_events(rd)
    assert evs[-1]["type"] == "run_finished"
    assert "interrupted" in evs[-1]["data"]["summary"]


# ----------------------------------------------------------------------------- toolbox gating

def _toolbox(tmp_path, mode="qpu", max_qpu=480.0):
    core = FakeCore()
    budget = Budget(tmp_path, 10, max_qpu)
    events = EventLog(tmp_path)
    backend = core.get_backend(mode)
    cal = core.get_calibration(backend)
    (tmp_path / "experiments").mkdir(exist_ok=True)
    return ToolBox(core=core, backend=backend, calibration=cal, mode=mode, budget=budget,
                   events=events, run_dir=tmp_path, log=quiet), core, budget


def test_hardware_requires_sim_validation(tmp_path):
    tb, core, budget = _toolbox(tmp_path)
    spec = {"n_qubits": 10, "hypothesis": "x"}
    r = tb.call("run_on_hardware", spec)
    assert "refused" in r["error"] and not core.calls
    tb.call("run_simulation", dict(spec, n_qubits=60))        # low fidelity, different config
    r = tb.call("run_on_hardware", spec)
    assert "refused" in r["error"]
    sim = tb.call("run_simulation", spec)
    assert sim["certified"]
    r = tb.call("run_on_hardware", dict(spec, shots=4000))     # shots may differ
    assert r["error"] is None and r["mode"] == "qpu"
    assert budget.qpu_seconds > 0
    r = tb.call("run_on_hardware", dict(spec, layout_strategy="bfs_tree"))  # different config
    assert "refused" in r["error"]


def test_hardware_refused_when_budget_short(tmp_path):
    tb, core, budget = _toolbox(tmp_path, max_qpu=5.0)
    tb.call("run_simulation", {"n_qubits": 10, "hypothesis": "x"})
    r = tb.call("run_on_hardware", {"n_qubits": 10, "hypothesis": "x", "shots": 4000})
    assert "exceeds remaining QPU budget" in r["error"]
    assert budget.qpu_seconds == 0


def test_hardware_tool_hidden_and_blocked_in_sim(tmp_path):
    assert "run_on_hardware" not in [d["name"] for d in tool_declarations("sim")]
    assert "run_on_hardware" in [d["name"] for d in tool_declarations("qpu")]
    tb, core, _ = _toolbox(tmp_path, mode="sim")
    assert "not available" in tb.call("run_on_hardware", {"n_qubits": 4, "hypothesis": "x"})["error"]


def test_malformed_calls_return_errors(tmp_path):
    tb, core, _ = _toolbox(tmp_path)
    assert "unknown tool" in tb.call("frobnicate", {})["error"]
    assert "invalid spec" in tb.call("run_simulation", {"n_qubits": "lots"})["error"]
    assert "invalid spec" in tb.call("run_simulation", {"n_qubits": 999})["error"]
    assert "invalid spec" in tb.call("run_simulation", {"n_qubits": 5, "layout_strategy": "magic"})["error"]
    assert "invalid spec" in tb.call("run_simulation", {"__malformed__": "{"})["error"]
    assert not core.calls


def test_normalize_spec_defaults_and_clamps():
    s, err = normalize_spec({"n_qubits": 12, "shots": 10**9, "exclude_qubits": [3, 3, 1]}, "sim", 156)
    assert err is None and s["shots"] == 8000 and s["exclude_qubits"] == [1, 3]
    assert s["dynamical_decoupling"] and s["mode"] == "sim" and set(s) == SPEC_KEYS


def test_calibration_summary_flags_bad_qubits():
    core = FakeCore()
    summ = summarize_calibration(core.get_calibration(core.get_backend("sim")))
    assert 5 in summ["bad_qubits"] and 17 in summ["bad_qubits"]
    assert len(json.dumps(summ)) < 3000          # compact, not the full dump


# ----------------------------------------------------------------------------- Gemini client

CANNED = {
    "candidates": [{
        "content": {"role": "model", "parts": [
            {"text": "N=16 certified with margin, so I'll push to N=24.", "thought": False},
            {"functionCall": {"name": "run_simulation",
                              "args": {"n_qubits": 24, "layout_strategy": "weighted_tree",
                                       "hypothesis": "should certify"}}},
        ]},
        "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 2100, "candidatesTokenCount": 80, "thoughtsTokenCount": 300},
}


def test_parse_gemini_response():
    r = parse_gemini_response(CANNED)
    assert r.text.startswith("N=16 certified")
    assert len(r.calls) == 1 and r.calls[0].name == "run_simulation"
    assert r.calls[0].args["n_qubits"] == 24
    assert r.finish_reason == "STOP"
    # thought summaries skipped; odd shapes tolerated
    r2 = parse_gemini_response({"candidates": [{"content": {"parts": [
        {"text": "secret plan", "thought": True}, {"functionCall": {"name": "get_history"}}]},
        "finishReason": "STOP"}]})
    assert r2.text == "" and r2.calls[0].name == "get_history" and r2.calls[0].args == {}
    r3 = parse_gemini_response({"promptFeedback": {"blockReason": "SAFETY"}})
    assert not r3.calls and "SAFETY" in r3.finish_reason
    r4 = parse_gemini_response({"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}]})
    assert not r4.calls and r4.finish_reason == "MALFORMED_FUNCTION_CALL"


def test_build_gemini_body():
    req = LLMRequest(system="sys", user_text="digest", tools=tool_declarations("qpu"))
    body = build_gemini_body(req)
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "digest"}]}]
    names = [d["name"] for d in body["tools"][0]["functionDeclarations"]]
    assert {"get_calibration_summary", "run_simulation", "run_on_hardware", "get_history", "finish"} <= set(names)
    assert body["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 384
    json.dumps(body)


class FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        r = self.responses.pop(0) if self.responses else FakeResp(200, CANNED)
        if isinstance(r, Exception):
            raise r
        return r


def test_gemini_retries_and_key_in_header(tmp_path):
    budget = Budget(tmp_path, 5, 0)
    sess = FakeSession([FakeResp(429, {"error": {"message": "slow down"}}),
                        FakeResp(503, {}), FakeResp(200, CANNED)])
    sleeps = []
    llm = GeminiLLM(budget, api_key="TESTKEY123", model="gemini-2.5-flash", session=sess,
                    sleep=sleeps.append)
    r = llm.generate(LLMRequest("s", "u", tool_declarations("sim")))
    assert r.calls[0].name == "run_simulation"
    assert len(sess.requests) == 3 and len(sleeps) == 2
    assert budget.llm_calls == 1                       # one logical call charged once
    for rq in sess.requests:
        assert "TESTKEY123" not in rq["url"] and rq["headers"]["x-goog-api-key"] == "TESTKEY123"
        assert rq["url"].endswith("/models/gemini-2.5-flash:generateContent") and rq["timeout"]
    assert "TESTKEY123" not in repr(llm)
    assert llm.usage.prompt_tokens == 2100 and llm.usage.output_tokens == 380


def test_gemini_gives_up_and_4xx(tmp_path):
    import requests as rq
    budget = Budget(tmp_path, 5, 0)
    llm = GeminiLLM(budget, api_key="k", session=FakeSession([rq.ConnectionError("x")] * 3),
                    sleep=lambda s: None)
    with pytest.raises(LLMError):
        llm.generate(LLMRequest("s", "u", []))
    llm2 = GeminiLLM(budget, api_key="k", session=FakeSession([FakeResp(403, {"error": {"message": "bad key"}})]),
                     sleep=lambda s: None)
    with pytest.raises(LLMError, match="403"):
        llm2.generate(LLMRequest("s", "u", []))
    # thinking config unsupported -> drops it and retries
    sess = FakeSession([FakeResp(400, {"error": {"message": "thinking_budget is not supported"}}),
                        FakeResp(200, CANNED)])
    llm3 = GeminiLLM(budget, api_key="k", session=sess, sleep=lambda s: None)
    llm3.generate(LLMRequest("s", "u", []))
    assert "thinkingConfig" not in sess.requests[1]["json"]["generationConfig"]


def test_gemini_budget_checked_before_network(tmp_path):
    budget = Budget(tmp_path, 0, 0)
    sess = FakeSession([])
    llm = GeminiLLM(budget, api_key="k", session=sess, sleep=lambda s: None)
    with pytest.raises(BudgetExceeded):
        llm.generate(LLMRequest("s", "u", []))
    assert sess.requests == []
    # dollar cap: once estimated spend reaches max_usd, no more calls
    (tmp_path / "b2").mkdir()
    llm2 = GeminiLLM(Budget(tmp_path / "b2", 100, 0), api_key="k", session=sess,
                     sleep=lambda s: None, max_usd=0.0001)
    llm2.generate(LLMRequest("s", "u", []))           # canned usage ~ $0.0016 > cap
    with pytest.raises(BudgetExceeded, match="spend"):
        llm2.generate(LLMRequest("s", "u", []))
    assert len(sess.requests) == 1


def test_loop_with_gemini_client_and_bad_replies(tmp_path):
    """Gemini that returns junk then 5xx forever: loop recovers via autopilot and still finishes."""
    junk = {"candidates": [{"content": {"parts": [{"text": "hmm"}]}, "finishReason": "STOP"}]}
    sess = FakeSession([FakeResp(200, junk)] * 7 + [FakeResp(500, {})] * 200)
    llm = GeminiLLM(Budget(tmp_path, 1, 0), api_key="k", session=sess, sleep=lambda s: None)
    rd = run_agent(mode="sim", llm=llm, max_llm_calls=40, core=FakeCore(), log=quiet,
                   run_dir=tmp_path / "r")
    evs = check_events(rd, 40, 480)
    assert any("autopilot" in e["data"]["text"] for e in evs if e["type"] == "thought")
    assert any(e["type"] == "experiment_result" for e in evs)
    assert any("consecutive LLM failures" in e["data"]["text"] for e in evs if e["type"] == "error")


def test_loop_with_gemini_happy_path(tmp_path):
    """Canned Gemini replies drive real experiments; the text part becomes a thought event."""
    def reply(name, args, text):
        return FakeResp(200, {"candidates": [{"content": {"parts": [
            {"text": text}, {"functionCall": {"name": name, "args": args}}]}, "finishReason": "STOP"}]})
    sess = FakeSession([
        reply("get_calibration_summary", {}, "Look at the chip first."),
        reply("run_simulation", {"n_qubits": 12, "hypothesis": "baseline"}, "Baseline at N=12."),
        reply("finish", {"summary": "Certified N=12 on the simulator."}, "Done."),
    ])
    llm = GeminiLLM(Budget(tmp_path, 1, 0), api_key="k", session=sess, sleep=lambda s: None)
    rd = run_agent(mode="sim", llm=llm, max_llm_calls=10, core=FakeCore(), log=quiet,
                   run_dir=tmp_path / "r")
    evs = check_events(rd, 10, 480)
    assert [e["data"]["text"] for e in evs if e["type"] == "thought"][:2] == ["Look at the chip first.", "Baseline at N=12."]
    assert evs[-1]["data"]["best_certified_n"] == 12
    assert "Certified N=12" in evs[-1]["data"]["summary"]
    assert json.loads((rd / "budget.json").read_text())["llm_calls"] == 3
    # digest stays compact (stateless conversation)
    assert all(len(r["json"]["contents"]) == 1 for r in sess.requests)

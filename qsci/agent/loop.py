"""The scientist loop: digest -> LLM -> one tool call -> events -> repeat.

Guarantees:
  * run_finished is ALWAYS emitted (normal finish, budget/time/experiment caps, LLM failure,
    core failure, Ctrl-C / SIGTERM).
  * LLM calls never exceed Budget.max_llm_calls (charged before each call).
  * Hardware runs never start unless Budget.can_spend_qpu(estimate) and the config was
    validated on the simulator (enforced in ToolBox).
  * If Gemini keeps failing, a scripted autopilot takes over so the night isn't wasted.
"""
from __future__ import annotations

import json
import signal
import time
from collections import deque
from pathlib import Path
from typing import Callable

from qsci.budget import Budget, BudgetExceeded
from qsci.events import EventLog, new_run_dir

from .llm import FakeLLM, GeminiLLM, LLMRequest
from .policy import AgentState, ScriptedPolicy
from .prompts import GOAL, SYSTEM_PROMPT, format_ci, history_row
from .tools import ToolBox, tool_declarations

MAX_CONSECUTIVE_LLM_FAILURES = 3
MAX_IDLE_TURNS = 6          # turns without a successful experiment before autopilot steps in
HISTORY_ROWS_IN_DIGEST = 18


def _result_text(result: dict, limit: int = 1800) -> str:
    s = json.dumps(result, default=str, separators=(",", ":"))
    return s if len(s) <= limit else s[:limit] + "...(truncated)"


class Agent:
    def __init__(self, *, toolbox: ToolBox, llm, budget: Budget, events: EventLog, mode: str,
                 backend_name: str, deadline: float, log: Callable[[str], None] = print) -> None:
        self.tb = toolbox
        self.llm = llm
        self.budget = budget
        self.events = events
        self.mode = mode
        self.backend_name = backend_name
        self.deadline = deadline
        self.t_start = time.time()
        self.log = log
        self.autopilot = ScriptedPolicy()
        self.hw_autopilot = ScriptedPolicy(max_sim_experiments=0)
        self.force_autopilot = llm is None
        self.thoughts: deque[str] = deque(maxlen=3)
        self.last_action: tuple[str, dict, dict] | None = None
        self.stop_reason = ""

    # ------------------------------------------------------------ state for prompt / policy
    def state(self) -> AgentState:
        snap = self.budget.snapshot()
        return AgentState(
            mode=self.mode, num_qubits=self.tb.num_qubits, history=list(self.tb.history),
            bad_qubits=self.tb.bad_qubits(), calibration_seen=self.tb.calibration_seen,
            qpu_remaining_s=snap["max_qpu_seconds"] - snap["qpu_seconds"],
            llm_calls_left=snap["max_llm_calls"] - snap["llm_calls"],
            estimate_qpu=self._safe_estimate, hw_available=self.mode == "qpu",
            extra={"hw_refused": set(self.tb.hw_refused)})

    def _safe_estimate(self, spec: dict) -> float:
        try:
            return self.tb.estimate_qpu(spec)
        except Exception:
            return float("inf")

    def digest(self) -> str:
        snap = self.budget.snapshot()
        hrs = (time.time() - self.t_start) / 3600
        total_h = (self.deadline - self.t_start) / 3600
        lines = [f"GOAL: {GOAL}.",
                 f"MODE: {self.mode} | backend: {self.backend_name} ({self.tb.num_qubits} qubits)"
                 + (" | hardware tool available" if self.mode == "qpu" else " | simulator only")]
        lines.append(f"BUDGET: LLM calls {snap['llm_calls']}/{snap['max_llm_calls']} used"
                     + (f"; QPU {snap['qpu_seconds']:.1f}/{snap['max_qpu_seconds']:.0f} s used"
                        if self.mode == "qpu" else "")
                     + f"; wall {hrs:.2f}/{total_h:.1f} h; experiments "
                     f"{len(self.tb.history)}/{self.tb.max_experiments}")
        for mode in (("sim", "qpu") if self.mode == "qpu" else ("sim",)):
            b = self.tb.best[mode]
            label = "hardware" if mode == "qpu" else "simulator"
            lines.append(f"RECORD ({label}): " + (
                f"N={b['spec']['n_qubits']} certified, {format_ci(b)} ({b['exp_id']})" if b else "none certified yet"))
        if self.tb.calibration_seen:
            cs = self.tb.calibration_summary()
            lines.append(f"CALIBRATION: medians {cs.get('median')}; bad_qubits {cs.get('bad_qubits')}; "
                         f"suggested_roots {cs.get('suggested_roots')}")
        hist = self.tb.history[-HISTORY_ROWS_IN_DIGEST:]
        if hist:
            lines.append(f"HISTORY (last {len(hist)} of {len(self.tb.history)}):")
            lines += ["  " + history_row(r) for r in hist]
        else:
            lines.append("HISTORY: no experiments yet.")
        if self.thoughts:
            lines.append("YOUR RECENT REASONING: " + " | ".join(self.thoughts))
        if self.last_action:
            name, args, result = self.last_action
            lines.append(f"LAST ACTION: {name}({_result_text(args, 400)}) -> {_result_text(result)}")
        if self.mode == "qpu" and snap["max_llm_calls"] - snap["llm_calls"] <= 30 \
                and not any(r["spec"].get("mode") == "qpu" for r in self.tb.history):
            lines.append("WARNING: few LLM calls left and no hardware run yet -- move to hardware now.")
        if snap["max_llm_calls"] - snap["llm_calls"] <= 3:
            lines.append("WARNING: LLM budget almost exhausted -- call finish soon.")
        lines.append("Reply with 1-3 sentences of reasoning as text, then call exactly one tool.")
        return "\n".join(lines)

    # ------------------------------------------------------------ main loop
    def run(self) -> None:
        idle = 0
        failures = 0
        max_turns = self.budget.max_llm_calls + self.tb.max_experiments + 50  # hard termination
        for _turn in range(max_turns):
            if self.tb.finished_summary is not None:
                self.stop_reason = "agent called finish"
                return
            if time.time() >= self.deadline:
                self.stop_reason = "max wall time reached"
                return
            if self.tb.experiments_left() <= 0:
                self.stop_reason = "experiment cap reached"
                return
            snap = self.budget.snapshot()
            llm_left = snap["max_llm_calls"] - snap["llm_calls"]

            policy = None
            if self.force_autopilot or idle >= MAX_IDLE_TURNS:
                policy = self.autopilot
            elif llm_left <= 0:
                if self.mode == "qpu" and not any(r["spec"].get("mode") == "qpu" for r in self.tb.history) \
                        and self.tb.sim_any_validated():
                    policy = self.hw_autopilot
                    if not self.force_autopilot:
                        self._thought("[autopilot] LLM call budget is exhausted; the scripted "
                                      "policy will confirm the best simulator result on hardware.")
                        self.force_autopilot = True
                        self.autopilot = self.hw_autopilot
                else:
                    self.stop_reason = "LLM call budget exhausted"
                    return

            if policy is not None:
                text, name, args = policy.next_action(self.state())
                text = "[autopilot] " + text
            else:
                req = LLMRequest(system=SYSTEM_PROMPT, user_text=self.digest(),
                                 tools=tool_declarations(self.mode), state=self.state())
                try:
                    reply = self.llm.generate(req)
                except BudgetExceeded:
                    self.tb.emit_budget()
                    continue  # loop re-checks; hands hardware phase to autopilot or stops
                except Exception as e:
                    failures += 1
                    self.tb.emit_budget()
                    self.events.emit("error", text=f"LLM call failed: {str(e)[:300]}")
                    self.log(f"  LLM error: {str(e)[:150]}")
                    if failures >= MAX_CONSECUTIVE_LLM_FAILURES:
                        self.force_autopilot = True
                        self.events.emit("error", text=f"{failures} consecutive LLM failures; "
                                         "switching to scripted autopilot for the rest of the run")
                    continue
                failures = 0
                self.tb.emit_budget()
                if not reply.calls:
                    if reply.text:
                        self._thought(reply.text)
                    idle += 1
                    self.last_action = ("(no tool call)", {}, {
                        "error": f"you did not call a tool (finish_reason={reply.finish_reason}); "
                                 "you must call exactly one tool each turn"})
                    continue
                call = reply.calls[0]
                name, args, text = call.name, call.args, reply.text
                if not text:
                    hyp = args.get("hypothesis") if isinstance(args, dict) else None
                    text = hyp or f"Calling {name}."

            self._thought(text)
            result = self.tb.call(name, args)
            if name in ("run_simulation", "run_on_hardware") and not result.get("error"):
                idle = 0
            else:
                idle += 1
            if policy is not None and name == "finish" and self.tb.finished_summary is None:
                self.tb.finished_summary = args.get("summary", "")  # autopilot finish is final
            self.last_action = (name, args, result)
        self.stop_reason = "turn limit reached"

    def _thought(self, text: str) -> None:
        text = " ".join(str(text).split())[:700]
        if not text:
            return
        self.thoughts.append(text[:240])
        self.events.emit("thought", text=text)
        self.log(f"> {text}")


# ------------------------------------------------------------------------- entry point

def _verified_line(tb: ToolBox, mode: str) -> str:
    parts = []
    for m in (("sim", "qpu") if mode == "qpu" else ("sim",)):
        b = tb.best[m]
        label = "hardware" if m == "qpu" else "simulator"
        parts.append(f"{label}: " + (f"N={b['spec']['n_qubits']} certified ({format_ci(b)}, {b['exp_id']})"
                                     if b else "nothing certified"))
    return "Verified from data -- " + "; ".join(parts) + "."


def run_agent(*, mode: str, llm: str | object = "gemini", max_llm_calls: int = 300,
              max_qpu_seconds: float = 480.0, backend_name: str | None = None,
              max_hours: float = 7.0, max_experiments: int = 80, runs_root: Path = Path("runs"),
              core=None, log: Callable[[str], None] = print, run_dir: Path | None = None) -> Path:
    if mode not in ("sim", "qpu"):
        raise ValueError("mode must be 'sim' or 'qpu'")
    run_dir = Path(run_dir) if run_dir else new_run_dir(mode, Path(runs_root))
    (run_dir / "experiments").mkdir(parents=True, exist_ok=True)
    events = EventLog(run_dir)
    budget = Budget(run_dir, max_llm_calls, max_qpu_seconds)
    planner = llm if isinstance(llm, str) else str(getattr(llm, "name", type(llm).__name__))
    deadline = time.time() + max_hours * 3600
    tb: ToolBox | None = None
    agent: Agent | None = None
    summary_extra = ""
    started = False
    prev_handler = None
    try:
        try:
            prev_handler = signal.signal(signal.SIGTERM, _raise_interrupt)
        except (ValueError, OSError):
            pass  # not main thread
        try:
            if core is None:
                import qsci.core as core  # noqa: PLW0127 - lazy: core may be developed in parallel
            backend = core.get_backend(mode, backend_name)
            bname = getattr(backend, "name", None)
            bname = bname() if callable(bname) else bname
            bname = str(bname or backend_name or "unknown")
        except Exception as e:
            events.emit("run_started", run_id=run_dir.name, mode=mode, backend=backend_name or "unknown",
                        goal=GOAL, planner=planner, max_llm_calls=budget.max_llm_calls, max_qpu_seconds=budget.max_qpu_seconds)
            started = True
            raise _SetupError(f"could not load backend: {type(e).__name__}: {e}") from e
        events.emit("run_started", run_id=run_dir.name, mode=mode, backend=bname, goal=GOAL, planner=planner,
                    max_llm_calls=budget.max_llm_calls, max_qpu_seconds=budget.max_qpu_seconds)
        started = True
        log(f"Q-Scientist run {run_dir.name} | mode={mode} backend={bname}")
        try:
            calibration = core.get_calibration(backend)
            (run_dir / "calibration.json").write_text(json.dumps(calibration, indent=2, default=str))
        except Exception as e:
            raise _SetupError(f"could not get calibration: {type(e).__name__}: {e}") from e
        tb = ToolBox(core=core, backend=backend, calibration=calibration, mode=mode, budget=budget,
                     events=events, run_dir=run_dir, max_experiments=max_experiments, log=log)
        tb.emit_budget()
        client = None
        if isinstance(llm, str):
            try:
                client = FakeLLM(budget) if llm == "fake" else GeminiLLM(budget)
            except Exception as e:
                events.emit("error", text=f"LLM unavailable ({e}); running scripted autopilot")
        else:
            client = llm
            if hasattr(client, "budget"):
                client.budget = budget  # injected clients must charge THIS run's budget
        agent = Agent(toolbox=tb, llm=client, budget=budget, events=events, mode=mode,
                      backend_name=bname, deadline=deadline, log=log)
        if client is not None:
            log(f"LLM: {getattr(client, 'name', type(client).__name__)} | cap {budget.max_llm_calls} calls"
                + (f" | QPU cap {budget.max_qpu_seconds:.0f}s" if mode == "qpu" else ""))
        agent.run()
    except KeyboardInterrupt:
        summary_extra = "Run interrupted by operator."
        events.emit("error", text="interrupted (Ctrl-C / SIGTERM)")
    except _SetupError as e:
        summary_extra = f"Run aborted during setup: {e}"
        events.emit("error", text=str(e)[:500])
        log(f"SETUP ERROR: {e}")
    except Exception as e:  # last line of defence
        summary_extra = f"Run aborted by unexpected error: {type(e).__name__}: {str(e)[:300]}"
        events.emit("error", text=summary_extra)
        log(summary_extra)
    finally:
        try:
            if prev_handler is not None:
                signal.signal(signal.SIGTERM, prev_handler)
        except (ValueError, OSError):
            pass
        _finish_run(run_dir, events, budget, tb, agent, mode, summary_extra, started, log)
    return run_dir


class _SetupError(Exception):
    pass


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def _finish_run(run_dir, events, budget, tb, agent, mode, summary_extra, started, log) -> None:
    if not started:
        events.emit("run_started", run_id=run_dir.name, mode=mode, backend="unknown", goal=GOAL,
                    max_llm_calls=budget.max_llm_calls, max_qpu_seconds=budget.max_qpu_seconds)
    best = tb.best.get(mode) if tb else None
    parts = []
    if tb and tb.finished_summary:
        parts.append(tb.finished_summary.strip())
    if agent and agent.stop_reason and agent.stop_reason != "agent called finish":
        parts.append(f"Stopped: {agent.stop_reason}.")
    if summary_extra:
        parts.append(summary_extra)
    if tb:
        parts.append(_verified_line(tb, mode))
    summary = " ".join(parts) or "Run finished."
    if tb:
        tb.emit_budget()
    events.emit("run_finished", best_certified_n=best["spec"]["n_qubits"] if best else None,
                best_exp_id=best["exp_id"] if best else None, summary=summary)
    usage = getattr(getattr(agent, "llm", None), "usage", None)
    if usage is not None:
        try:
            (run_dir / "llm_usage.json").write_text(json.dumps(usage.snapshot(), indent=2))
        except Exception:
            pass
    snap = budget.snapshot()
    log(f"Run finished: {summary}")
    log(f"Budget: {snap['llm_calls']}/{snap['max_llm_calls']} LLM calls, "
        f"{snap['qpu_seconds']:.1f}/{snap['max_qpu_seconds']:.0f} QPU s"
        + (f", est LLM cost ${usage.est_cost_usd():.3f}" if usage is not None and usage.calls else "")
        + f" | {run_dir}")

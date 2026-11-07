"""LLM clients: Gemini over plain REST, plus a scripted FakeLLM with the same interface.

Interface (both clients):
    reply = llm.generate(LLMRequest(system=..., user_text=..., tools=[...], state=...))
    reply.text   -> str        model's visible reasoning ("" if none)
    reply.calls  -> list[ToolCall(name, args)]
    reply.usage  -> dict token counts

Every generate() call charges Budget.charge_llm() BEFORE doing anything, so the hard cap can
never be exceeded (BudgetExceeded propagates to the caller).
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash"
# Rough gemini-2.5-flash list prices (USD per 1M tokens). Only used for the cost estimate print.
PRICE_IN_PER_M = 0.30
PRICE_OUT_PER_M = 2.50


class LLMError(Exception):
    """Unrecoverable LLM failure for this call (after retries)."""


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class LLMRequest:
    system: str
    user_text: str
    tools: list[dict]                 # list of functionDeclaration dicts
    state: Any = None                 # structured agent state (used by FakeLLM / policy)


@dataclass
class LLMReply:
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None


class UsageMeter:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def add(self, usage: dict) -> None:
        self.calls += 1
        self.prompt_tokens += int(usage.get("promptTokenCount") or 0)
        self.output_tokens += int(usage.get("candidatesTokenCount") or 0)
        self.output_tokens += int(usage.get("thoughtsTokenCount") or 0)

    def est_cost_usd(self) -> float:
        return (self.prompt_tokens * PRICE_IN_PER_M + self.output_tokens * PRICE_OUT_PER_M) / 1e6

    def snapshot(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens, "est_cost_usd": round(self.est_cost_usd(), 4)}


# ---------------------------------------------------------------- request / response helpers

def build_gemini_body(req: LLMRequest, *, max_output_tokens: int = 1024,
                      thinking_budget: int | None = 384) -> dict:
    body: dict = {
        "systemInstruction": {"parts": [{"text": req.system}]},
        "contents": [{"role": "user", "parts": [{"text": req.user_text}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": max_output_tokens},
    }
    if thinking_budget is not None:
        body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    if req.tools:
        body["tools"] = [{"functionDeclarations": req.tools}]
        body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
    return body


def parse_gemini_response(payload: dict) -> LLMReply:
    """Turn a generateContent JSON response into an LLMReply. Never raises on odd shapes."""
    reply = LLMReply(usage=dict(payload.get("usageMetadata") or {}))
    cands = payload.get("candidates") or []
    if not cands:
        block = (payload.get("promptFeedback") or {}).get("blockReason")
        reply.finish_reason = f"NO_CANDIDATES{(': ' + block) if block else ''}"
        return reply
    cand = cands[0] or {}
    reply.finish_reason = cand.get("finishReason")
    texts = []
    for part in ((cand.get("content") or {}).get("parts") or []):
        if not isinstance(part, dict) or part.get("thought"):
            continue  # skip internal thought summaries
        if "functionCall" in part:
            fc = part.get("functionCall") or {}
            name = fc.get("name")
            args = fc.get("args")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"__malformed__": args}
            if not isinstance(args, dict):
                args = {}
            if isinstance(name, str) and name:
                reply.calls.append(ToolCall(name=name, args=args))
        elif isinstance(part.get("text"), str):
            texts.append(part["text"])
    reply.text = " ".join(t.strip() for t in texts if t.strip())
    return reply


# ---------------------------------------------------------------- Gemini client

class GeminiLLM:
    def __init__(self, budget, *, api_key: str | None = None, model: str | None = None,
                 timeout: float = 60.0, max_attempts: int = 3,
                 session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 thinking_budget: int | None = None, max_usd: float | None = None) -> None:
        self.budget = budget
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or ""
        if not self._api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.session = session or requests.Session()
        self.sleep = sleep
        tb = thinking_budget if thinking_budget is not None else os.environ.get("GEMINI_THINKING_BUDGET", "384")
        self.thinking_budget: int | None = int(tb)
        self.usage = UsageMeter()
        # second, dollar-denominated cap (estimated from token usage at list price)
        self.max_usd = float(max_usd if max_usd is not None else os.environ.get("GEMINI_MAX_USD", "1.0"))
        self.name = f"gemini:{self.model}"

    def __repr__(self) -> str:  # never expose the key
        return f"GeminiLLM(model={self.model!r})"

    def generate(self, req: LLMRequest) -> LLMReply:
        from qsci.budget import BudgetExceeded
        if self.usage.est_cost_usd() >= self.max_usd:
            raise BudgetExceeded(f"estimated Gemini spend reached ${self.max_usd:.2f}")
        self.budget.charge_llm()  # raises BudgetExceeded before any network call
        url = GEMINI_URL.format(model=self.model)
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        last_err = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            body = build_gemini_body(req, thinking_budget=self.thinking_budget)
            try:
                resp = self.session.post(url, headers=headers, json=body, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = f"network error: {type(e).__name__}"
                self._backoff(attempt, None)
                continue
            status = resp.status_code
            if status == 200:
                try:
                    payload = resp.json()
                except ValueError:
                    last_err = "invalid JSON from Gemini"
                    self._backoff(attempt, None)
                    continue
                reply = parse_gemini_response(payload)
                self.usage.add(reply.usage)
                return reply
            detail = _error_detail(resp)
            if status == 400 and self.thinking_budget is not None and "thinking" in detail.lower():
                # model doesn't support thinkingBudget -> drop it and retry immediately
                self.thinking_budget = None
                last_err = f"HTTP 400 (thinking config unsupported): {detail}"
                continue
            if status == 429 or status >= 500:
                last_err = f"HTTP {status}: {detail}"
                self._backoff(attempt, resp.headers.get("Retry-After"))
                continue
            raise LLMError(f"HTTP {status}: {detail}")  # 4xx: retrying won't help
        raise LLMError(f"Gemini failed after {self.max_attempts} attempts: {last_err}")

    def _backoff(self, attempt: int, retry_after: str | None) -> None:
        if attempt >= self.max_attempts:
            return
        delay = 2.0 ** attempt + random.uniform(0, 1)
        if retry_after:
            try:
                delay = max(delay, min(float(retry_after), 60.0))
            except ValueError:
                pass
        self.sleep(delay)


def _error_detail(resp) -> str:
    try:
        msg = (resp.json().get("error") or {}).get("message") or ""
    except Exception:
        msg = (getattr(resp, "text", "") or "")
    return str(msg)[:200]


# ---------------------------------------------------------------- Fake (scripted) client

class FakeLLM:
    """Deterministic scripted scientist. Uses the same budget accounting as the real client."""

    def __init__(self, budget, policy=None) -> None:
        from .policy import ScriptedPolicy
        self.budget = budget
        self.policy = policy or ScriptedPolicy()
        self.usage = UsageMeter()
        self.name = "fake"

    def generate(self, req: LLMRequest) -> LLMReply:
        self.budget.charge_llm()
        text, name, args = self.policy.next_action(req.state)
        reply = LLMReply(text=text, calls=[ToolCall(name, args)], finish_reason="STOP",
                         usage={"promptTokenCount": 0, "candidatesTokenCount": 0})
        self.usage.add(reply.usage)
        return reply

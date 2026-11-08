"""Optional Gemini polish of the abstract and discussion.

Guarantees: at most `max_calls` requests; the API key is only ever sent in the x-goog-api-key header
and never logged; any rewrite that introduces a number not present in the draft is rejected.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import requests

NUM_RE = re.compile(r"(?<![A-Za-z])[-−]?\d+(?:\.\d+)?")

SYSTEM = (
    "You are the copy editor of a short physics paper written by an autonomous AI scientist. "
    "Rewrite the given section so it reads like a polished arXiv paper: precise, sober, no hype, "
    "no marketing words, no first-person plural excess. HARD RULES: (1) keep every number exactly "
    "as written, digit for digit; (2) do not add any number, claim, citation or result that is not "
    "in the draft; (3) keep any caveats; (4) plain text only, no markdown, no headings; separate "
    "paragraphs with a blank line; (5) similar length to the draft."
)


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace("−", "-").lstrip("-") for m in NUM_RE.finditer(text)}


def numbers_preserved(draft: str, rewrite: str) -> bool:
    return _numbers(rewrite) <= _numbers(draft)


class Polisher:
    def __init__(self, run_dir: Path, enabled: bool = True, max_calls: int = 2):
        self.key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.model = os.environ.get("GEMINI_MODEL", "").strip() or "gemini-2.5-flash"
        self.enabled = enabled and bool(self.key)
        self.max_calls = max_calls
        self.calls = 0
        self.used = False
        self.notes: list[str] = []
        self.budget = None
        bpath = Path(run_dir) / "budget.json"
        if self.enabled and bpath.exists():
            try:
                import json

                from qsci.budget import Budget

                b = json.loads(bpath.read_text())
                self.budget = Budget(run_dir, int(b.get("max_llm_calls", 0)), float(b.get("max_qpu_seconds", 0)))
            except Exception as exc:  # budget is best-effort for the paper writer
                self.notes.append(f"budget unavailable ({type(exc).__name__})")

    def polish(self, section: str, draft: str, facts_summary: str) -> str:
        if not self.enabled or self.calls >= self.max_calls:
            return draft
        if self.budget is not None:
            from qsci.budget import BudgetExceeded

            try:
                self.budget.charge_llm()
            except BudgetExceeded:
                self.notes.append("LLM budget exhausted; kept template text")
                self.enabled = False
                return draft
        self.calls += 1
        base = (
            f"Section: {section}\n\nVerified facts (computed from data, authoritative):\n{facts_summary}\n\n"
            f"Draft to rewrite:\n<<<\n{draft}\n>>>"
        )
        feedback = ""
        for attempt in range(2):  # one retry, told exactly which numbers it may not introduce
            if attempt and self.budget is not None:  # the retry is a real model call: meter it too
                from qsci.budget import BudgetExceeded
                try:
                    self.budget.charge_llm()
                except BudgetExceeded:
                    self.notes.append(f"{section}: no budget for a retry; kept template text")
                    return draft
            text = self._call(section, base + feedback)
            if text is None:
                return draft
            if not text or len(text) < 0.5 * len(draft) or len(text) > 1.8 * len(draft):
                self.notes.append(f"{section}: rewrite length out of range; kept template text")
                return draft
            extra = _numbers(text) - _numbers(draft)
            if not extra:
                break
            feedback = (f"\n\nYour previous rewrite was rejected because it contained numbers not written exactly that "
                        f"way in the draft: {sorted(extra)}. Copy every number exactly as the draft writes it "
                        f"(same digits, same rounding, same fractions such as 1/2); do not convert or round.")
        else:
            self.notes.append(f"{section}: rewrite introduced new numbers; kept template text")
            return draft
        self.used = True
        return text

    def _call(self, section: str, prompt: str) -> str | None:
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self.key, "Content-Type": "application/json"},
                json={
                    "systemInstruction": {"parts": [{"text": SYSTEM}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 8192,
                                         "thinkingConfig": {"thinkingBudget": 1024}},
                },
                timeout=60,
            )
            if r.status_code != 200:
                self.notes.append(f"{section}: Gemini HTTP {r.status_code}; kept template text")
                return None
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        except Exception as exc:
            self.notes.append(f"{section}: Gemini call failed ({type(exc).__name__}); kept template text")
            return None
        return text.strip().strip("<>").strip()

# Q-Scientist — an autonomous AI physicist on IBM quantum hardware

An AI agent (Gemini) runs GHZ-entanglement experiments on its own — noisy simulator first,
then a real IBM Heron QPU — trying to **certify genuine multipartite entanglement
(fidelity lower confidence bound > 0.5) on as many qubits as possible**. It reasons out loud,
respects a hard QPU/LLM budget, and writes its own paper at the end.

## Run
```bash
cp .env.example .env          # add GEMINI_API_KEY, IBM_QUANTUM_TOKEN, IBM_QUANTUM_INSTANCE
uv run python -m scripts.run_agent --mode sim            # free: simulator only
uv run python -m scripts.run_agent --mode qpu            # real hardware (budget-capped)
uv run python -m scripts.run_agent --mode sim --llm fake # no keys needed: scripted policy
uv run python -m scripts.run_sweep --mode qpu --ns 16,24,32   # no-AI fallback
uv run python -m scripts.write_paper runs/<run_id>        # paper.html + paper.pdf
uv run python -m scripts.serve_ui                         # mission control at http://localhost:8765
uv run python -m pytest -q                                # 60 tests
```

## How it works
- `qsci/core` — calibration → error-weighted GHZ tree layout → circuits → fidelity
  F = (P₀…₀ + P₁…₁)/2 + C/2 from populations + 2N+2 parity-oscillation circuits, bootstrap CIs,
  readout mitigation; one SamplerV2 job per hardware experiment. Fast Pauli-frame noisy simulator.
- `qsci/agent` — Gemini function-calling loop; hardware is gated on a certified simulation and a budget check;
  certification is recomputed from data so the model can't over-claim.
- `ui/` — mission-control replay of the whole night. `qsci/paper` — paper generated from the run log.

See `CONTRACTS.md` for data formats.

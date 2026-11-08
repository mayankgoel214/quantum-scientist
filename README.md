# Q-Scientist — an autonomous AI physicist on IBM quantum hardware

**Live:** https://q-scientist.vercel.app (replay of the real ibm_fez run) · **Paper:** [runs/20261107-220808-qpu/paper/paper.pdf](runs/20261107-220808-qpu/paper/paper.pdf)

Publish a static snapshot: `uv run python -m scripts.export_static && cd site && vercel deploy --prod`


An AI agent (Gemini) runs GHZ-entanglement experiments on its own — noisy simulator first,
then a real IBM Heron QPU — trying to **certify genuine multipartite entanglement
(fidelity lower confidence bound > 0.5) on as many qubits as possible**. It reasons out loud,
respects a hard QPU/LLM budget, and writes its own paper at the end.

## Results on real hardware (IBM `ibm_fez`, 156-qubit Heron)

| GHZ size · layout | Twin prediction (frozen before the run) | Hardware fidelity [95% CI] | Genuinely entangled? |
|---|---|---|---|
| 16 · error-aware tree | 0.906 | **0.841** [0.820, 0.862] | ✅ certified |
| 24 · error-aware tree | 0.866 | **0.723** [0.700, 0.746] | ✅ certified |
| 24 · textbook chain | 0.705 | 0.107 [0.031, 0.163] | ❌ |
| 32 · error-aware tree | 0.796 | 0.537 [0.400, 0.641] | ❌ (lower bound below 0.5) |
| **40 · error-aware tree** | 0.727 | **0.654** [0.520, 0.788] | ✅ **certified (record)** |
| 48 · error-aware tree | 0.648 | 0.455 [0.344, 0.555] | ❌ |
| 56 · error-aware tree | 0.604 | 0.319 [0.196, 0.465] | ❌ |
| 60 · error-aware tree | 0.642 | 0.244 [0.031, 0.394] | ❌ |

All 8 experiments were designed by Gemini in one batch and queued behind ~20,000 public jobs; they returned overnight.
**Findings:**
- **Layout is the difference between entangled and not.** At 24 qubits the agent's error-aware tree reached F = 0.723,
  while the textbook linear CNOT chain on the same chip collapsed to 0.107.
- **Record: 40 qubits certified**, but by a narrow margin (lower bound 0.520) and 32 qubits did not certify, so the
  agent's round 2 re-runs the record with 2× shots as an independent confirmation (queued on IBM; see `pending.json`).
- **The digital twin is honest about its limits:** it called certification right in 3 of 8 cases and overestimated
  fidelity more as N grew, pointing to correlated noise that per-gate calibration data doesn't capture.

207 s of QPU used, out of 600 s free.

![Fidelity vs N](runs/20261107-220808-qpu/paper/fig_fidelity_vs_n.png)
![GHZ tree on the chip](runs/20261107-220808-qpu/paper/fig_chip_map.png)

## Run
```bash
cp .env.example .env          # add GEMINI_API_KEY, IBM_QUANTUM_TOKEN, IBM_QUANTUM_INSTANCE
uv run python -m scripts.run_agent --mode sim            # free: simulator only
uv run python -m scripts.run_agent --mode qpu            # real hardware (budget-capped)
uv run python -m scripts.run_agent --mode sim --llm fake # no keys needed: scripted policy
uv run python -m scripts.run_sweep --mode qpu --ns 16,24,32   # no-AI fallback

# Queue-aware hardware run (what we used on ibm_fez, where the open queue holds ~20k jobs):
uv run python -m scripts.qpu_batch submit --backend ibm_fez           # Gemini designs a batch, digital twin predicts, all jobs submitted at once
uv run python -m scripts.qpu_batch collect runs/<run_id> --watch      # gather results as IBM finishes them
uv run python -m scripts.qpu_batch extend runs/<run_id>               # round 2: agent reads hardware results, confirms its record
uv run python -m scripts.check_keys                                   # which keys are set (never prints values)
uv run python -m scripts.write_paper runs/<run_id>        # paper.html + paper.pdf
uv run python -m scripts.serve_ui                         # mission control at http://localhost:8765
uv run python -m pytest -q                                # 64 tests
```

## How it works
- `qsci/core` — calibration → error-weighted GHZ tree layout → circuits → fidelity
  F = (P₀…₀ + P₁…₁)/2 + C/2 from populations + 2N+2 parity-oscillation circuits, bootstrap CIs,
  readout mitigation; one SamplerV2 job per hardware experiment. Fast Pauli-frame noisy simulator.
- `qsci/agent` — Gemini function-calling loop; hardware is gated on a certified simulation and a budget check;
  certification is recomputed from data so the model can't over-claim.
- `ui/` — mission-control replay of the whole night. `qsci/paper` — paper generated from the run log.

See `CONTRACTS.md` for data formats.

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
| **24 · error-aware tree** | 0.866 | **0.723** [0.700, 0.746] | ✅ certified |
| **24 · shallow BFS tree** (round 2) | 0.845 | **0.842** [0.720, 0.972] | ✅ certified: **confirmed record** |
| 24 · textbook chain | 0.705 | 0.107 [0.031, 0.163] | ❌ |
| 28 · error-aware tree (round 2) | 0.815 | 0.580 [0.444, 0.703] | ❌ |
| 32 · error-aware tree | 0.796 | 0.537 [0.400, 0.641] | ❌ |
| 36 · error-aware tree (round 2) | 0.776 | 0.489 [0.327, 0.609] | ❌ |
| 40 · error-aware tree | 0.727 | 0.654 [0.520, 0.788] | ✅ certified, then **withdrawn** ↓ |
| 40 · same qubits, 2× shots (round 2 confirmation) | 0.726 | 0.414 [0.326, 0.486] | ❌ confirmation failed |
| 48 / 56 / 60 · error-aware tree | 0.648 / 0.604 / 0.642 | 0.455 / 0.319 / 0.244 | ❌ |

**Findings:**
- **Confirmed record: 24 qubits** of genuine multipartite entanglement, certified in two independent experiments with
  different layouts (F = 0.723 and 0.842).
- **The AI caught its own false positive.** Round 1 certified 40 qubits by a hair (lower bound 0.520). The agent's
  pre-registered confirmation re-ran it on the same qubits with twice the shots and got F = 0.414, so the paper, the
  site and the agent's log withdraw it. Best-of-several selection biases records upward; this run shows it happening.
- **Layout is the difference between entangled and not.** At 24 qubits the agent's error-aware tree reached F = 0.723,
  while the textbook linear CNOT chain on the same chip collapsed to 0.107.
- **The digital twin is honest about its limits:** it predicted all 12 experiments would certify; 4 did. Its
  overestimate grows with N, pointing to correlated noise that per-gate calibration data doesn't capture.

12 hardware experiments in two rounds, 322 s of QPU used out of 600 s free.

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

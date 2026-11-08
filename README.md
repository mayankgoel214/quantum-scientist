# Q-Scientist — an autonomous AI physicist on IBM quantum hardware

**Live:** https://q-scientist.vercel.app (replay of the real ibm_fez run) · **Paper:** [runs/20261107-220808-qpu/paper/paper.pdf](runs/20261107-220808-qpu/paper/paper.pdf)

Publish a static snapshot: `uv run python -m scripts.export_static && cd site && vercel deploy --prod`


An AI agent (Gemini) runs GHZ-entanglement experiments on its own — noisy simulator first,
then a real IBM Heron QPU — trying to **certify genuine multipartite entanglement
(fidelity lower confidence bound > 0.5) on as many qubits as possible**. It reasons out loud,
respects a hard QPU/LLM budget, and writes its own paper at the end.

## Results on real hardware (IBM `ibm_fez`, 156-qubit Heron)

| GHZ size | Twin prediction (frozen before the run) | Hardware fidelity [95% CI] | Genuinely entangled? |
|---|---|---|---|
| 16 qubits | 0.906 | **0.841** [0.820, 0.862] | ✅ certified |
| **24 qubits** | 0.866 | **0.723** [0.700, 0.746] | ✅ certified (record) |

Gemini designed a batch of 8 hardware experiments (16 to 60 qubits). The public queue held about 20,000 jobs, so 2 returned
before the deadline; the other 6 are still queued, and their pre-registered predictions are in `runs/20261107-220808-qpu/pending.json`.
**Finding:** the digital twin predicted certification correctly 2/2, but its fidelity error doubled with size (0.065 → 0.143),
which points to correlated noise that per-gate calibration data doesn't capture. 31 s of QPU used, out of 600 s free.

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

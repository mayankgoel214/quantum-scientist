# Q-Scientist: an AI that does quantum experiments on real IBM hardware and writes the paper

> Devpost write-up for Qiskit Fall Fest 2026, advanced track. All numbers come from run `runs/20261107-220808-qpu`.

## One-liner
An autonomous agent (Gemini) that designs GHZ-entanglement experiments, predicts every result with a digital twin
*before* it runs, executes them on IBM's 156-qubit `ibm_fez` via Qiskit Runtime, certifies genuine multipartite
entanglement with proper statistics, and writes its own arXiv-style paper. Every number in the paper comes from the
run log.

## Inspiration
"How many qubits can this chip really entangle?" is a simple question with a hard answer. You have to choose a
layout on a 156-qubit heavy-hex lattice, shape the circuit, decide on error suppression, and then *prove*
entanglement rather than assume it. That loop of hypothesis → experiment → analysis → next hypothesis is exactly what
a scientist does. We wanted to see whether an AI could run that loop on real quantum hardware, honestly. That means
it records its predictions before the data exists and confirms its own record before claiming it.

## What it does
1. **Reads the live chip.** Pulls `ibm_fez` calibration (T1/T2, readout and CZ error per qubit and coupler) and draws
   the chip map.
2. **Plans like a scientist.** Gemini designs a batch of GHZ experiments: sizes, layout strategy (an error-weighted
   shallow tree vs. the textbook linear CNOT chain), and shots. It writes a hypothesis for each one.
3. **Pre-registers predictions.** Before anything is submitted, every experiment runs through a *digital twin*: our
   own fast Pauli-frame noise simulator driven by the live calibration. The predicted fidelity is frozen to disk.
4. **Runs on real hardware.** Each experiment is one SamplerV2 job (2N+3 circuits plus readout calibration), with
   dynamical decoupling and measurement twirling, and the planned physical layout pinned.
5. **Certifies entanglement properly.** Fidelity is computed from populations plus parity oscillations (the
   Fourier amplitude at frequency N), with per-qubit readout mitigation and a bootstrap 95% CI. A state counts as
   genuinely multipartite entangled only if the **lower** CI bound is above 0.5.
6. **Closes the loop.** In round 2 the agent reads its hardware results and its own prediction errors, then designs a
   follow-up that must include a **confirmation re-run of its record**. A best-of-several result is biased upward,
   so the record has to survive an independent repeat.
7. **Writes the paper.** A 6-page PDF with abstract, methods, figures, a results table, limitations (including its own
   selection bias) and the IBM job IDs for reproducibility. A mission-control web UI replays the night.

## Results (real hardware, ibm_fez)
| | Result |
|---|---|
| Largest certified GHZ state | **24 qubits**, F = 0.723 (95% CI [0.700, 0.746]); threshold is 0.5 |
| Also certified | 16 qubits, F = 0.841 (95% CI [0.820, 0.862]) |
| Error-aware tree vs textbook chain (N = 24) | tree F = 0.723 on hardware; chain job still in IBM's public queue (twin predicts 0.705) |
| Digital twin vs hardware | predicted certification correctly 2/2; overestimated F by 0.065 (N=16) and 0.143 (N=24) |
| QPU time used | 31 s of the free 600 s (billed via `job.usage()`) |
| LLM cost | 1 Gemini 2.5 Flash planning call (well under $0.01) |

Gemini submitted a batch of 8 hardware experiments (N = 16 to 60). `ibm_fez` had ~20,000 jobs queued, so by the
deadline 2 had returned; the other 6 are still queued and their *pre-registered* twin predictions are published in
the run log, so they can be checked against hardware the moment they finish. We report only what actually ran.

## How we built it (Qiskit)
- **qiskit 2.x**: GHZ trees built from a spanning tree of the coupling map, then transpiled with a fixed
  `initial_layout` so the agent's layout is what actually runs. Parity circuits are produced by parameter binding.
- **qiskit-ibm-runtime**: `QiskitRuntimeService` (ibm_quantum_platform channel), `SamplerV2` in job mode,
  `dynamical_decoupling` (XpXm) and `twirling.enable_measure`, plus `job.usage()` for exact budget accounting.
- **Fake backends** (`FakeFez`) for the offline simulator path. **qiskit-aer** MPS is kept as a reference engine.
- A custom **Pauli-frame simulator** (numpy) for the digital twin. It handles 80 qubits in about 7 s, where Aer
  MPS took about 7 minutes at N=30. It models per-gate depolarizing error, per-layer T1/T2 idle decay and readout
  error from the live calibration.
- **Gemini 2.5 Flash** over REST with function calling for the interactive agent and JSON mode for batch planning.
  It runs under hard budget caps for both LLM calls and QPU seconds, which are charged *before* spending.
- Paper: matplotlib figures plus HTML, rendered to PDF with headless Chrome. UI: vanilla JS/SVG replaying the event log.

## Challenges
- **The queue.** `ibm_fez` had about 20,000 pending jobs. A one-experiment-at-a-time agent would have finished one
  or two experiments in a night. We rebuilt the hardware path to plan a batch, submit everything at once and collect
  results asynchronously, then added a second round so the agent still reacts to real data.
- **Simulation speed.** Aer MPS took about 7 minutes per 30-qubit experiment, too slow for an agent loop. Our
  Pauli-frame simulator is about 100× faster.
- **Honest statistics.** A naive fidelity estimate let *non*-entangled states appear just above 0.5 because of shot
  noise. We added bias correction, and a test that checks separable states are never certified.
- **LLM gotchas.** Gemini's "thinking" tokens silently truncated the JSON plan. We also caught the planner being
  over-optimistic: its first batch started at N = 40 because the simulator looked good. Guardrails now force a safe
  anchor and an informative comparison.

## What we learned
- The gap between a calibration-driven noise model and the real chip, measured out-of-sample: the twin was right
  about *whether* entanglement certifies, but its fidelity error doubled from 0.065 at N=16 to 0.143 at N=24. Error
  that grows with N means correlated noise (crosstalk, leakage, non-Markovian dephasing) the per-gate calibration
  numbers don't capture, and it means the twin's optimistic predictions for N ≥ 40 should not be trusted.
- Pre-registration matters: because predictions were frozen before the data, the twin's miss is a real
  measurement rather than something we could quietly tune away.

## What's next
- Error detection / post-selection to push certified N further.
- Let the agent re-plan around qubits that the twin systematically mispredicts. Those are where the noise model is wrong.

## How AI was used
Built with AI coding assistance (Claude) under the organizers' AI-allowed rule. At runtime, Gemini is the
experiment-planning agent. Every number in the paper is computed from the run log, never typed by a model.

## Links
- Live mission control (replay the night): https://q-scientist.vercel.app
- Repo: https://github.com/mayankgoel214/quantum-scientist
- Paper PDF: https://github.com/mayankgoel214/quantum-scientist/blob/main/runs/20261107-220808-qpu/paper/paper.pdf

# 3-minute demo script (Zoom judging, 10am)

**Before the call:** run `uv run python -m scripts.serve_ui`, open http://localhost:8765 and pick the qpu run.
Have the paper PDF open in a second tab. Share the browser tab, not the whole screen.

---

**0:00 Hook (15 s)**
"Last night an AI ran quantum experiments on IBM's 156-qubit `ibm_fez` while I slept. It decided what to run,
predicted every result *before* the hardware ran it, proved real entanglement, and wrote this paper. I'll show you
the night."

**0:15 The question (20 s)**. On screen: chip map.
"How many qubits can this chip *genuinely* entangle? A GHZ state on N qubits only counts if its fidelity's lower
95% confidence bound is above 0.5. That's the standard witness for genuine multipartite entanglement. The agent
has to choose which qubits, which circuit shape, and how to spend 10 free minutes of QPU time."

**0:35 Replay (60 s)**. Press Replay at 20×.
- Agent log: "Here's the AI agent's plan. It wrote a hypothesis for every experiment, including a head-to-head test: its
  error-aware tree layout vs the textbook linear CNOT chain."
- "Before submitting, every job went through my digital twin, a noise simulator driven by the live
  calibration. Those predictions were frozen to disk first. That's pre-registration."
- Chip map lights up: "This is the exact set of physical qubits it chose, avoiding the red, high-error couplers."
- Parity plot: "This oscillation at frequency N is the fingerprint of an N-qubit coherence. It's how I measure the
  entanglement, not just the populations."

**1:35 Results (40 s)**. Big number on the site, then paper Fig. 5.
- "All 12 hardware experiments came back. The record is 24 qubits, F = 0.723 with a tight interval, 0.700 to 0.746.
  The threshold is 0.5. A second 24-qubit run with a different tree also certified."
- "The best moment of the night: round 1 certified 40 qubits, by just 0.02. But before any results existed, the
  validator had already forced a re-run of the best result into round 2, because the best of several noisy runs is
  biased upward. Same qubits, twice the shots: 0.414. Failed. So 40 is not claimed anywhere. The confirmation step
  caught a false positive."
- "Circuit shape is the difference between entangled and not: at 24 qubits a shallow tree got 0.723, a linear CNOT
  chain on the same chip got 0.107. A tree that ignores error rates did just as well, so the win is circuit depth."
- "The digital twin overestimated every one of the 12 experiments, more at large sizes. That's correlated noise the
  calibration data doesn't capture, and I only know it because the predictions were frozen first."

**2:15 The paper (30 s)**. Scroll the PDF.
"Every number here is computed from the run log, never typed by the model. It includes its own limitations section,
the bootstrap CIs, and the IBM job IDs so anyone can check."

**2:45 Close (15 s)**
"Qiskit Runtime, SamplerV2 with DD and twirling, a custom Pauli-frame simulator, a few cents of AI API calls, and about 5 minutes of free QPU. An AI scientist that keeps itself honest. Thank you."

---

**Likely judge questions**
- *Why trust the fidelity?* Populations plus parity Fourier amplitude, readout mitigation, bootstrap CI, and the
  certification only uses the lower bound. A unit test checks that separable states are never certified.
- *Is the AI actually deciding?* Yes. The batch plan, hypotheses and round-2 follow-up are the AI agent's, logged verbatim.
  Validators only enforce safety rules (budget, a small anchor, the confirmation run).
- *Why not bigger N?* No error detection or post-selection yet. That's the next step, and the twin's errors show
  where the noise model is wrong.
- *Wasn't 40 a real result?* It cleared the bar once by 0.02 and failed an identical re-run (0.414). Could be a
  lucky draw or calibration drift; either way it's not reproducible, so it's not claimed.
- *Isn't readout mitigation inflating this?* It amplifies noise at large N (some mitigated parities exceed ±1), which
  is why I quote the 24-qubit run with the tightest interval. An unmitigated check is the next step.
- *Did the AI decide to withdraw 40?* The re-run was forced by the validator before any results existed; the rule
  that a failed re-run cancels the claim is in the code and applied to the paper, site and log.

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
- Agent log: "Here's Gemini's plan. It wrote a hypothesis for every experiment, including a head-to-head test: its
  error-aware tree layout vs the textbook linear CNOT chain."
- "Before submitting, every job went through our digital twin, a noise simulator driven by the live
  calibration. Those predictions were frozen to disk first. That's pre-registration."
- Chip map lights up: "This is the exact set of physical qubits it chose, avoiding the red, high-error couplers."
- Parity plot: "This oscillation at frequency N is the fingerprint of an N-qubit coherence. It's how we measure the
  entanglement, not just the populations."

**1:35 Results (40 s)**. Big number on the site, then paper Fig. 5.
- "All 12 hardware experiments came back overnight. Confirmed record: 24 qubits, certified twice with two different
  layouts, F = 0.723 and 0.842. The threshold is 0.5."
- "The best moment of the night: round 1 certified 40 qubits, just barely, lower bound 0.520. A best-of-several
  result is biased upward, so the agent is forced to re-run its record before claiming it. Same qubits, twice the
  shots: 0.414. Failed. So it withdrew the claim, in its log, on this site and in its paper. It caught its own
  false positive."
- "Layout is the difference between entangled and not: at 24 qubits the AI's error-aware tree got 0.723, the
  textbook CNOT chain on the same chip got 0.107."
- "The digital twin predicted all 12 would certify; 4 did. Its error grows with size, the signature of correlated
  noise calibration data doesn't capture, and we only know that because predictions were frozen first."

**2:15 The paper (30 s)**. Scroll the PDF.
"Every number here is computed from the run log, never typed by the model. It includes its own limitations section,
the bootstrap CIs, and the IBM job IDs so anyone can check."

**2:45 Close (15 s)**
"Qiskit Runtime, SamplerV2 with DD and twirling, a custom Pauli-frame simulator, a few cents of Gemini, and about 5 minutes of free QPU. An AI scientist that keeps itself honest. Thank you."

---

**Likely judge questions**
- *Why trust the fidelity?* Populations plus parity Fourier amplitude, readout mitigation, bootstrap CI, and the
  certification only uses the lower bound. A unit test checks that separable states are never certified.
- *Is the AI actually deciding?* Yes. The batch plan, hypotheses and round-2 follow-up are Gemini's, logged verbatim.
  Validators only enforce safety rules (budget, a small anchor, the confirmation run).
- *Why not bigger N?* No error detection or post-selection yet. That's the next step, and the twin's errors show
  where the noise model is wrong.
- *Wasn't 40 a real result?* It cleared the bar once by 0.02 and failed an identical re-run (0.414). Could be a
  lucky draw or calibration drift; either way it's not reproducible, so it's not claimed.

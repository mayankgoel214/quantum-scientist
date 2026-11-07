# Q-Scientist — shared contracts

An autonomous AI agent (Gemini) that runs GHZ-entanglement experiments on IBM quantum
hardware overnight, trying to certify genuine multipartite entanglement (fidelity > 0.5)
on as many qubits as possible, then writes its own paper.

Python 3.12, managed with `uv` (`uv run ...`). Deps already installed: qiskit 2.x,
qiskit-ibm-runtime 0.51, qiskit-aer 0.18, numpy, matplotlib, requests, python-dotenv, pytest.
No other dependencies without a very good reason. No Anthropic SDK. Gemini via plain REST (`requests`).

Secrets come from `.env` (see `.env.example`) via python-dotenv. Never print key values.

## Layout

```
qsci/core/      quantum core (calibration, layout, GHZ circuits, fidelity, execution)
qsci/budget.py  hard caps for QPU seconds + LLM calls (persisted)
qsci/events.py  event log writer (the ONLY way anything writes events)
qsci/agent/     Gemini agent loop
qsci/paper/     auto-written paper (HTML + PDF) from a run directory
ui/             mission-control web page (static; reads a run directory)
runs/<run_id>/  all outputs of one run
scripts/        CLI entry points
```

## Run directory: `runs/<run_id>/`

```
calibration.json   snapshot of backend calibration at run start (schema below)
events.jsonl       append-only event log (schema below)
experiments/<exp_id>.json   full ExperimentResult per experiment
budget.json        {"llm_calls": int, "qpu_seconds": float, "max_llm_calls": int, "max_qpu_seconds": float}
paper/paper.html, paper/paper.pdf, paper/fig_*.png
```

`run_id` format: `YYYYMMDD-HHMMSS-<mode>` e.g. `20261107-201500-sim`.

### calibration.json

```json
{
  "backend": "ibm_fez", "num_qubits": 156, "source": "live" | "fake",
  "timestamp": "ISO8601",
  "qubits": [{"id": 0, "t1_us": 120.3, "t2_us": 80.1, "readout_error": 0.012, "sx_error": 0.0003, "x": 3.0, "y": 1.0}],
  "edges":  [{"a": 0, "b": 1, "error": 0.004, "gate": "cz"}]
}
```
`x`,`y` are 2D plotting coordinates for the chip map (core computes them from the coupling map,
e.g. via a deterministic heavy-hex layout or spring layout with fixed seed). Missing values → null.

### ExperimentSpec (what the agent asks for)

```json
{
  "n_qubits": 24,
  "layout_strategy": "weighted_tree" | "bfs_tree" | "best_chain",
  "root": null,                     // optional physical root qubit; null = auto
  "exclude_qubits": [],             // physical qubits to avoid
  "circuit_variant": "tree" | "linear",
  "dynamical_decoupling": true,
  "measure_twirling": true,
  "readout_mitigation": true,
  "shots": 2000,
  "mode": "sim" | "qpu",
  "hypothesis": "free text from the agent"
}
```

### ExperimentResult (`experiments/<exp_id>.json`, also summarized in the event)

```json
{
  "exp_id": "e007", "spec": { ...ExperimentSpec },
  "physical_qubits": [..], "tree_edges": [[parent, child], ..], "root": 12,
  "circuit_depth": 9, "two_qubit_gate_count": 23,
  "fidelity": 0.61, "ci_low": 0.57, "ci_high": 0.65,
  "populations": 0.72, "coherence": 0.50,
  "parity_curve": [{"phi": 0.0, "parity": 0.48}, ...],
  "certified": true,                // ci_low > 0.5
  "qpu_seconds": 3.2,               // 0 for sim
  "wall_seconds": 41.0,
  "backend": "ibm_fez", "job_ids": ["..."],
  "error": null                     // string if failed; other fields may be null
}
```

Fidelity method (standard GHZ witness):
F = (P(0…0) + P(1…1))/2 + C/2.
Populations circuit: GHZ then measure all in Z.
Coherence: for j = 0..2N+1, φ_j = jπ/(N+1): apply Rz(φ_j) on every GHZ qubit, then H (X-basis), measure all;
parity ⟨P_j⟩ = Σ (−1)^{popcount} p(bitstring); ideally ⟨P_j⟩ = C·cos(Nφ_j + δ).
C = 2·|Σ_j e^{−iNφ_j} ⟨P_j⟩| / (2N+2)   (Fourier amplitude at frequency N — this is what qsci/core implements).
Confidence interval: bootstrap over shots (≥200 resamples) or analytic binomial propagation.

## Event log (`events.jsonl`) — one JSON object per line

Common fields: `{"ts": "ISO8601", "seq": int, "type": str, "data": {...}}`

| type | data |
|---|---|
| `run_started` | `{run_id, mode, backend, goal, planner, max_llm_calls, max_qpu_seconds}` — planner e.g. "gemini:gemini-2.5-flash" or "fake" |
| `thought` | `{text}` — agent's reasoning, 1–4 sentences, shown in UI feed |
| `experiment_started` | `{exp_id, spec}` |
| `experiment_result` | `{exp_id, n_qubits, mode, fidelity, ci_low, ci_high, certified, physical_qubits, tree_edges, root, qpu_seconds, error}` |
| `budget` | `{llm_calls, max_llm_calls, qpu_seconds, max_qpu_seconds}` |
| `milestone` | `{text, n_qubits, fidelity}` — e.g. new certified record |
| `error` | `{text}` |
| `run_finished` | `{best_certified_n, best_exp_id, summary}` |

`qsci/events.py` API:
```python
class EventLog:
    def __init__(self, run_dir: Path): ...
    def emit(self, type: str, **data) -> dict   # appends + flushes, thread-safe, assigns seq/ts
def new_run_dir(mode: str, root: Path = Path("runs")) -> Path
def read_events(run_dir: Path) -> list[dict]
```

`qsci/budget.py` API:
```python
class BudgetExceeded(Exception): ...
class Budget:
    def __init__(self, run_dir: Path, max_llm_calls: int, max_qpu_seconds: float): ...
    def charge_llm(self) -> None          # raises BudgetExceeded BEFORE the call if at cap
    def can_spend_qpu(self, est_seconds: float) -> bool
    def charge_qpu(self, seconds: float) -> None
    def snapshot(self) -> dict
```

## Core public API (`qsci.core`)

```python
def get_backend(mode: str, name: str | None = None)   # "sim" -> FakeFez (156q Heron); "qpu" -> live IBM via QiskitRuntimeService
def get_calibration(backend) -> dict                  # calibration.json schema
def plan_layout(calibration: dict, n: int, strategy: str, root: int | None = None,
                exclude: list[int] = ()) -> dict      # {"root", "qubits", "tree_edges", "depth"}
def estimate_qpu_seconds(spec: dict) -> float         # conservative estimate used by the budget
def run_experiment(spec: dict, backend, calibration: dict, exp_id: str) -> dict   # ExperimentResult
```

Simulation must be fast enough for the agent loop: noisy Aer sim using the fake backend's noise
model, `method="matrix_product_state"` (GHZ has bond dimension 2), and should handle N up to ~40
in well under a minute per experiment on a laptop. Restrict the simulated circuit to only the active
qubits (map physical → compact indices, keep the noise of the chosen physical qubits) so it stays fast.
On QPU: one SamplerV2 job per experiment containing all 2N+3 circuits as PUBs; use `initial_layout`
with the planned physical qubits and optimization_level that keeps the layout.

## Scripts (entry points)

- `uv run python -m scripts.run_agent --mode sim|qpu [--max-llm-calls N] [--max-qpu-seconds S]`
- `uv run python -m scripts.run_sweep --mode sim|qpu --ns 8,16,24,32` — NO-AI fallback, same event log format
- `uv run python -m scripts.write_paper runs/<run_id>`
- UI: `uv run python -m scripts.serve_ui` → http://localhost:8765 lists runs and replays/live-tails one.

"""Assemble the paper: numbers -> templated prose -> HTML -> PDF (headless Chrome)."""
from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .data import Experiment, RunFacts, load_run
from .figures import make_all
from .llm import Polisher

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "chromium", "chromium-browser",
]

STRATEGY_TEXT = {
    "weighted_tree": "<i>weighted_tree</i> grows a spanning tree outward from a root qubit, at each step adding the "
                     "neighbouring qubit with the lowest combined cost of CZ error, readout error and dephasing (from the "
                     "calibration snapshot), with a penalty on tree depth, which steers the state away from poorly calibrated couplers",
    "bfs_tree": "<i>bfs_tree</i> takes a breadth-first tree around the root, minimising tree height and hence circuit depth "
                "without regard to error rates",
    "best_chain": "<i>best_chain</i> selects a low-error one-dimensional path through the coupling graph",
}

REFERENCES = [
    ("ghz89", "D. M. Greenberger, M. A. Horne and A. Zeilinger, “Going beyond Bell’s theorem,” in <i>Bell’s Theorem, Quantum Theory and Conceptions of the Universe</i>, M. Kafatos (ed.), Kluwer, pp. 69–72 (1989); arXiv:0712.0921."),
    ("ghz90", "D. M. Greenberger, M. A. Horne, A. Shimony and A. Zeilinger, “Bell’s theorem without inequalities,” <i>Am. J. Phys.</i> <b>58</b>, 1131 (1990)."),
    ("guhne09", "O. Gühne and G. Tóth, “Entanglement detection,” <i>Phys. Rep.</i> <b>474</b>, 1–75 (2009)."),
    ("toth05", "G. Tóth and O. Gühne, “Detecting genuine multipartite entanglement with two local measurements,” <i>Phys. Rev. Lett.</i> <b>94</b>, 060501 (2005)."),
    ("monz11", "T. Monz <i>et al.</i>, “14-qubit entanglement: creation and coherence,” <i>Phys. Rev. Lett.</i> <b>106</b>, 130506 (2011)."),
    ("omran19", "A. Omran <i>et al.</i>, “Generation and manipulation of Schrödinger cat states in Rydberg atom arrays,” <i>Science</i> <b>365</b>, 570 (2019)."),
    ("song19", "C. Song <i>et al.</i>, “Generation of multicomponent atomic Schrödinger cat states of up to 20 qubits,” <i>Science</i> <b>365</b>, 574 (2019)."),
    ("mooney21", "G. J. Mooney, G. A. L. White, C. D. Hill and L. C. L. Hollenberg, “Generation and verification of 27-qubit Greenberger–Horne–Zeilinger states in a superconducting quantum computer,” <i>J. Phys. Commun.</i> <b>5</b>, 095004 (2021)."),
    ("kim23", "Y. Kim <i>et al.</i>, “Evidence for the utility of quantum computing before fault tolerance,” <i>Nature</i> <b>618</b>, 500 (2023)."),
    ("chamberland20", "C. Chamberland, G. Zhu, T. J. Yoder, J. B. Hertzberg and A. W. Cross, “Topological and subsystem codes on low-degree graphs with flag qubits,” <i>Phys. Rev. X</i> <b>10</b>, 011022 (2020)."),
    ("viola99", "L. Viola, E. Knill and S. Lloyd, “Dynamical decoupling of open quantum systems,” <i>Phys. Rev. Lett.</i> <b>82</b>, 2417 (1999)."),
    ("vdberg22", "E. van den Berg, Z. K. Minev and K. Temme, “Model-free readout-error mitigation for quantum expectation values,” <i>Phys. Rev. A</i> <b>105</b>, 032620 (2022)."),
    ("efron79", "B. Efron, “Bootstrap methods: another look at the jackknife,” <i>Ann. Statist.</i> <b>7</b>, 1–26 (1979)."),
    ("qiskit24", "A. Javadi-Abhari <i>et al.</i>, “Quantum computing with Qiskit,” arXiv:2405.08810 (2024)."),
]
REF_NUM = {key: i + 1 for i, (key, _) in enumerate(REFERENCES)}


def cite(*keys: str) -> str:
    return "[" + ", ".join(f'<a href="#ref-{k}">{REF_NUM[k]}</a>' for k in keys) + "]"


# ---------------------------------------------------------------- formatting helpers
def f3(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def f2(x) -> str:
    return "–" if x is None else f"{x:.2f}"


def pct(x) -> str:
    return f"{100 * x:.0f}%"


def esc(s) -> str:
    return html.escape(str(s), quote=False)


def fmt_dt(dt: datetime | None) -> str:
    if not dt:
        return "unknown"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def plain_to_html(text: str) -> str:
    paras = [p.strip() for p in text.replace("\r", "").split("\n\n") if p.strip()]
    out = []
    for p in paras:
        p = esc(" ".join(p.split()))
        # "Lead-in. rest" -> bold lead-in, if the draft used that convention
        if "§" in p:
            lead, rest = p.split("§", 1)
            p = f"<b>{lead.strip()}</b> {rest.strip()}"
        out.append(f"<p>{p}</p>")
    return "\n".join(out)


def strip_marks(text: str) -> str:
    return text.replace("§", "")


def _sentences(text: str, limit: int = 340) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for stop in (". ", "; ", ", "):
        i = cut.rfind(stop)
        if i > limit * 0.5:
            return cut[: i + 1].rstrip(",;") + " …"
    return cut.rstrip() + " …"


# ---------------------------------------------------------------- selections
def pick_thoughts(facts: RunFacts, k: int = 5) -> list[dict]:
    th = facts.thoughts
    if len(th) <= k:
        return th
    from .data import _parse_ts

    def t(ev):
        return _parse_ts(ev.get("ts"))

    chosen: list[int] = [0]
    anchors = []
    qpu = [e for e in facts.experiments if e.mode == "qpu" and e.started]
    if qpu:
        anchors.append(qpu[0].started)  # reasoning before first hardware job
    fails = [e for e in facts.experiments if e.mode == "qpu" and e.ok and not e.certified and e.finished]
    if fails:
        anchors.append(fails[0].finished)  # reaction to first hardware failure (thought right after)
    rec = facts.record("qpu")
    if rec and rec.finished:
        anchors.append(rec.finished)
    for i, a in enumerate(anchors):
        after = i >= 1  # for failures / records we want the reaction, i.e. the next thought after
        best = None
        for j, ev in enumerate(th):
            ts = t(ev)
            if ts is None:
                continue
            if after and ts >= a:
                best = j
                break
            if not after and ts <= a:
                best = j
        if best is not None and best not in chosen:
            chosen.append(best)
    if len(th) - 1 not in chosen:
        chosen.append(len(th) - 1)
    for j in range(len(th)):  # top up evenly if anchors collided
        if len(chosen) >= k:
            break
        if j not in chosen and j % max(1, len(th) // k) == 0:
            chosen.append(j)
    return [th[j] for j in sorted(chosen)[:k]]


# ---------------------------------------------------------------- the paper
class Paper:
    def __init__(self, facts: RunFacts, figs: dict, operator: str, polisher: Polisher):
        self.f = facts
        self.figs = figs
        self.operator = operator
        self.pol = polisher
        self.rec_hw = facts.record("qpu")
        self.rec_sim = facts.record("sim")
        self.cal = facts.calib_stats()
        self.mit = facts.mitigation_summary()
        self.n_exp = len(facts.experiments)
        self.n_sim = len(facts.by_mode("sim"))
        self.n_qpu = len(facts.by_mode("qpu"))
        self.n_err = sum(1 for e in facts.experiments if e.error)
        self.gap = facts.sim_hw_gap()

    # -------------- key numbers (also used by the tests and the LLM prompt)
    def key_numbers(self) -> dict:
        d = {
            "run_id": self.f.run_id, "backend": self.f.backend, "experiments": self.n_exp,
            "sim_experiments": self.n_sim, "hardware_experiments": self.n_qpu, "failed_jobs": self.n_err,
            "qpu_seconds_used": round(self.f.qpu_seconds_used, 1), "qpu_seconds_max": self.f.max_qpu_seconds,
            "llm_calls": self.f.llm_calls, "llm_calls_max": self.f.max_llm_calls,
            "duration_hours": round(self.f.duration_hours, 1) if self.f.duration_hours else None,
        }
        if self.f.batch:
            d["layout_comparison"] = [
                {"N": lp["n"], "tree_F": round(lp["tree"].fidelity, 3), "chain_F": round(lp["chain"].fidelity, 3),
                 "tree_certified": lp["tree"].certified, "chain_certified": lp["chain"].certified}
                for lp in self.f.layout_pairs()]
            tp = self.f.twin_pairs()
            if tp:
                d["twin_overestimate_mean"] = round(sum(tw.fidelity - hw.fidelity for hw, tw in tp) / len(tp), 3)
        for tag, r in (("hardware_record", self.rec_hw), ("sim_record", self.rec_sim)):
            if r:
                d[tag] = {"exp_id": r.exp_id, "N": r.n, "fidelity": round(r.fidelity, 3),
                          "ci_low": round(r.ci_low, 3), "ci_high": round(r.ci_high, 3)}
        return d

    def title(self) -> str:
        r = self.rec_hw
        if r:
            return (f"Certified genuine multipartite entanglement of a {r.n}-qubit GHZ state on {esc(self.f.backend)}, "
                    f"found by an autonomous AI scientist")
        if self.f.batch:
            return (f"No certified GHZ state on {esc(self.f.backend)}: an autonomous AI scientist's pre-registered "
                    "hardware batch and digital-twin predictions")
        if self.rec_sim:
            return (f"An autonomous AI scientist searches for large certified GHZ states: "
                    f"{self.rec_sim.n} qubits in noisy simulation")
        return "An autonomous AI scientist searches for large certified GHZ states: a negative result"

    def same_n_certified(self, r):
        """Certified hardware experiments at the record size, and how many physical qubits they all share."""
        same = [e for e in self.f.by_mode("qpu") if e.ok and e.certified and e.n == r.n and not self.f.refuted(e)]
        sets = [set(e.raw.get("physical_qubits") or []) for e in same]
        shared = len(set.intersection(*sets)) if len(sets) > 1 and all(sets) else None
        return same, shared

    def planner_phrase(self) -> str:
        planner = str(self.f.run_started.get("planner") or "gemini")
        auto = sum(1 for th in self.f.thoughts if str((th.get("data") or {}).get("text", "")).startswith("[autopilot]"))
        if planner.startswith("gemini"):
            model = planner.split(":", 1)[1] if ":" in planner else "Gemini"
            extra = f" (a scripted fallback made {auto} of its decisions)" if auto else ""
            return f"A language-model agent ({model}){extra}"
        return "A scripted baseline policy (no language model; dry run of the agent loop)"

    # -------------- abstract (plain text; may be LLM-polished)
    def abstract_draft(self) -> str:
        f, r, s = self.f, self.rec_hw, self.rec_sim
        h = f.duration_hours
        dur = "" if not h else (f" over {h:.1f} hours" if h >= 1 else f" over {max(1, round(h * 60))} minute{'s' if round(h * 60) > 1 else ''}")
        if f.backend.startswith("fake") and not self.n_qpu:
            target = f"a noisy simulator of a {self.cal['n_qubits']}-qubit IBM Heron processor ({f.backend})"
        else:
            target = f"the {self.cal['n_qubits']}-qubit IBM processor {f.backend}"
        budget = (f", spending {f.qpu_seconds_used:.0f} s of a {f.max_qpu_seconds:.0f} s QPU budget"
                  if self.n_qpu and f.max_qpu_seconds else "")
        if f.batch:
            r1 = [p for p in self.f.pending if (p.get("round") or 1) == 1]
            r2 = [p for p in self.f.pending if (p.get("round") or 1) == 2]
            back = "" if self.n_qpu >= len(self.f.pending) else f" ({self.n_qpu} had returned from the queue when this paper was generated)"
            how = (f"{self.planner_phrase()} designed a batch of {len(r1) or self.n_qpu} hardware experiments in a single planning step "
                   f"(the public queue made adaptive scheduling impossible)"
                   + (f" and, after reading every result, a second round of {len(r2)} that included a pre-registered confirmation of its own record"
                      if r2 else "")
                   + f"{back}. Before submission every experiment was predicted by a "
                   f"digital twin, a noise simulation driven by the live calibration. "
                   + (f"The run used {f.qpu_seconds_used:.0f} s of a self-imposed {f.max_qpu_seconds:.0f} s QPU budget. " if f.max_qpu_seconds else ""))
        else:
            how = (f"{self.planner_phrase()} planned, executed and interpreted {self.n_exp} experiments "
                   f"({self.n_sim} in a noisy simulator, {self.n_qpu} on hardware){dur}{budget}. ")
        parts = [
            f"We report an autonomous search for large Greenberger–Horne–Zeilinger (GHZ) states on {target}. "
            + how
            + "Fidelity to the ideal GHZ state was estimated from population and parity-oscillation measurements, "
            "with bootstrap confidence intervals, and genuine multipartite entanglement was claimed only when the "
            "lower confidence bound exceeded 1/2."
        ]
        if r:
            parts.append(
                f"On hardware the agent certified an N = {r.n} GHZ state with F = {r.fidelity:.3f} "
                f"(CI {r.ci_low:.3f}–{r.ci_high:.3f})"
                + (f", and N = {s.n} in simulation (F = {s.fidelity:.3f})." if s and not f.batch else ".")
            )
        elif s:
            parts.append(f"No hardware experiment was certified; in simulation the largest certified state had N = {s.n} "
                         f"(F = {s.fidelity:.3f}, CI {s.ci_low:.3f}–{s.ci_high:.3f}).")
        else:
            parts.append("No experiment met the certification criterion.")
        if r:
            same, shared = self.same_n_certified(r)
            if len(same) > 1:
                parts.append(f"N = {r.n} certified in {len(same)} separate hardware experiments with different tree layouts"
                             + (f", on overlapping qubits ({shared} of {r.n} shared)." if shared else "."))
        for orig, conf in f.refutations():
            parts.append(f"An apparent N = {orig.n} record (F = {orig.fidelity:.3f}, CI {orig.ci_low:.3f}–{orig.ci_high:.3f}) failed its "
                         f"pre-registered confirmation on the same qubits (F = {conf.fidelity:.3f}, CI {conf.ci_low:.3f}–{conf.ci_high:.3f}, "
                         f"{conf.spec.get('shots')} shots), so it is withdrawn and not claimed.")
        for lp in self.f.layout_pairs():
            t, c = lp["tree"], lp["chain"]
            parts.append(f"At N = {lp['n']} a shallow tree circuit (two-qubit depth {t.raw.get('circuit_depth') or '–'}) reached F = {t.fidelity:.3f} "
                         f"versus {c.fidelity:.3f} for a linear CNOT chain (depth {c.raw.get('circuit_depth') or '–'}) on the same device"
                         f"{' (only the tree certified)' if t.certified and not c.certified else ''}.")
        if f.batch and f.twin_pairs():
            errs = [tw.fidelity - hw.fidelity for hw, tw in f.twin_pairs()]
            parts.append(f"The digital twin, whose predictions were frozen before submission, overestimated hardware fidelity in "
                         f"{sum(1 for x in errs if x > 0)} of {len(errs)} experiments (mean {sum(errs) / len(errs):+.3f}, range "
                         f"{min(errs):+.3f} to {max(errs):+.3f}), quantifying the noise it does not model.")
        elif self.gap:
            diffs = [sv - hv for _, sv, hv in self.gap]
            parts.append(
                f"Hardware fidelities trailed simulation by {min(diffs):.2f} to {max(diffs):.2f} across "
                f"{len(self.gap)} matched sizes, a gap the agent learned to budget for by gating hardware jobs on simulated headroom."
            )
        best = r or s
        strength = "statistical fragility" if best and best.ci_low - 0.5 < 0.03 else "statistical strength"
        parts.append(f"We discuss the {strength} of the record, the role of error mitigation, and the "
                     "limits of an agent that selects its own best result. This paper was generated automatically from the run log.")
        return " ".join(parts)

    # -------------- 1 introduction (HTML)
    def introduction(self) -> str:
        return f"""
<p>Greenberger–Horne–Zeilinger (GHZ) states {cite('ghz89', 'ghz90')},
<span class="m">|GHZ<sub>N</sub>⟩ = (|0⟩<sup>⊗N</sup> + |1⟩<sup>⊗N</sup>)/√2</span>, are the canonical maximally
entangled states of <i>N</i> qubits. Because their single coherence term
|0…0⟩⟨1…1| dephases <i>N</i> times faster than a single qubit, the largest GHZ state a device can prepare is a
compact, demanding benchmark of its two-qubit gates, readout and coherence all at once. GHZ states of increasing
size have marked progress across platforms, from trapped ions {cite('monz11')} to Rydberg atoms {cite('omran19')}
and superconducting circuits {cite('song19', 'mooney21')}, and today's processors with more than a hundred qubits {cite('kim23')} make
much larger states conceivable.</p>
<p>Preparing a state is not the same as proving it is entangled. A standard witness {cite('guhne09', 'toth05')}
uses the fidelity <i>F</i> = ⟨GHZ<sub>N</sub>|ρ|GHZ<sub>N</sub>⟩: any state with <i>F</i> &gt; 1/2 is
<em>genuinely multipartite entangled</em> (GME), i.e. it cannot be written as a mixture of states that are
separable across any bipartition. We therefore treat certification as a statistical claim:
the lower end of a confidence interval on <i>F</i> must exceed 1/2.</p>
<p>Scaling the certified <i>N</i> on a {self.cal['n_qubits']}-qubit heavy-hex processor {cite('chamberland20')} is a search
problem over layouts, circuit shapes and mitigation settings under a tight hardware budget. In this work the
search was run end-to-end by an autonomous agent built on a large language model (Gemini), which proposed every
experiment{', predicted each hardware result with a digital twin before it ran,' if self.f.batch else ', decided when a result in simulation justified spending hardware time,'} and wrote this report.
The human operator set the goal and budget and ran the submit and collect commands; which experiments to run was the agent's choice.</p>
"""

    # -------------- 2 methods (HTML)
    def methods(self) -> str:
        c, m, f = self.cal, self.mit, self.f
        src = "a live calibration query" if c["source"] == "live" else "a static calibration snapshot shipped with Qiskit (fake backend)" if c["source"] == "fake" else "a calibration snapshot"
        def num(x, fmt):
            return "n/a" if x is None else fmt.format(x)
        strategies = [STRATEGY_TEXT[s] for s in m["strategies"] if s in STRATEGY_TEXT]
        strat_txt = "; ".join(strategies) if strategies else "a layout heuristic"
        r = self.rec_hw or self.rec_sim
        depth_txt = ""
        if r and r.raw.get("circuit_depth"):
            depth_txt = (f" For the record experiment ({r.exp_id}, N = {r.n}) the tree has "
                         f"{r.raw.get('two_qubit_gate_count', r.n - 1)} two-qubit gates arranged in {r.raw['circuit_depth']} layers of "
                         f"two-qubit-gate depth" + (f" (total transpiled depth {r.raw['transpiled_depth']})" if r.raw.get('transpiled_depth') else "") + ".")
        shots = " or ".join(str(s) for s in m["shots"]) or "n/a"
        mit = []
        if m["rem"]:
            mit.append(f"readout-error mitigation in {pct(m['rem'])} of experiments, inverting per-qubit (tensored) confusion "
                       "matrices estimated from all-|0⟩ and all-|1⟩ calibration circuits; because both the GHZ populations and the "
                       "parity are product-form functionals of the bitstring, this inversion is applied exactly, shot by shot")
        if m["hw_dd"] is not None:
            hw = []
            if m["hw_dd"]:
                hw.append(f"XpXm dynamical decoupling on idle qubits {cite('viola99')} ({pct(m['hw_dd'])})")
            if m["hw_twirl"]:
                hw.append(f"measurement twirling {cite('vdberg22')} ({pct(m['hw_twirl'])})")
            if hw:
                mit.append("on hardware experiments, " + " and ".join(hw) + " via the Qiskit Runtime sampler; these are not modelled in simulation")
        mit_txt = "; ".join(mit) if mit else "none"
        return f"""
<h3>2.1 Hardware and calibration</h3>
<p>Experiments targeted <b>{esc(f.backend)}</b>, a {c['n_qubits']}-qubit IBM Heron processor with
{c['n_edges']} tunable couplers arranged in a heavy-hex lattice and native {esc(c['gate'].upper())} entangling gates.
At the start of the run the agent recorded {src} (timestamp {esc(c['timestamp'] or 'unknown')}).
Median values across the chip were
T<sub>1</sub> = {num(c['median_t1'], '{:.0f}')} µs, T<sub>2</sub> = {num(c['median_t2'], '{:.0f}')} µs,
readout error {num(c['median_ro'] and 100 * c['median_ro'], '{:.2f}')}%,
single-qubit (√X) error {num(c['median_sx'] and 1e4 * c['median_sx'], '{:.1f}')}×10<sup>−4</sup> and
two-qubit error {num(c['median_2q'] and 100 * c['median_2q'], '{:.2f}')}%. The snapshot (Fig. 3) drove layout
selection and the noise model of the simulator: a Pauli-frame simulation (depolarizing gate errors, per-layer T<sub>1</sub>/T<sub>2</sub> idle
decoherence from real gate durations, and readout error) that keeps the per-qubit and per-coupler noise of
the physical qubits actually used. Hardware runs, where present, need no such approximation.</p>

<h3>2.2 Layouts and GHZ circuits</h3>
<p>A GHZ state on <i>N</i> physical qubits is prepared by a Hadamard on a root qubit followed by CNOTs
(compiled to CZ) fanning out along a spanning tree of the coupling graph, so the entangling depth scales with
the tree height rather than with <i>N</i>. The agent chose among layout strategies: {strat_txt}.
It could also exclude specific qubits and select a linear (chain) circuit instead of a tree.{depth_txt}</p>

<h3>2.3 Fidelity estimation</h3>
<p>Following Refs. {cite('omran19', 'mooney21')}, the fidelity splits into a population and a coherence term,</p>
<div class="eq"><span><i>F</i> = ½ (<i>P</i> + <i>C</i>),&ensp; <i>P</i> = <i>p</i>(0…0) + <i>p</i>(1…1),</span><span class="eqn">(1)</span></div>
<p>where <i>P</i> is read off a computational-basis measurement. The coherence <i>C</i> = 2|ρ<sub>0…0,1…1</sub>| is
obtained from parity oscillations: after preparing the state, every qubit receives a phase rotation
R<sub>z</sub>(φ) and is measured in the <i>X</i> basis, and the parity</p>
<div class="eq"><span>⟨𝒫(φ)⟩ = Σ<sub><i>x</i></sub> (−1)<sup>|<i>x</i>|</sup> <i>p</i><sub>φ</sub>(<i>x</i>) = <i>C</i> cos(<i>N</i>φ + δ)</span><span class="eqn">(2)</span></div>
<p>is sampled at the 2<i>N</i> + 2 phases φ<sub><i>j</i></sub> = <i>j</i>π/(<i>N</i> + 1), <i>j</i> = 0,…,2<i>N</i>+1.
Only a GHZ-type coherence oscillates at frequency <i>N</i>, so <i>C</i> is its Fourier amplitude,</p>
<div class="eq"><span><i>C</i> = <span class="frac"><span>1</span><span><i>N</i>+1</span></span>
 | Σ<sub><i>j</i></sub> e<sup><i>iN</i>φ<sub><i>j</i></sub></sup> ⟨𝒫(φ<sub><i>j</i></sub>)⟩ |.</span><span class="eqn">(3)</span></div>
<p>Each experiment therefore runs 2<i>N</i> + 3 circuits ({shots} shots each), plus two readout-calibration circuits when
mitigation is on. A 95% confidence interval on <i>F</i> comes from a percentile bootstrap that resamples the shots of
every circuit independently {cite('efron79')}, and an experiment is <em>certified</em> when
<span class="m"><i>F</i><sub>low</sub> &gt; 1/2</span>, which by the witness of Refs. {cite('guhne09', 'toth05')}
implies genuine multipartite entanglement of the prepared state. Error suppression and mitigation: {mit_txt}.
Circuits were built and transpiled with Qiskit {cite('qiskit24')}, keeping the planned physical layout fixed.</p>
"""

    # -------------- 3 agent (HTML)
    def agent(self) -> str:
        f = self.f
        gated, total = f.gated_fraction()
        quotes = pick_thoughts(f)
        from .data import _parse_ts
        t0 = f.started_at
        qhtml = []
        for ev in quotes:
            ts = _parse_ts(ev.get("ts"))
            stamp = f"t + {(ts - t0).total_seconds() / 60:.0f} min" if ts and t0 else ""
            qhtml.append(f'<blockquote><p>“{esc(_sentences(ev["data"]["text"]))}”</p><footer>— agent, {stamp}</footer></blockquote>')
        llm = f.llm_calls
        llm_txt = (f"It used {llm} of its {f.max_llm_calls} permitted model calls" if llm is not None and f.max_llm_calls
                   else "Model calls were metered")
        qpu_txt = (f" and {f.qpu_seconds_used:.1f} of {f.max_qpu_seconds:.0f} QPU seconds ({100 * f.qpu_seconds_used / f.max_qpu_seconds:.0f}%)"
                   if f.max_qpu_seconds else "")
        gate_txt = (f" In practice {gated} of {total} hardware experiments were preceded by a simulation at the same or larger <i>N</i>."
                    if total else "")
        excl = sorted({q for e in f.experiments for q in (e.spec.get("exclude_qubits") or [])})
        excl_txt = (f" Over the run the agent chose to exclude {len(excl)} physical qubit{'s' if len(excl) != 1 else ''} "
                    f"({', '.join('q' + str(q) for q in excl)}) after inspecting calibration data and failed runs." if excl else "")
        if f.batch:
            plan_q = [ev for ev in f.thoughts if str(ev["data"]["text"]).startswith("[batch plan]")]
            rat = esc(_sentences(plan_q[0]["data"]["text"].replace("[batch plan]", "").strip(), 900)) if plan_q else ""
            def _sp(rows):
                return "; ".join(f"{esc(e['exp_id'])}: N = {e['spec']['n_qubits']} {esc(e['spec']['layout_strategy'])}"
                                 + (f" (re-run of {esc(e['spec']['confirmation_of'])})" if e['spec'].get('confirmation_of') else "")
                                 for e in rows)
            r1 = [e for e in f.pending if (e.get("round") or 1) == 1]
            r2 = [e for e in f.pending if (e.get("round") or 1) == 2]
            specs = (f"round 1, {_sp(r1)}" + (f". Round 2, designed after reading every round-1 result, with the validator "
                     f"requiring a pre-registered re-run of the largest certified result: {_sp(r2)}" if r2 else ""))
            return f"""
<p>In its interactive mode the agent runs a closed loop over simulator and hardware experiments. On the public
queue of {esc(f.backend)}, however, a single job can wait hours, so a loop that waits for each result could finish only
one or two hardware experiments in a night. For this run the agent therefore designed each round of hardware experiments in a single
planning step. Its context was the live calibration summary, its own earlier simulator results and a cost table for
each experiment size. Every proposed experiment was then checked by a validator, which enforces a safe small-<i>N</i>
anchor, a matched tree-versus-chain comparison and the QPU budget. Each experiment was also run through a digital
twin (the noise simulator of §2.1 driven by the <em>live</em> calibration) to record a prediction <em>before</em> the
hardware result existed. All jobs were then submitted at once and collected as they completed.</p>
<p>The experiments were: {specs}. Every model call was charged against a hard cap before it was made; the run used {f.qpu_seconds_used:.0f} of its {f.max_qpu_seconds or 0:.0f} self-imposed QPU seconds (the open plan allows 600). The agent's round-1 rationale, verbatim:</p>
<blockquote><p>“{rat}”</p><footer>— agent, batch plan</footer></blockquote>
"""
        return f"""
<p>The agent runs a closed loop: it reads a compact summary of the calibration snapshot, all previous
results and the remaining budget; reasons in natural language (logged as <i>thought</i> events); and emits a
structured experiment specification (number of qubits, layout strategy, root, excluded qubits, circuit variant,
mitigation switches, shots, simulator or hardware, and a free-text hypothesis). The specification is
validated, checked against a hard budget that is charged <em>before</em> any spending, executed, and the
result is appended to the log that the agent reads on its next turn.</p>
<p>Over this run the agent logged {len(f.thoughts)} reasoning steps and issued {self.n_exp} experiment requests
({self.n_sim} simulated, {self.n_qpu} on hardware). {llm_txt}{qpu_txt}. Because hardware time is scarce,
the agent was instructed to use the noisy simulator as a gate: a hardware run is justified only when simulation
at that size shows headroom above the threshold.{gate_txt}{excl_txt} A selection of its logged reasoning,
verbatim, follows.</p>
{''.join(qhtml)}
"""

    # -------------- 4 results (HTML)
    def results(self) -> str:
        f, r, s = self.f, self.rec_hw, self.rec_sim
        out = []
        if r:
            rc = f.record_calib(r)
            margin = r.ci_low - 0.5
            out.append(
                f"<p>The largest state certified on hardware is the <b>N = {r.n}</b> GHZ state of experiment {r.exp_id} "
                f"(Table 1), with <i>F</i> = {r.fidelity:.3f} and confidence interval [{r.ci_low:.3f}, {r.ci_high:.3f}]: "
                f"the lower bound clears the GME threshold by {margin:.3f}. Its populations and coherence are "
                f"<i>P</i> = {f3(r.raw.get('populations'))} and <i>C</i> = {f3(r.raw.get('coherence'))}; as is typical, the "
                f"coherence term limits the fidelity. The state used a {esc(r.spec.get('layout_strategy', 'tree'))} layout rooted at "
                f"q{r.raw.get('root')} (Fig. 3)"
                + (f" whose couplers have a mean CZ error of {100 * rc['mean_2q']:.2f}%" if rc['mean_2q'] is not None else "")
                + (f" and mean readout error {100 * rc['mean_ro']:.2f}%" if rc['mean_ro'] is not None else "")
                + (f", against chip medians of {100 * self.cal['median_2q']:.2f}% and {100 * self.cal['median_ro']:.2f}%."
                   if self.cal['median_2q'] and self.cal['median_ro'] else ".")
                + "</p>"
            )
        else:
            out.append("<p>No hardware experiment met the certification criterion "
                       f"({self.n_qpu} hardware experiment{'s' if self.n_qpu != 1 else ''} attempted).</p>")
        if r and r.mode == "qpu":
            def maxp(e):
                return max((abs(p.get("parity", 0)) for p in (e.raw.get("parity_curve") or [])), default=0)
            same, _ = self.same_n_certified(r)
            noisy = [e for e in same if e is not r and maxp(e) > 1]
            allmax = max((maxp(e) for e in f.by_mode("qpu") if e.ok), default=0)
            if noisy or allmax > 1:
                out.append(
                    "<p><b>Readout mitigation and noise.</b> Inverting the readout model amplifies shot noise as <i>N</i> grows: "
                    f"across the run, individual mitigated parities reach |⟨P⟩| = {allmax:.2f}, outside the physical range ±1."
                    + "".join(f" The second N = {r.n} certification, {e.exp_id} (<i>F</i> = {e.fidelity:.3f} [{e.ci_low:.3f}, {e.ci_high:.3f}]), "
                              f"has mitigated parities up to {maxp(e):.2f} and an interval {(e.ci_high - e.ci_low) / (r.ci_high - r.ci_low):.0f}× wider than "
                              f"{r.exp_id}'s, so {r.exp_id} is quoted as the record." for e in noisy)
                    + " The certified values are properties of the mitigated estimator; an unmitigated check was not performed.</p>")
        for orig, conf in f.refutations():
            out.append(
                f"<p><b>A withdrawn record.</b> In the first round, {orig.exp_id} certified an N = {orig.n} state with "
                f"<i>F</i> = {orig.fidelity:.3f} [{orig.ci_low:.3f}, {orig.ci_high:.3f}], a lower bound only {orig.ci_low - 0.5:.3f} above the "
                f"threshold, while the smaller N = {', '.join(str(e.n) for e in f.by_mode('qpu') if e.ok and not e.certified and e.n < orig.n and e.spec.get('layout_strategy') == orig.spec.get('layout_strategy') and (e.raw.get('spec') or {}).get('confirmation_of') is None) or '–'} "
                f"tree states did not certify. Its validator-enforced confirmation, {conf.exp_id}, re-ran the identical specification on the same physical qubits "
                f"with {conf.spec.get('shots')} shots and measured <i>F</i> = {conf.fidelity:.3f} [{conf.ci_low:.3f}, {conf.ci_high:.3f}]: not certified. "
                f"N = {orig.n} is therefore not claimed; the first result is best read as an upward fluctuation selected from several noisy attempts, "
                f"possibly compounded by calibration drift between the two runs. This is the selection effect discussed in Section 5, observed directly.</p>")
        if s and not f.batch:
            out.append(f"<p>In the noisy simulator the largest certified state has <b>N = {s.n}</b> ({s.exp_id}, "
                       f"<i>F</i> = {s.fidelity:.3f}, CI [{s.ci_low:.3f}, {s.ci_high:.3f}]).</p>")
        pf = self.figs.get("parity_fit")
        pe = self.figs.get("parity_exp")
        if pf and pe:
            out.append(
                f"<p>Figure 2 shows the parity oscillation of {pe.exp_id}. For a GHZ-type coherence, folding the {len(pf['phi'])} "
                f"phase settings by <i>N</i>φ collapses them onto a single cosine, the signature of an <i>N</i>-body coherence. An independent "
                f"least-squares fit of Eq. (2) gives amplitude {pf['amp']:.3f} (residual RMS {pf['rms']:.3f}), "
                + (f"consistent with the estimator value <i>C</i> = {f3(pe.raw.get('coherence'))}.</p>"
                   if pe.raw.get('coherence') is None or abs(pf['amp'] - pe.raw['coherence']) <= max(0.03, 0.15 * pe.raw['coherence'])
                   else f"which does <b>not</b> agree with the estimator value <i>C</i> = {f3(pe.raw.get('coherence'))}; the stored "
                        "parity curve does not oscillate at frequency <i>N</i> as Eq. (2) requires, so this coherence value should be "
                        "treated as unverified.</p>")
            )
        fits = []
        for mode, label in (("qpu", "hardware"), ("sim", "simulation")):
            fit = f.decay_fit(mode)
            nstar = f.predicted_threshold_n(mode)
            if fit:
                fits.append(f"{label}: <i>N</i><sub>e</sub> ≈ {fit[1]:.0f}" + (f", crossing <i>F</i> = 1/2 near <i>N</i> ≈ {nstar:.0f}" if nstar else ""))
        if fits:
            out.append("<p>Fidelity decays roughly exponentially with size (Fig. 1). Fitting "
                       "<i>F</i>(<i>N</i>) = <i>F</i><sub>0</sub> e<sup>−<i>N</i>/<i>N</i><sub>e</sub></sup> to the central values gives "
                       + "; ".join(fits) + ". These fits are descriptive only; they ignore the different layouts used at each size.</p>")
        for lp in f.layout_pairs():
            t, c = lp["tree"], lp["chain"]
            pt, pc = lp["tree_twin"], lp["chain_twin"]
            verdict = ("Only the tree certified GME." if t.certified and not c.certified else
                       "Both certified." if t.certified and c.certified else
                       "The chain did better, contrary to the prediction." if c.fidelity > t.fidelity else
                       "Neither certified.")
            out.append(
                f"<p><b>Tree circuit versus linear chain.</b> At <i>N</i> = {lp['n']} the error-weighted tree ({t.exp_id}, "
                f"two-qubit depth {t.raw.get('circuit_depth') or '–'}) reached <i>F</i> = {t.fidelity:.3f} [{t.ci_low:.3f}, {t.ci_high:.3f}], "
                f"against {c.fidelity:.3f} [{c.ci_low:.3f}, {c.ci_high:.3f}] for the linear CNOT chain ({c.exp_id}, depth "
                f"{c.raw.get('circuit_depth') or '–'}), a difference of {t.fidelity - c.fidelity:+.3f}. {verdict}"
                + (f" The twin had predicted {pt.fidelity:.3f} and {pc.fidelity:.3f} respectively, so the sign of the "
                   f"difference was {'correctly' if (pt.fidelity - pc.fidelity) * (t.fidelity - c.fidelity) > 0 else '<b>not</b>'} "
                   "anticipated before the hardware ran." if pt and pc else "")
                + "".join(f" A breadth-first tree that ignores error rates ({b.exp_id}) reached <i>F</i> = {b.fidelity:.3f} at the same size, "
                          "so the advantage over the chain comes from circuit depth rather than from error-weighting the layout."
                          for b in f.by_mode("qpu") if b.ok and b.n == lp["n"] and b.spec.get("layout_strategy") == "bfs_tree"
                          and b.fidelity >= t.fidelity - 0.02)
                + " (Fig. 5)</p>")
        tp = f.twin_pairs()
        if tp:
            errs = [tw.fidelity - hw.fidelity for hw, tw in tp]
            agree = sum(1 for hw, tw in tp if hw.certified == tw.certified)
            out.append(
                f"<p><b>Prediction versus hardware.</b> Across {len(tp)} hardware experiments the digital twin overestimated fidelity by "
                f"{sum(errs) / len(errs):.3f} on average (range {min(errs):+.3f} to {max(errs):+.3f}) and predicted the certification "
                f"outcome correctly in {agree} of {len(tp)} cases. Because every prediction was recorded before submission, this "
                "is an honest out-of-sample test of the noise model.</p>")
            if "twin" in self.figs["paths"]:
                p5 = self.figs["paths"]["twin"]
                out.append(f'<figure class="col"><img src="{p5.name}" alt="Figure 5"><figcaption><b>Figure 5.</b> Hardware '
                           "fidelity (circles, bootstrap CI; filled = certified) against the digital-twin prediction recorded before "
                           "submission (squares, dashed), for the error-aware tree and the textbook linear chain.</figcaption></figure>")
        if self.gap and not f.batch:
            rows = ", ".join(f"{d[1] - d[2]:.2f} at N = {d[0]}" for d in self.gap)
            out.append(f"<p>Where both were measured, the best hardware fidelity trails the best simulated one by {rows}. "
                       "Figure 4 places every experiment on the run's timeline together with budget consumption.</p>")
        else:
            out.append("<p>Figure 4 places every experiment on the run's timeline together with budget consumption.</p>")
        return "\n".join(out)

    def table(self) -> str:
        rows = []
        for e in self.f.experiments:
            if e.error:
                status = '<span class="err">error</span>'
                fid = ci = P = C = "–"
            else:
                status = '<span class="ok">✓</span>' if e.certified else '<span class="no">✗</span>'
                fid = f3(e.fidelity)
                ci = f"[{f3(e.ci_low)}, {f3(e.ci_high)}]"
                P, C = f3(e.raw.get("populations")), f3(e.raw.get("coherence"))
            excl = e.spec.get("exclude_qubits") or []
            layout = esc(e.spec.get("layout_strategy", "–")) + ("" if e.spec.get("circuit_variant", "tree") == "tree" else f" / {esc(e.spec.get('circuit_variant'))}")
            if excl:
                layout += f" <span class='dim'>(−{len(excl)})</span>"
            hl = ' class="rec"' if (self.rec_hw and e.exp_id == self.rec_hw.exp_id) or (self.rec_sim and not self.f.batch and e.exp_id == self.rec_sim.exp_id) else ""
            qs = e.raw.get("qpu_seconds")
            rows.append(
                f"<tr{hl}><td>{esc(e.exp_id)}</td><td>{'HW' if e.mode == 'qpu' else 'twin' if e.raw.get('twin_of') else 'sim'}</td><td>{e.n}</td><td class='l'>{layout}</td>"
                f"<td>{e.raw.get('circuit_depth') or '–'}</td><td>{P}</td><td>{C}</td><td><b>{fid}</b></td><td>{ci}</td>"
                f"<td>{status}</td><td>{f2(qs) if e.mode == 'qpu' and qs else '–'}</td></tr>")
        return f"""
<figure class="wide table">
<figcaption><b>Table 1.</b> All experiments in chronological order. HW = {esc(self.f.backend)}; sim = noisy simulator; twin = digital-twin prediction recorded before submission.
Layout “(−k)” marks k qubits excluded by the agent. <i>P</i>, <i>C</i>, <i>F</i> as in Eqs. (1)–(3); CI is the
bootstrap confidence interval; ✓ = certified (<i>F</i><sub>low</sub> &gt; 1/2). Shaded rows are the records.</figcaption>
<table>
<thead><tr><th>ID</th><th>Mode</th><th><i>N</i></th><th class="l">Layout</th><th>2q depth</th><th><i>P</i></th><th><i>C</i></th><th><i>F</i></th><th>CI</th><th>GME</th><th>QPU s</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</figure>"""

    # -------------- 5 discussion (plain text; may be LLM-polished). "§" marks a bold lead-in.
    def discussion_draft(self) -> str:
        f, r = self.f, self.rec_hw or self.rec_sim
        paras = []
        if r:
            margin = r.ci_low - 0.5
            attempts = [e for e in f.by_mode(r.mode) if abs(e.n - r.n) <= 4]
            n_attempts = len([e for e in attempts if not e.error])
            fragile = "narrow" if margin < 0.03 else "moderate" if margin < 0.08 else "comfortable"
            paras.append(
                f"Statistical caveats.§ The record lower bound exceeds 1/2 by only {margin:.3f}, a {fragile} margin. "
                f"The interval reflects shot noise alone; it does not include drift between the {2 * r.n + 3} circuits of the "
                f"experiment or between experiments. More importantly, the agent chose which configurations to run and which "
                f"result to call a record: {n_attempts} {'hardware' if r.mode == 'qpu' else 'simulator'} experiments were run within four qubits "
                f"of the record size, and reporting the best of several noisy estimates biases the headline upward. "
                f"No correction for this selection was applied. "
                + (f"The danger is not hypothetical: in this run an apparent N = {f.refutations()[0][0].n} record failed its pre-registered "
                   f"confirmation and was withdrawn (Section 4). "
                   if f.refutations() else "")
                + (f"The N = {r.n} record itself was certified in {len(self.same_n_certified(r)[0])} separate experiments with different tree "
                   f"layouts{f' sharing {self.same_n_certified(r)[1]} of {r.n} qubits' if self.same_n_certified(r)[1] else ''}, which mitigates but "
                   f"does not remove this concern; neither was itself re-run."
                   if len([e for e in f.by_mode(r.mode) if e.ok and e.certified and e.n == r.n]) > 1 else
                   f"A pre-registered confirmation run of the identical specification, ideally with more shots, is the appropriate next "
                   f"step before treating N = {r.n} as established.")
            )
        if self.mit["rem"] or self.mit["twirl"]:
            paras.append(
                "Error mitigation.§ "
                f"Readout-error mitigation was enabled in {pct(self.mit['rem'])} of experiments. It reweights measured "
                "populations and parities through the inverse of a per-qubit confusion model, so the certified fidelity is a property "
                "of the mitigated estimator rather than of raw counts. The model assumes readout errors that are uncorrelated across "
                "qubits and free of crosstalk, and the error rates are estimated from just two calibration circuits whose own shot "
                "noise is not propagated into the bootstrap interval, so the interval is somewhat too narrow. A conservative reading of the result would require the "
                "unmitigated fidelity to also exceed 1/2, which we did not separately establish."
            )
        if self.gap:
            diffs = [sv - hv for _, sv, hv in self.gap]
            paras.append(
                f"Simulation versus hardware.§ The {'digital twin' if f.batch else 'simulator'} overestimated hardware fidelity at every matched size, by "
                f"{min(diffs):.2f} to {max(diffs):.2f}. The noise model is built from a single calibration snapshot and "
                "omits crosstalk, leakage, non-Markovian dephasing and temporal drift, so the gap is expected to grow with N "
                "and with circuit duration. "
                + ("Because the twin predictions were frozen before the jobs ran, the size of this gap is a direct, "
                   "unbiased measurement of how much of the device's error budget the model misses."
                   if f.batch else "Simulation was useful to the agent as an upper bound and a gate for hardware "
                   "spending, not as a predictor.")
            )
        dur = f.duration_hours
        cal_note = ("The calibration data came from a static snapshot shipped with Qiskit rather than a live query, so "
                    "layout choices may not reflect the device state at execution time. "
                    if self.cal["source"] == "fake" else "")
        paras.append(
            "Scope and other limitations.§ "
            + cal_note
            + (f"The run lasted {dur:.1f} hours on one calibration snapshot, during which device parameters can drift. " if dur else "")
            + (f"{self.n_err} job{'s' if self.n_err != 1 else ''} failed for operational reasons and carried no evidence. " if self.n_err else "")
            + "A fidelity above 1/2 certifies genuine multipartite entanglement but does not characterize the full state, and "
            "the witness depends on the assumption that the measurements are the intended ones. Finally, the agent's search "
            "was short and greedy; it explored a small part of the space of roots, layouts and settings, so larger certified "
            "states on this device are plausible."
        )
        return "\n\n".join(paras)

    # -------------- 6 reproducibility (HTML)
    def reproducibility(self) -> str:
        f = self.f
        vers = []
        for pkg in ("qiskit", "qiskit-ibm-runtime", "qiskit-aer", "numpy"):
            try:
                vers.append(f"{pkg} {metadata.version(pkg)}")
            except metadata.PackageNotFoundError:
                pass
        jobs = [(e.exp_id, j) for e in f.by_mode("qpu") for j in (e.raw.get("job_ids") or [])]
        model = f.run_started.get("model") or f.run_started.get("planner") or "Gemini"
        jobs_html = ("<p class='jobs'>" + "; ".join(f"{esc(x)}: <code>{esc(j)}</code>" for x, j in jobs) + "</p>") if jobs else "<p>No hardware jobs were submitted.</p>"
        return f"""
<table class="kv">
<tr><td>Run ID</td><td><code>{esc(f.run_id)}</code></td></tr>
<tr><td>Run mode / backend</td><td>{esc(f.run_mode)} / {esc(f.backend)}</td></tr>
<tr><td>Run start / end</td><td>{fmt_dt(f.started_at)} / {fmt_dt(f.finished_at)}</td></tr>
<tr><td>Calibration</td><td>{esc({'live': 'live backend query', 'fake': 'static Qiskit snapshot (fake backend)'}.get(self.cal['source'], self.cal['source'] or 'unknown'))}, {esc(self.cal['timestamp'] or 'unknown')}</td></tr>
<tr><td>Budget used</td><td>{f.qpu_seconds_used:.1f} / {f.max_qpu_seconds or 0:.0f} QPU s (self-imposed cap; open plan 600 s); LLM calls capped and metered</td></tr>
<tr><td>Agent model</td><td>{esc(model)}</td></tr>
<tr><td>Software</td><td>{esc(', '.join(vers))}</td></tr>
</table>
<p>Hardware job IDs:</p>
{jobs_html}
<p>Every number in this paper is computed from <code>runs/{esc(f.run_id)}/</code> (<code>calibration.json</code>,
<code>events.jsonl</code>, <code>experiments/*.json</code>) and can be regenerated with
<code>uv run python -m scripts.write_paper runs/{esc(f.run_id)}</code>.</p>
"""

    # -------------- assembly
    def render(self) -> str:
        figs = self.figs["paths"]
        facts_summary = json.dumps(self.key_numbers(), indent=1, default=str)
        abstract = self.pol.polish("Abstract", self.abstract_draft(), facts_summary)
        discussion_draft = self.discussion_draft()
        discussion = self.pol.polish("Discussion and limitations", strip_marks(discussion_draft), facts_summary)
        discussion_html = plain_to_html(discussion_draft if discussion == strip_marks(discussion_draft) else discussion)

        def fig(key: str, cls: str, num: int, caption: str) -> str:
            if key not in figs:
                return ""
            return (f'<figure class="{cls}"><img src="{figs[key].name}" alt="Figure {num}">'
                    f"<figcaption><b>Figure {num}.</b> {caption}</figcaption></figure>")

        r = self.rec_hw or (None if self.f.batch else self.rec_sim)
        pe, ce = self.figs.get("parity_exp"), self.figs.get("chip_exp")
        fig1 = fig("fidelity", "col", 1, "GHZ fidelity versus size for every successful experiment. Error bars: bootstrap "
                   "confidence intervals. Filled markers are certified (lower bound above 1/2); open markers are not. "
                   "Dotted lines: exponential fits <i>F</i><sub>0</sub>e<sup>−<i>N</i>/<i>N</i><sub>e</sub></sup>. "
                   "Shaded: region below the GME threshold.")
        fig2 = fig("parity", "wide", 2, f"Parity oscillation of {pe.exp_id if pe else ''} (<i>N</i> = {pe.n if pe else ''}, "
                   f"{'hardware' if pe and pe.mode == 'qpu' else 'simulator'}). (i) Measured parity at each phase "
                   "φ<sub><i>j</i></sub> with the fitted <i>C</i> cos(<i>N</i>φ + δ). (ii) The same points folded onto one period.") if pe else ""
        fig3 = fig("chip", "col", 3, f"Calibration snapshot of {esc(self.f.backend)}: couplers coloured by CZ error, "
                   f"with the GHZ tree of {ce.exp_id if ce else ''} highlighted (root ★" + ("; ✕ marks qubits excluded by the agent" if any(e.spec.get("exclude_qubits") for e in self.f.experiments) else "") + ").") if ce else ""
        fig4 = fig("timeline", "wide", 4, "Timeline of the run. Top: qubit number of each experiment against wall-clock time, "
                   "coloured by outcome; the line tracks the certified hardware record. Bottom: cumulative fraction of the QPU-time "
                   "and LLM-call budgets consumed.")
        today = datetime.now(timezone.utc).strftime("%B %-d, %Y")
        polish_note = ("Abstract and discussion were copy-edited by Gemini under a numbers-must-not-change check."
                       if self.pol.used else "All prose was generated deterministically from templates.")
        refs = "\n".join(f'<li id="ref-{k}">{t}</li>' for k, t in REFERENCES)
        rec_line = (f"Record: N = {r.n}, F = {r.fidelity:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]" if r else "")
        return TEMPLATE.format(
            title=self.title(),
            operator=esc(self.operator),
            date=today,
            run_id=esc(self.f.run_id),
            abstract=plain_to_html(abstract),
            intro=self.introduction(),
            methods=self.methods(),
            agent=self.agent(),
            results=self.results(),
            table=self.table(),
            discussion=discussion_html,
            repro=self.reproducibility(),
            refs=refs,
            fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4,
            css=CSS,
            polish_note=polish_note,
            rec_line=esc(rec_line),
        )


CSS = r"""
@page { size: Letter; margin: 0.7in 0.68in 0.75in 0.68in;
  @bottom-center { content: counter(page); font: 8pt 'STIX Two Text', 'Times New Roman', serif; color: #555; } }
:root { --ink: #1b1b1b; --dim: #5b5b5b; --rule: #c9c9c9; --accent: #1f4e8c; }
* { box-sizing: border-box; }
html { background: #ecebe8; }
body { font-family: 'STIX Two Text', 'Times New Roman', Times, serif; font-size: 9.7pt; line-height: 1.32;
  color: var(--ink); background: #fff; max-width: 8.5in; margin: 24px auto; padding: 0.7in 0.68in;
  box-shadow: 0 2px 18px rgba(0,0,0,.12); font-kerning: normal; font-variant-numeric: lining-nums; }
@media print { html { background: #fff; } body { margin: 0; padding: 0; box-shadow: none; max-width: none; } }
.preprint { display: flex; justify-content: space-between; font-size: 7.6pt; color: var(--dim);
  border-bottom: 0.6pt solid var(--rule); padding-bottom: 3pt; margin-bottom: 16pt; letter-spacing: .02em; }
h1 { font-size: 16.5pt; line-height: 1.2; text-align: center; margin: 0 0.25in 8pt; font-weight: 600; }
.authors { text-align: center; font-size: 10.5pt; margin-bottom: 2pt; }
.affil { text-align: center; font-size: 8.3pt; color: var(--dim); font-style: italic; margin-bottom: 2pt; }
.date { text-align: center; font-size: 8.3pt; color: var(--dim); margin-bottom: 12pt; }
.abstract { margin: 0 0.42in 12pt; font-size: 9pt; text-align: justify; hyphens: auto; }
.abstract p { margin: 0; text-indent: 0; }
.abstract h2 { font-size: 8.6pt; text-align: center; text-transform: uppercase; letter-spacing: .14em; margin: 0 0 3pt; }
.cols { column-count: 2; column-gap: 0.26in; text-align: justify; hyphens: auto; orphans: 3; widows: 3; }
h2 { font-size: 10pt; text-transform: uppercase; letter-spacing: .06em; font-weight: 700; margin: 11pt 0 4pt;
  break-after: avoid; text-align: left; }
h3 { font-size: 9.7pt; font-style: italic; font-weight: 600; margin: 7pt 0 2pt; break-after: avoid; text-align: left; }
p { margin: 0 0 4pt; text-indent: 1.1em; }
h2 + p, h3 + p, .eq + p, figure + p, blockquote + p { text-indent: 0; }
.eq { display: flex; justify-content: center; align-items: center; position: relative; margin: 5pt 0 6pt;
  font-size: 10pt; text-align: center; break-inside: avoid; }
.eqn { position: absolute; right: 0; color: var(--dim); font-size: 9pt; }
.frac { display: inline-flex; flex-direction: column; vertical-align: middle; text-align: center; margin: 0 2pt;
  font-size: 0.9em; line-height: 1.1; }
.frac > span:first-child { border-bottom: 0.6pt solid currentColor; padding: 0 2pt; }
.m { white-space: nowrap; }
sub, sup { font-size: 0.72em; line-height: 0; }
a { color: var(--accent); text-decoration: none; }
figure { margin: 6pt 0 8pt; break-inside: avoid; }
figure img { width: 100%; display: block; }
figure.wide { column-span: all; margin: 8pt 0 10pt; }
figcaption { font-size: 8.1pt; line-height: 1.28; margin-top: 3pt; text-align: justify; color: #262626; }
figure.table figcaption { margin: 0 0 4pt; }
blockquote { margin: 4pt 0 6pt; padding: 3pt 0 3pt 8pt; border-left: 1.6pt solid var(--accent);
  font-size: 8.8pt; break-inside: avoid; background: #f6f8fb; }
blockquote p { text-indent: 0; margin: 0; font-style: italic; }
blockquote footer { font-size: 7.6pt; color: var(--dim); text-align: right; margin-top: 1pt; }
table { border-collapse: collapse; width: 100%; font-size: 7.8pt; font-variant-numeric: tabular-nums; }
thead th { border-top: 0.9pt solid #000; border-bottom: 0.6pt solid #000; padding: 2.5pt 4pt; font-weight: 600; text-align: right; }
tbody td { padding: 1.6pt 4pt; text-align: right; white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0.9pt solid #000; }
th.l, td.l { text-align: left; }
tr.rec td { background: #eef3fa; }
.ok { color: #2a7f62; font-weight: 700; } .no { color: #b5443b; font-weight: 700; } .err { color: #7a7a7a; font-style: italic; }
.dim { color: var(--dim); }
table.kv { font-size: 8pt; margin: 2pt 0 6pt; }
table.kv td { text-align: left; white-space: normal; padding: 1.4pt 3pt; vertical-align: top; border: 0; }
table.kv td:first-child { color: var(--dim); width: 34%; }
table.kv tr:last-child td { border-bottom: 0; }
code { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 0.8em; }
.jobs code { word-break: break-all; }
p code { overflow-wrap: anywhere; }
.jobs { font-size: 7.6pt; text-indent: 0; }
ol.refs { font-size: 7.9pt; padding-left: 1.6em; margin: 0; text-align: left; }
ol.refs li { margin-bottom: 2pt; }
.colophon { column-span: all; margin-top: 12pt; padding-top: 4pt; border-top: 0.6pt solid var(--rule);
  font-size: 7.4pt; color: var(--dim); text-align: center; }
"""

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style></head>
<body>
<div class="preprint"><span>Q-Scientist preprint · run {run_id}</span><span>{rec_line}</span></div>
<h1>{title}</h1>
<div class="authors">Q-Scientist (autonomous agent, Gemini) and {operator}</div>
<div class="affil">Q-Scientist autonomous laboratory · Qiskit Fall Fest 2026</div>
<div class="date">{date}</div>
<div class="abstract"><h2>Abstract</h2>{abstract}</div>
<div class="cols">
<h2>1&ensp;Introduction</h2>
{intro}
{fig1}
<h2>2&ensp;Methods</h2>
{methods}
{fig3}
<h2>3&ensp;The autonomous agent</h2>
{agent}
{fig4}
<h2>4&ensp;Results</h2>
{results}
{fig2}
{table}
<h2>5&ensp;Discussion and limitations</h2>
{discussion}
<h2>6&ensp;Reproducibility</h2>
{repro}
<h2>References</h2>
<ol class="refs">{refs}</ol>
<div class="colophon">Generated automatically by Q-Scientist from run {run_id}. Numbers are computed from the run log, never typed by hand.
{polish_note}</div>
</div>
</body></html>
"""


# ---------------------------------------------------------------- PDF
def find_chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
        found = shutil.which(c)
        if found:
            return found
    return None


def html_to_pdf(html_path: Path, pdf_path: Path) -> tuple[bool, str]:
    chrome = find_chrome()
    if not chrome:
        return False, "Google Chrome/Chromium not found"
    if pdf_path.exists():
        pdf_path.unlink()
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
           "--run-all-compositor-stages-before-draw", "--virtual-time-budget=4000",
           f"--print-to-pdf={pdf_path}", "--no-pdf-header-footer", html_path.resolve().as_uri()]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"Chrome failed: {type(exc).__name__}"
    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return True, "ok"
    return False, f"Chrome exited {proc.returncode} without writing a PDF"


# ---------------------------------------------------------------- entry point
def write_paper(run_dir: Path, operator: str = "Mayank Goel", use_llm: bool = True, pdf: bool = True,
                log=lambda m: print(m, file=sys.stderr)) -> dict:
    run_dir = Path(run_dir)
    facts = load_run(run_dir)
    if not facts.experiments:
        raise SystemExit(f"No experiments found in {run_dir}; nothing to write.")
    out = run_dir / "paper"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("fig_*.png"):
        old.unlink()
    figs = make_all(facts, out)
    polisher = Polisher(run_dir, enabled=use_llm)
    paper = Paper(facts, figs, operator, polisher)
    html_text = paper.render()
    html_path = out / "paper.html"
    html_path.write_text(html_text, encoding="utf-8")
    for note in polisher.notes:
        log(f"[paper] {note}")
    if use_llm and not polisher.enabled and not polisher.notes:
        log("[paper] GEMINI_API_KEY not set; using template prose")
    result = {"html": html_path, "pdf": None, "figures": list(figs["paths"].values()),
              "key_numbers": paper.key_numbers(), "llm_used": polisher.used}
    if pdf:
        ok, msg = html_to_pdf(html_path, out / "paper.pdf")
        if ok:
            result["pdf"] = out / "paper.pdf"
        else:
            log(f"[paper] PDF skipped ({msg}); open {html_path} in a browser and print to PDF instead.")
    return result

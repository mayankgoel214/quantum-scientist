"""Publication-style matplotlib figures for the auto-written paper."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .data import Experiment, RunFacts  # noqa: E402

DPI = 200
HW = "#1f4e8c"       # hardware: deep blue
SIM = "#d9822b"      # simulator: amber
OK = "#2a7f62"       # certified: green
BAD = "#b5443b"      # not certified: brick
ERR = "#7a7a7a"      # job error
INK = "#222222"
GRID = "#e6e6e6"
COL_W = 3.45         # inches, one column
FULL_W = 7.1         # inches, two columns

STYLE = {
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 7.2,
    "legend.frameon": False,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}


def _save(fig, path: Path) -> Path:
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def fig_fidelity_vs_n(facts: RunFacts, path: Path) -> Path:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(COL_W, 2.55))
        ns_all = [e.n for e in facts.experiments if e.ok] or [8, 16]
        xmax = max(ns_all) + 4
        ax.axhspan(0, 0.5, color="#f4f1ec", zorder=0, lw=0)
        ax.axhline(0.5, color="#555555", ls=(0, (4, 3)), lw=0.8, zorder=1)
        ax.text(min(ns_all) - 3.3, 0.507, "GME threshold  F = 0.5", ha="left", va="bottom", fontsize=6.8, color="#555555")
        for mode, color, marker, label, dx in (("sim", SIM, "s", "Noisy simulator", -0.45), ("qpu", HW, "o", "Hardware (" + facts.backend + ")", 0.45)):
            exps = [e for e in facts.by_mode(mode) if e.ok]
            if not exps:
                continue
            fit = facts.decay_fit(mode)
            if fit:
                f0, ne = fit
                xs = np.linspace(min(e.n for e in exps) - 2, xmax, 200)
                ax.plot(xs, f0 * np.exp(-xs / ne), color=color, lw=0.8, alpha=0.45, ls=":", zorder=2)
            for e in exps:
                lo = e.fidelity - (e.ci_low if e.ci_low is not None else e.fidelity)
                hi = (e.ci_high if e.ci_high is not None else e.fidelity) - e.fidelity
                ax.errorbar(e.n + dx, e.fidelity, yerr=[[lo], [hi]], fmt=marker, ms=4.2, color=color,
                            mfc=color if e.certified else "white", mec=color, mew=0.9, elinewidth=0.8,
                            capsize=1.8, capthick=0.7, zorder=4)
        rec = facts.record("qpu")
        if rec:
            ax.annotate(f"record: N = {rec.n}", xy=(rec.n + 0.45, rec.fidelity), xytext=(rec.n - 9, rec.fidelity - 0.17),
                        fontsize=6.8, color=HW, arrowprops=dict(arrowstyle="-", color=HW, lw=0.6))
        handles = []
        if facts.by_mode("sim"):
            handles.append(Line2D([], [], marker="s", color=SIM, ls="", ms=4.2, label="Noisy simulator"))
        if facts.by_mode("qpu"):
            handles.append(Line2D([], [], marker="o", color=HW, ls="", ms=4.2, label=f"Hardware ({facts.backend})"))
        if any(e.ok and not e.certified for e in facts.experiments):
            handles.append(Line2D([], [], marker="o", color="#666", mfc="white", ls="", ms=4.2, label="not certified (CI$_{\\rm low} \\leq 0.5$)"))
        ax.legend(handles=handles, loc="upper right", handletextpad=0.3, borderaxespad=0.2)
        ax.set_xlim(min(ns_all) - 4, xmax)
        lows = [e.ci_low for e in facts.experiments if e.ok and e.ci_low is not None] or [0.3]
        ax.set_ylim(max(0.0, min(min(lows) - 0.08, 0.42)), 1.02)
        ax.set_xlabel("Number of qubits $N$")
        ax.set_ylabel("GHZ fidelity $F$")
        return _save(fig, path)


def fit_parity(curve: list[dict], n: int) -> dict:
    phi = np.array([p["phi"] for p in curve], dtype=float)
    par = np.array([p["parity"] for p in curve], dtype=float)
    A = np.column_stack([np.cos(n * phi), np.sin(n * phi), np.ones_like(phi)])
    (a, b, c), *_ = np.linalg.lstsq(A, par, rcond=None)
    amp = float(math.hypot(a, b))
    delta = float(math.atan2(-b, a))
    resid = par - A @ np.array([a, b, c])
    return {"amp": amp, "delta": delta, "offset": float(c), "phi": phi, "parity": par,
            "rms": float(np.sqrt(np.mean(resid ** 2)))}


def fig_parity(exp: Experiment, path: Path) -> tuple[Path, dict]:
    curve = exp.raw.get("parity_curve") or []
    fit = fit_parity(curve, exp.n)
    color = HW if exp.mode == "qpu" else SIM
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.2), gridspec_kw={"width_ratios": [2.3, 1]})
        phi, par = fit["phi"], fit["parity"]
        dense = np.linspace(0, max(phi.max(), 1e-6), 3000)
        ax1.plot(dense / np.pi, fit["amp"] * np.cos(exp.n * dense + fit["delta"]) + fit["offset"], color=color, lw=0.5, alpha=0.35)
        ax1.plot(phi / np.pi, par, "o", ms=3.0, color=color, mec="white", mew=0.4, zorder=3)
        ax1.axhline(0, color="#999", lw=0.5)
        ax1.set_xlabel(r"Phase $\varphi\,/\,\pi$")
        ax1.set_ylabel(r"Parity $\langle P(\varphi)\rangle$")
        ax1.set_xlim(-0.02, phi.max() / np.pi + 0.02)
        lim = max(0.15, np.abs(par).max() * 1.25)
        ax1.set_ylim(-lim, lim)
        ax1.set_title(f"(i) {exp.exp_id}: N = {exp.n}, {len(phi)} phase settings", loc="left")
        # folded: everything collapses onto one period
        folded = np.mod(exp.n * phi + fit["delta"] + np.pi, 2 * np.pi) - np.pi
        xs = np.linspace(-np.pi, np.pi, 300)
        ax2.plot(xs / np.pi, fit["amp"] * np.cos(xs) + fit["offset"], color=color, lw=1.0)
        ax2.plot(folded / np.pi, par, "o", ms=2.8, color=color, mec="white", mew=0.4, alpha=0.9)
        ax2.axhline(0, color="#999", lw=0.5)
        ax2.set_ylim(-lim, lim)
        ax2.set_xlabel(r"$(N\varphi+\delta)\,/\,\pi$  (folded)")
        ax2.set_title(f"(ii) fit amplitude C = {fit['amp']:.3f}", loc="left")
        ax2.set_yticklabels([])
        fig.tight_layout(w_pad=1.2)
        return _save(fig, path), fit


def fig_chip_map(facts: RunFacts, exp: Experiment | None, path: Path) -> Path:
    qs = [q for q in facts.calibration.get("qubits") or [] if q.get("x") is not None and q.get("y") is not None]
    pos = {q["id"]: (q["x"], q["y"]) for q in qs}
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(COL_W, 3.45))
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        segs, errs = [], []
        for e in facts.calibration.get("edges") or []:
            if e["a"] in pos and e["b"] in pos:
                segs.append([pos[e["a"]], pos[e["b"]]])
                err = e.get("error")
                errs.append(min(err, 0.05) if err is not None else np.nan)
        errs_arr = np.array(errs, dtype=float)
        finite = errs_arr[np.isfinite(errs_arr)]
        vmax = float(np.percentile(finite, 95)) if finite.size else 0.02
        vmin = float(finite.min()) if finite.size else 0.0
        cmap = plt.get_cmap("YlOrRd")
        lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(vmin, vmax), linewidths=1.6, zorder=1, capstyle="round")
        lc.set_array(np.nan_to_num(errs_arr, nan=vmax))
        ax.add_collection(lc)
        used = set(exp.raw.get("physical_qubits") or []) if exp else set()
        excluded = set((exp.spec.get("exclude_qubits") or [])) if exp else set()
        xs = [pos[q][0] for q in pos]; ys = [pos[q][1] for q in pos]
        ax.scatter(xs, ys, s=9, c="white", edgecolors="#8a8a8a", linewidths=0.5, zorder=2)
        if exp:
            tree = [[pos[a], pos[b]] for a, b in (exp.raw.get("tree_edges") or []) if a in pos and b in pos]
            ax.add_collection(LineCollection(tree, colors=HW, linewidths=2.6, zorder=3, capstyle="round"))
            ux = [pos[q][0] for q in used if q in pos]; uy = [pos[q][1] for q in used if q in pos]
            ax.scatter(ux, uy, s=14, c=HW, edgecolors="white", linewidths=0.4, zorder=4)
            ex = [q for q in excluded if q in pos]
            if ex:
                ax.scatter([pos[q][0] for q in ex], [pos[q][1] for q in ex], s=26, marker="x", c=BAD, linewidths=1.0, zorder=5)
            root = exp.raw.get("root")
            if root in pos:
                ax.scatter([pos[root][0]], [pos[root][1]], s=70, marker="*", c="#f2c14e", edgecolors=INK, linewidths=0.5, zorder=6)
        ax.set_aspect("equal")
        ax.margins(0.03)
        cb = fig.colorbar(lc, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=35)
        cb.set_label("two-qubit (CZ) gate error", fontsize=7.2)
        cb.ax.tick_params(labelsize=6.5)
        cb.outline.set_linewidth(0.4)
        handles = [Line2D([], [], color=HW, lw=2.4, label=f"GHZ tree ({exp.exp_id}, N={exp.n})" if exp else "GHZ tree")]
        if exp:
            handles.append(Line2D([], [], marker="*", color="#f2c14e", mec=INK, mew=0.5, ls="", ms=8, label=f"root q{exp.raw.get('root')}"))
            if excluded:
                handles.append(Line2D([], [], marker="x", color=BAD, ls="", ms=5, label="excluded by agent"))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=3, fontsize=6.6, handletextpad=0.3, columnspacing=0.8)
        return _save(fig, path)


def fig_timeline(facts: RunFacts, path: Path) -> Path:
    t0 = facts.started_at
    with plt.rc_context(STYLE):
        fig, (ax, axb) = plt.subplots(2, 1, figsize=(FULL_W, 3.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})

        total_h = facts.duration_hours or 0.0
        unit, scale = ("Hours", 3600.0) if total_h >= 1.5 else ("Minutes", 60.0)

        def hrs(t):
            return (t - t0).total_seconds() / scale if t and t0 else 0.0

        best = 0
        labelled: list[tuple[float, int]] = []
        rec_x, rec_y = [0.0], [0]
        for e in facts.experiments:
            x = hrs(e.started or e.finished)
            marker = "o" if e.mode == "qpu" else "s"
            if not e.ok:
                ax.scatter([x], [e.n], marker="x", s=30, c=ERR, linewidths=1.1, zorder=4)
                continue
            c = OK if e.certified else BAD
            ax.scatter([x], [e.n], marker=marker, s=34 if e.mode == "qpu" else 26, c=c if e.mode == "qpu" else "white",
                       edgecolors=c, linewidths=1.0, zorder=4)
            span = max(hrs(facts.finished_at), 1e-9)
            if e.mode == "qpu" and all(abs(x - px) > 0.035 * span or pn != e.n for px, pn in labelled):
                ax.annotate(e.exp_id, (x, e.n), xytext=(0, 5.5), textcoords="offset points", ha="center", fontsize=5.6, color="#777")
                labelled.append((x, e.n))
            if e.mode == "qpu" and e.certified and e.n > best:
                best = e.n
                rec_x.append(x); rec_y.append(best)
        end = hrs(facts.finished_at)
        rec_x.append(end); rec_y.append(best)
        ax.step(rec_x, rec_y, where="post", color=HW, lw=1.0, alpha=0.6, zorder=2)
        handles = [
            Line2D([], [], marker="o", color=OK, ls="", ms=5, label="hardware, certified"),
            Line2D([], [], marker="o", color=BAD, ls="", ms=5, label="hardware, not certified"),
            Line2D([], [], marker="s", color=OK, mfc="white", ls="", ms=4.5, label="simulator, certified"),
            Line2D([], [], marker="s", color=BAD, mfc="white", ls="", ms=4.5, label="simulator, not certified"),
            Line2D([], [], marker="x", color=ERR, ls="", ms=5, label="job error"),
            Line2D([], [], color=HW, alpha=0.6, lw=1.0, label="hardware record"),
        ]
        ax.legend(handles=handles, loc="upper left", ncol=3, fontsize=6.6, columnspacing=1.0, handletextpad=0.3)
        ns = [e.n for e in facts.experiments] or [0]
        ax.set_ylim(0, max(ns) * 1.5 + 4)
        ax.set_ylabel("N attempted")
        # budget panel
        bx, qpu, llm = [0.0], [0.0], [0.0]
        maxq = facts.max_qpu_seconds or 0
        maxl = facts.max_llm_calls or 0
        for ev in facts.events:
            if ev.get("type") != "budget":
                continue
            from .data import _parse_ts
            d = ev.get("data") or {}
            bx.append(hrs(_parse_ts(ev.get("ts"))))
            mq = d.get("max_qpu_seconds") or maxq
            ml = d.get("max_llm_calls") or maxl
            qpu.append(100 * (d.get("qpu_seconds") or 0) / mq if mq else 0)
            llm.append(100 * (d.get("llm_calls") or 0) / ml if ml else 0)
        bx.append(end); qpu.append(qpu[-1]); llm.append(llm[-1])
        axb.step(bx, qpu, where="post", color=HW, lw=1.1, label="QPU seconds")
        axb.fill_between(bx, qpu, step="post", color=HW, alpha=0.12, lw=0)
        axb.step(bx, llm, where="post", color="#7b5ea7", lw=1.0, ls="--", label="LLM calls")
        axb.set_ylim(0, max(10, max(qpu + llm) * 1.25))
        axb.set_ylabel("budget used (%)")
        axb.set_xlabel(f"{unit} since run start")
        axb.legend(loc="upper left", ncol=2, fontsize=6.6)
        axb.set_xlim(-0.03 * max(end, 0.1), end * 1.02 + 0.01)
        return _save(fig, path)


def make_all(facts: RunFacts, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = facts.record_overall()
    figs = {"fidelity": fig_fidelity_vs_n(facts, out_dir / "fig_fidelity_vs_n.png")}
    parity_exp = rec if rec and rec.raw.get("parity_curve") else next(
        (e for e in sorted(facts.experiments, key=lambda e: -e.n) if e.ok and e.raw.get("parity_curve")), None)
    fit = None
    if parity_exp:
        figs["parity"], fit = fig_parity(parity_exp, out_dir / "fig_parity.png")
    chip_exp = rec or next((e for e in facts.experiments if e.raw.get("tree_edges")), None)
    if facts.calibration.get("qubits"):
        figs["chip"] = fig_chip_map(facts, chip_exp, out_dir / "fig_chip_map.png")
    figs["timeline"] = fig_timeline(facts, out_dir / "fig_timeline.png")
    return {"paths": figs, "parity_exp": parity_exp, "parity_fit": fit, "chip_exp": chip_exp}

"""Guardrails for the queue-aware hardware batch planner (scripts/qpu_batch.py)."""
from scripts.qpu_batch import _full_spec, validate_plan
from qsci.core import estimate_qpu_seconds


def test_validator_adds_anchor_and_matched_pair():
    plan = [{"n_qubits": 48, "layout_strategy": "weighted_tree", "shots": 1000},
            {"n_qubits": 56, "layout_strategy": "best_chain", "shots": 1000}]
    clean, notes = validate_plan(plan, 480)
    assert any(e["n_qubits"] <= 16 for e in clean), notes
    chain = {e["n_qubits"] for e in clean if e["layout_strategy"] == "best_chain"}
    tree = {e["n_qubits"] for e in clean if e["layout_strategy"] == "weighted_tree"}
    assert {n for n in chain & tree if 20 <= n <= 32}, notes


def test_validator_respects_budget_and_keeps_protected():
    plan = [{"n_qubits": n, "layout_strategy": "weighted_tree", "shots": 1000} for n in (12, 24, 40, 48, 56, 64)]
    plan.append({"n_qubits": 24, "layout_strategy": "best_chain", "shots": 1000})
    clean, _ = validate_plan(plan, 200)
    total = sum(estimate_qpu_seconds(_full_spec(e)) for e in clean)
    assert total <= 200
    assert any(e["n_qubits"] == 12 for e in clean)
    assert any(e["n_qubits"] == 24 and e["layout_strategy"] == "best_chain" for e in clean)


def test_validator_drops_garbage():
    clean, notes = validate_plan([{"n_qubits": "x"}, {"n_qubits": 20, "layout_strategy": "nonsense"},
                                  {"n_qubits": 999, "layout_strategy": "weighted_tree"}], 480)
    assert all(e["layout_strategy"] in ("weighted_tree", "best_chain", "bfs_tree") for e in clean)
    assert all(4 <= e["n_qubits"] <= 100 for e in clean)
    assert len(notes) >= 3


def test_chain_uses_linear_circuit():
    assert _full_spec({"n_qubits": 24, "layout_strategy": "best_chain"})["circuit_variant"] == "linear"
    assert _full_spec({"n_qubits": 24, "layout_strategy": "weighted_tree"})["circuit_variant"] == "tree"


def test_failed_confirmation_withdraws_record():
    from pathlib import Path
    from qsci.paper.data import Experiment, RunFacts

    def ex(i, n, lo, conf=None):
        spec = {"n_qubits": n, "layout_strategy": "weighted_tree"}
        if conf:
            spec["confirmation_of"] = conf
        return Experiment(i, "qpu", n, {"spec": spec, "fidelity": lo + 0.05, "ci_low": lo, "ci_high": lo + 0.1})

    exps = [ex("e1", 24, 0.70), ex("e2", 40, 0.52), ex("e3", 40, 0.33, conf="e2")]
    f = RunFacts(Path("."), "r", {}, [], exps, None)
    assert f.refuted(exps[1]) and f.record("qpu").exp_id == "e1"
    assert [(a.exp_id, b.exp_id) for a, b in f.refutations()] == [("e2", "e3")]
    exps[2].raw["ci_low"] = 0.51  # a confirmation that also certifies keeps the record
    assert not f.refuted(exps[1]) and f.record("qpu").n == 40  # larger N wins regardless of CI width
    twin = Experiment("e3-twin", "sim", 40, {"spec": {"n_qubits": 40, "confirmation_of": "e2"}, "fidelity": 0.7,
                                             "ci_low": 0.68, "ci_high": 0.72})
    exps[2].raw["ci_low"] = 0.33
    f2 = RunFacts(Path("."), "r", {}, [], exps + [twin], None)
    assert f2.refuted(exps[1]), "a twin prediction must not count as a confirmation"

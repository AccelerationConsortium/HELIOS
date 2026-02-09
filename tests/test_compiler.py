"""Compiler unit tests — covers cycle detection, edge cases, and determinism."""
from __future__ import annotations

import pytest

from app.services.compiler import CompileError, compile_protocol


def _proto(*steps: dict) -> dict:
    return {"steps": list(steps)}


POLICY = {"max_temp_c": 95, "max_volume_ul": 1000}
INPUTS = {"instrument_id": "sim-1"}


# ─── basic happy path ───────────────────────────────────────────────

def test_single_step_compiles() -> None:
    compiled, h = compile_protocol(
        protocol=_proto({"id": "a", "primitive": "aspirate", "params": {"volume_ul": 50}}),
        inputs=INPUTS,
        policy_snapshot=POLICY,
    )
    assert compiled["kind"] == "linearized_dag"
    assert len(compiled["steps"]) == 1
    assert isinstance(h, str) and len(h) == 64


def test_hash_deterministic() -> None:
    proto = _proto(
        {"id": "a", "primitive": "aspirate"},
        {"id": "b", "primitive": "heat", "depends_on": ["a"]},
    )
    _, h1 = compile_protocol(protocol=proto, inputs=INPUTS, policy_snapshot=POLICY)
    _, h2 = compile_protocol(protocol=proto, inputs=INPUTS, policy_snapshot=POLICY)
    assert h1 == h2


def test_different_inputs_different_hash() -> None:
    proto = _proto({"id": "a", "primitive": "aspirate"})
    _, h1 = compile_protocol(protocol=proto, inputs=INPUTS, policy_snapshot=POLICY)
    _, h2 = compile_protocol(protocol=proto, inputs={"instrument_id": "sim-2"}, policy_snapshot=POLICY)
    assert h1 != h2


# ─── dependency chain ───────────────────────────────────────────────

def test_linear_chain_compiles() -> None:
    proto = _proto(
        {"id": "a", "primitive": "aspirate"},
        {"id": "b", "primitive": "heat", "depends_on": ["a"]},
        {"id": "c", "primitive": "eis", "depends_on": ["b"]},
    )
    compiled, _ = compile_protocol(protocol=proto, inputs=INPUTS, policy_snapshot=POLICY)
    assert len(compiled["steps"]) == 3


def test_diamond_dependency_compiles() -> None:
    proto = _proto(
        {"id": "a", "primitive": "aspirate"},
        {"id": "b", "primitive": "heat", "depends_on": ["a"]},
        {"id": "c", "primitive": "eis", "depends_on": ["a"]},
        {"id": "d", "primitive": "wait", "depends_on": ["b", "c"]},
    )
    compiled, _ = compile_protocol(protocol=proto, inputs=INPUTS, policy_snapshot=POLICY)
    assert len(compiled["steps"]) == 4


# ─── cycle detection ────────────────────────────────────────────────

def test_self_cycle_raises() -> None:
    proto = _proto({"id": "a", "primitive": "aspirate", "depends_on": ["a"]})
    with pytest.raises(CompileError, match="circular dependency"):
        compile_protocol(protocol=proto, inputs=INPUTS, policy_snapshot=POLICY)


def test_two_node_cycle_raises() -> None:
    proto = _proto(
        {"id": "a", "primitive": "aspirate", "depends_on": ["b"]},
        {"id": "b", "primitive": "heat", "depends_on": ["a"]},
    )
    with pytest.raises(CompileError, match="circular dependency"):
        compile_protocol(protocol=proto, inputs=INPUTS, policy_snapshot=POLICY)


def test_three_node_cycle_raises() -> None:
    proto = _proto(
        {"id": "a", "primitive": "aspirate", "depends_on": ["c"]},
        {"id": "b", "primitive": "heat", "depends_on": ["a"]},
        {"id": "c", "primitive": "eis", "depends_on": ["b"]},
    )
    with pytest.raises(CompileError, match="circular dependency"):
        compile_protocol(protocol=proto, inputs=INPUTS, policy_snapshot=POLICY)


# ─── validation errors ──────────────────────────────────────────────

def test_missing_steps_key_raises() -> None:
    with pytest.raises(CompileError, match="protocol.steps must be a list"):
        compile_protocol(protocol={}, inputs=INPUTS, policy_snapshot=POLICY)


def test_steps_not_a_list_raises() -> None:
    with pytest.raises(CompileError, match="protocol.steps must be a list"):
        compile_protocol(protocol={"steps": "not-a-list"}, inputs=INPUTS, policy_snapshot=POLICY)


def test_duplicate_step_id_raises() -> None:
    proto = _proto(
        {"id": "dup", "primitive": "aspirate"},
        {"id": "dup", "primitive": "heat"},
    )
    with pytest.raises(CompileError, match="duplicate step id"):
        compile_protocol(protocol=proto, inputs=INPUTS, policy_snapshot=POLICY)


def test_missing_dependency_raises() -> None:
    proto = _proto({"id": "a", "primitive": "aspirate", "depends_on": ["nonexistent"]})
    with pytest.raises(CompileError, match="does not exist"):
        compile_protocol(protocol=proto, inputs=INPUTS, policy_snapshot=POLICY)


def test_missing_primitive_raises() -> None:
    proto = _proto({"id": "a"})
    with pytest.raises(CompileError, match="primitive is required"):
        compile_protocol(protocol=proto, inputs=INPUTS, policy_snapshot=POLICY)


def test_depends_on_not_list_raises() -> None:
    proto = _proto({"id": "a", "primitive": "aspirate", "depends_on": "b"})
    with pytest.raises(CompileError, match="depends_on must be a list"):
        compile_protocol(protocol=proto, inputs=INPUTS, policy_snapshot=POLICY)


def test_resources_not_list_raises() -> None:
    proto = _proto({"id": "a", "primitive": "aspirate", "resources": "res"})
    with pytest.raises(CompileError, match="resources must be a list"):
        compile_protocol(protocol=proto, inputs=INPUTS, policy_snapshot=POLICY)


# ─── auto-generated step IDs ────────────────────────────────────────

def test_auto_generated_ids_when_missing() -> None:
    proto = _proto({"primitive": "aspirate"}, {"primitive": "heat"})
    compiled, _ = compile_protocol(protocol=proto, inputs=INPUTS, policy_snapshot=POLICY)
    keys = [s["step_key"] for s in compiled["steps"]]
    assert "step-1" in keys
    assert "step-2" in keys

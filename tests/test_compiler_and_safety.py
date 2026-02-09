from __future__ import annotations

from app.services.compiler import compile_protocol
from app.services.safety import evaluate_preflight


def test_compiler_hash_is_stable_for_same_input() -> None:
    protocol = {
        "steps": [
            {"id": "a", "primitive": "aspirate", "params": {"volume_ul": 100}},
            {"id": "b", "primitive": "heat", "depends_on": ["a"], "params": {"temp_c": 40}},
        ]
    }
    inputs = {"instrument_id": "sim-1"}
    policy = {"max_temp_c": 90, "max_volume_ul": 200}

    compiled_1, hash_1 = compile_protocol(protocol=protocol, inputs=inputs, policy_snapshot=policy)
    compiled_2, hash_2 = compile_protocol(protocol=protocol, inputs=inputs, policy_snapshot=policy)

    assert compiled_1 == compiled_2
    assert hash_1 == hash_2


def test_safety_rejects_over_threshold_heat() -> None:
    compiled = {
        "steps": [
            {"step_key": "s1", "primitive": "heat", "params": {"temp_c": 120}},
        ]
    }
    policy = {
        "max_temp_c": 80,
        "max_volume_ul": 500,
        "allowed_primitives": ["heat"],
    }

    result = evaluate_preflight(compiled_graph=compiled, policy_snapshot=policy)
    assert result.allowed is False
    assert result.violations

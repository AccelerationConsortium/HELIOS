from __future__ import annotations

import json
from datetime import UTC, datetime

from app.services.llm_proposer_canary import (
    LLMCanaryState,
    canary_arms,
    should_offer_llm_arm,
    update_canary_state,
)

_NOW = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)


def test_initial_state_is_enabled_low_weight():
    state = LLMCanaryState()
    assert state.enabled is True
    assert state.disabled_reason is None
    assert 0.0 < state.weight <= 0.2
    assert state.n_uses == 0


def test_good_rewards_keep_enabled_and_raise_ema():
    state = LLMCanaryState()
    for _ in range(5):
        state = update_canary_state(state, reward=0.5, now=_NOW)

    assert state.enabled is True
    assert state.n_uses == 5
    assert state.reward_ema > 0.0
    assert state.consecutive_poor == 0


def test_consecutive_poor_rewards_auto_disable():
    state = LLMCanaryState()
    for _ in range(3):  # max_consecutive_poor default 3
        state = update_canary_state(state, reward=-0.4, now=_NOW)

    assert state.enabled is False
    assert state.disabled_reason == "consecutive_poor_rewards"


def test_safety_warnings_auto_disable():
    state = LLMCanaryState()
    state = update_canary_state(state, reward=0.3, safety_warning=True, now=_NOW)
    state = update_canary_state(state, reward=0.3, safety_warning=True, now=_NOW)

    assert state.enabled is False
    assert state.disabled_reason == "safety_warnings"


def test_low_reward_ema_floor_auto_disable():
    state = LLMCanaryState()
    # reward -0.3 is above the (lowered) poor threshold -0.5, so the
    # consecutive-poor path never fires; but the EMA settles at -0.3, below the
    # default floor -0.2, disabling after enough samples.
    for _ in range(6):
        state = update_canary_state(
            state, reward=-0.3, poor_reward_threshold=-0.5, now=_NOW
        )

    assert state.enabled is False
    assert state.disabled_reason == "reward_ema_below_floor"


def test_disabled_state_is_sticky():
    state = LLMCanaryState(enabled=False, disabled_reason="safety_warnings")
    after = update_canary_state(state, reward=0.9, now=_NOW)
    assert after.enabled is False
    assert after.disabled_reason == "safety_warnings"


def test_should_offer_llm_arm_respects_flag_and_state():
    enabled = LLMCanaryState()
    disabled = LLMCanaryState(enabled=False, disabled_reason="safety_warnings")

    assert should_offer_llm_arm(enabled, canary_enabled=True) is True
    assert should_offer_llm_arm(enabled, canary_enabled=False) is False
    assert should_offer_llm_arm(disabled, canary_enabled=True) is False


def test_canary_arms_adds_low_weight_llm_arm_when_offered():
    state = LLMCanaryState()

    arms, priors = canary_arms(("bo", "random"), state, canary_enabled=True)
    assert "llm" in arms
    assert priors["llm"] == state.weight

    arms_off, priors_off = canary_arms(("bo", "random"), state, canary_enabled=False)
    assert "llm" not in arms_off
    assert priors_off == {}


def test_state_is_json_safe_and_deterministic():
    state = update_canary_state(LLMCanaryState(), reward=0.4, now=_NOW)
    json.dumps(state.model_dump(mode="json"))
    assert state.updated_at == _NOW


def test_config_flag_default_false(monkeypatch):
    monkeypatch.delenv("LLM_PROPOSER_CANARY_ENABLED", raising=False)
    from app.core.config import Settings

    assert Settings().llm_proposer_canary_enabled is False


def test_import_smoke():
    import app.services.llm_proposer_canary  # noqa: F401

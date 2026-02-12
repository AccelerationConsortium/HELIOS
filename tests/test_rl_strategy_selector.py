"""Tests for RL-based strategy selector."""
from __future__ import annotations

import pytest
import tempfile
from pathlib import Path

from app.services.rl_strategy_selector import (
    RLState,
    RLStrategySelector,
    RLConfig,
    ExperienceReplay,
    Experience,
    ACTIONS,
    QLearningAgent,
)
from app.services.rl_reward import (
    RewardConfig,
    compute_immediate_reward,
    compute_terminal_reward,
    analyze_reward_trace,
)
from app.services.rl_data_collector import action_from_backend_name
from app.services.strategy_selector import (
    CampaignSnapshot,
    DiagnosticSignals,
    compute_diagnostics,
)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def make_test_snapshot(round_num: int = 1, max_rounds: int = 10) -> CampaignSnapshot:
    """Create a test campaign snapshot."""
    return CampaignSnapshot(
        round_number=round_num,
        max_rounds=max_rounds,
        n_observations=round_num * 5,
        n_dimensions=3,
        has_categorical=False,
        has_log_scale=False,
        kpi_history=tuple([90.0 + i * 0.5 for i in range(round_num)]),
        direction="maximize",
        last_batch_kpis=(91.0, 91.5, 92.0),
        last_batch_params=([{"temp": 25}, {"temp": 30}, {"temp": 35}]),
        best_kpi_so_far=92.0,
        all_params=tuple([{"temp": 20 + i} for i in range(round_num * 3)]),
        all_kpis=tuple([90.0 + i * 0.3 for i in range(round_num * 3)]),
    )


# ---------------------------------------------------------------------------
# RLState tests
# ---------------------------------------------------------------------------

class TestRLState:
    """Test RL state representation."""

    def test_from_snapshot(self):
        """Test state extraction from snapshot."""
        snapshot = make_test_snapshot(round_num=5, max_rounds=10)
        diagnostics = compute_diagnostics(snapshot)

        state = RLState.from_snapshot(snapshot, diagnostics)

        # Verify features
        assert 0.0 <= state.progress <= 1.0
        assert state.progress == pytest.approx(0.5, abs=0.01)  # round 5/10
        assert 0.0 <= state.space_coverage <= 1.0
        assert state.has_categorical == 0.0
        assert state.has_log_scale == 0.0

    def test_to_array(self):
        """Test conversion to numpy array."""
        snapshot = make_test_snapshot()
        diagnostics = compute_diagnostics(snapshot)
        state = RLState.from_snapshot(snapshot, diagnostics)

        arr = state.to_array()

        assert arr.shape == (15,)  # 16 features (was 15 in code, should be 15)
        assert all(0.0 <= x <= 1.0 for x in arr)


# ---------------------------------------------------------------------------
# Experience Replay tests
# ---------------------------------------------------------------------------

class TestExperienceReplay:
    """Test experience replay buffer."""

    def test_add_and_sample(self):
        """Test adding and sampling experiences."""
        buffer = ExperienceReplay(capacity=10)
        snapshot = make_test_snapshot()
        diagnostics = compute_diagnostics(snapshot)
        state = RLState.from_snapshot(snapshot, diagnostics)

        # Add experiences
        for i in range(5):
            exp = Experience(
                state=state,
                action=i % 4,
                reward=0.1 * i,
                next_state=state,
                done=False,
            )
            buffer.add(exp)

        assert len(buffer) == 5

        # Sample
        batch = buffer.sample(3)
        assert len(batch) == 3
        assert all(isinstance(exp, Experience) for exp in batch)

    def test_capacity_limit(self):
        """Test buffer capacity limit."""
        buffer = ExperienceReplay(capacity=3)
        snapshot = make_test_snapshot()
        diagnostics = compute_diagnostics(snapshot)
        state = RLState.from_snapshot(snapshot, diagnostics)

        # Add 5 experiences (capacity 3)
        for i in range(5):
            exp = Experience(state=state, action=0, reward=i, next_state=None, done=False)
            buffer.add(exp)

        # Should only keep last 3
        assert len(buffer) == 3

    def test_save_load(self):
        """Test saving and loading buffer."""
        buffer = ExperienceReplay()
        snapshot = make_test_snapshot()
        diagnostics = compute_diagnostics(snapshot)
        state = RLState.from_snapshot(snapshot, diagnostics)

        # Add some experiences
        for i in range(3):
            exp = Experience(state=state, action=i, reward=0.1, next_state=None, done=False)
            buffer.add(exp)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.pkl"
            buffer.save(path)

            # Load into new buffer
            new_buffer = ExperienceReplay()
            new_buffer.load(path)

            assert len(new_buffer) == 3


# ---------------------------------------------------------------------------
# Q-Learning Agent tests
# ---------------------------------------------------------------------------

class TestQLearningAgent:
    """Test Q-learning agent."""

    def test_select_action(self):
        """Test action selection."""
        config = RLConfig(epsilon=0.1)
        agent = QLearningAgent(config)

        snapshot = make_test_snapshot()
        diagnostics = compute_diagnostics(snapshot)
        state = RLState.from_snapshot(snapshot, diagnostics)

        # Deterministic mode (no exploration)
        action = agent.select_action(state, explore=False)
        assert action in ACTIONS.keys()

        # Exploration mode
        action = agent.select_action(state, explore=True)
        assert action in ACTIONS.keys()

    def test_update(self):
        """Test Q-value update."""
        config = RLConfig(learning_rate=0.1, gamma=0.9)
        agent = QLearningAgent(config)

        snapshot = make_test_snapshot()
        diagnostics = compute_diagnostics(snapshot)
        state = RLState.from_snapshot(snapshot, diagnostics)
        next_state = state  # Simplified

        # Get initial Q-value
        state_key = agent._discretize_state(state)
        action = 0
        initial_q = agent.q_table[state_key][action]

        # Update
        agent.update(state, action, reward=1.0, next_state=next_state, done=False)

        # Q-value should change
        updated_q = agent.q_table[state_key][action]
        assert updated_q != initial_q

    def test_save_load(self):
        """Test saving and loading Q-table."""
        config = RLConfig()
        agent = QLearningAgent(config)

        snapshot = make_test_snapshot()
        diagnostics = compute_diagnostics(snapshot)
        state = RLState.from_snapshot(snapshot, diagnostics)

        # Update some Q-values
        agent.update(state, 0, 1.0, None, True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q_table.pkl"
            agent.save(path)

            # Load into new agent
            new_agent = QLearningAgent(config)
            new_agent.load(path)

            # Q-values should match
            state_key = agent._discretize_state(state)
            assert new_agent.q_table[state_key][0] == pytest.approx(agent.q_table[state_key][0])


# ---------------------------------------------------------------------------
# RLStrategySelector tests
# ---------------------------------------------------------------------------

class TestRLStrategySelector:
    """Test RL strategy selector."""

    def test_select_action(self):
        """Test action selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RLConfig(model_save_path=f"{tmpdir}/model.pkl")
            selector = RLStrategySelector(config)

            snapshot = make_test_snapshot()
            diagnostics = compute_diagnostics(snapshot)

            action, backend = selector.select_action(snapshot, diagnostics, explore=False)

            assert action in ACTIONS.keys()
            assert isinstance(backend, str)

    def test_learn_from_experience(self):
        """Test online learning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RLConfig(model_save_path=f"{tmpdir}/model.pkl")
            selector = RLStrategySelector(config)

            snapshot = make_test_snapshot()
            diagnostics = compute_diagnostics(snapshot)
            state = RLState.from_snapshot(snapshot, diagnostics)

            # Learn from transition
            selector.learn_from_experience(
                state=state,
                action=0,
                reward=0.5,
                next_state=state,
                done=False,
            )

            # Check replay buffer
            assert len(selector.replay_buffer) == 1

    def test_save_load(self):
        """Test saving and loading model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RLConfig(
                model_save_path=f"{tmpdir}/model.pkl",
                replay_save_path=f"{tmpdir}/replay.pkl",
            )
            selector = RLStrategySelector(config)

            snapshot = make_test_snapshot()
            diagnostics = compute_diagnostics(snapshot)
            state = RLState.from_snapshot(snapshot, diagnostics)

            # Learn something
            selector.learn_from_experience(state, 0, 0.5, state, False)

            # Save
            selector.save()

            # Load into new selector
            new_selector = RLStrategySelector(config)

            # Should have loaded buffer
            assert len(new_selector.replay_buffer) == 1


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------

class TestRewardComputation:
    """Test reward computation."""

    def test_immediate_reward(self):
        """Test immediate reward computation."""
        reward = compute_immediate_reward(
            kpi_prev=90.0,
            kpi_curr=92.0,
            direction="maximize",
            round_num=5,
            max_rounds=10,
            n_qc_failures=0,
        )

        # KPI improved by 2.0 → normalized ~0.2 (scale=10)
        assert reward.kpi_improvement > 0.0
        assert reward.round_cost < 0.0  # Negative
        assert reward.total > 0.0  # Net positive due to improvement

    def test_terminal_reward(self):
        """Test terminal reward computation."""
        snapshot = make_test_snapshot(round_num=10, max_rounds=10)
        diagnostics = compute_diagnostics(snapshot)

        terminal = compute_terminal_reward(
            snapshot=snapshot,
            diagnostics=diagnostics,
            target_reached=True,
        )

        # Should have convergence bonus
        assert terminal > 0.0

    def test_analyze_reward_trace(self):
        """Test reward trace analysis."""
        rewards = [
            compute_immediate_reward(90.0, 91.0, "maximize", 1, 10, 0),
            compute_immediate_reward(91.0, 92.0, "maximize", 2, 10, 0),
            compute_immediate_reward(92.0, 92.5, "maximize", 3, 10, 1),  # 1 QC failure
        ]

        analysis = analyze_reward_trace(rewards, direction="maximize")

        assert "total_reward" in analysis
        assert "avg_reward" in analysis
        assert analysis["n_rounds"] == 3


# ---------------------------------------------------------------------------
# Data collector tests
# ---------------------------------------------------------------------------

class TestDataCollector:
    """Test historical data collection."""

    def test_action_from_backend_name(self):
        """Test backend → action mapping."""
        assert action_from_backend_name("lhs") == 0  # explore
        assert action_from_backend_name("built_in") == 1  # exploit
        assert action_from_backend_name("optuna_tpe") == 1  # exploit
        assert action_from_backend_name("optuna_cmaes") == 2  # refine
        assert action_from_backend_name("unknown_backend") == 1  # default


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestRLIntegration:
    """End-to-end integration test."""

    def test_full_workflow(self):
        """Test complete RL workflow: select → learn → save → load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RLConfig(
                model_save_path=f"{tmpdir}/model.pkl",
                replay_save_path=f"{tmpdir}/replay.pkl",
                epsilon=0.2,
            )
            selector = RLStrategySelector(config)

            # Simulate a few rounds of a campaign
            for round_num in range(1, 6):
                snapshot = make_test_snapshot(round_num=round_num, max_rounds=10)
                diagnostics = compute_diagnostics(snapshot)

                # Select action
                action, backend = selector.select_action(snapshot, diagnostics, explore=True)

                # Simulate outcome
                state = RLState.from_snapshot(snapshot, diagnostics)
                next_snapshot = make_test_snapshot(round_num=round_num + 1, max_rounds=10)
                next_diagnostics = compute_diagnostics(next_snapshot)
                next_state = RLState.from_snapshot(next_snapshot, next_diagnostics)

                # Compute reward
                kpi_prev = snapshot.best_kpi_so_far
                kpi_next = next_snapshot.best_kpi_so_far
                if kpi_prev and kpi_next:
                    reward = (kpi_next - kpi_prev) / 10.0  # Simplified
                else:
                    reward = 0.0

                # Learn
                selector.learn_from_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=(round_num == 5),
                )

            # Save
            selector.save()

            # Load into new selector
            new_selector = RLStrategySelector(config)

            # Should have learned experience
            assert len(new_selector.replay_buffer) == 5

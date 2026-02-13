"""Tests for RL deep integration: router, enhanced DQN, PER, PPO, reward learning."""
from __future__ import annotations

import pytest
import tempfile
import json
from pathlib import Path

import numpy as np

from app.services.strategy_selector import (
    CampaignSnapshot,
    compute_diagnostics,
)

# Check torch availability for conditional test skipping
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Shared test helpers
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
# SumTree and PER tests
# ---------------------------------------------------------------------------

class TestSumTree:
    """Test SumTree data structure."""

    def test_basic_operations(self):
        from app.services.prioritized_replay import SumTree

        tree = SumTree(capacity=4)
        assert len(tree) == 0
        assert tree.total == 0.0

        tree.add(1.0, "a")
        tree.add(2.0, "b")
        tree.add(3.0, "c")

        assert len(tree) == 3
        assert tree.total == pytest.approx(6.0, abs=0.01)

    def test_sampling(self):
        from app.services.prioritized_replay import SumTree

        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        tree.add(3.0, "c")

        # Sample at different cumulative sums
        idx, priority, data = tree.get(0.5)
        assert data == "a"

        idx, priority, data = tree.get(2.5)
        assert data == "b"

        idx, priority, data = tree.get(5.0)
        assert data == "c"

    def test_update(self):
        from app.services.prioritized_replay import SumTree

        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.add(2.0, "b")

        assert tree.total == pytest.approx(3.0)

        tree.update(0, 5.0)  # Update "a" priority
        assert tree.total == pytest.approx(7.0)

    def test_circular_overwrite(self):
        from app.services.prioritized_replay import SumTree

        tree = SumTree(capacity=2)
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        assert len(tree) == 2

        tree.add(3.0, "c")  # Should overwrite "a"
        assert len(tree) == 2
        assert tree.total == pytest.approx(5.0)  # 2 + 3


class TestPrioritizedReplayBuffer:
    """Test PER buffer."""

    def test_add_and_sample(self):
        from app.services.prioritized_replay import PrioritizedReplayBuffer, PERConfig

        config = PERConfig(capacity=100)
        buffer = PrioritizedReplayBuffer(config)

        # Add some transitions
        for i in range(50):
            buffer.add(("state", i, 0.1, "next_state", False))

        assert len(buffer) == 50

        # Sample
        transitions, indices, is_weights = buffer.sample(16)
        assert len(transitions) == 16
        assert len(indices) == 16
        assert len(is_weights) == 16
        assert all(w > 0 for w in is_weights)

    def test_priority_update(self):
        from app.services.prioritized_replay import PrioritizedReplayBuffer, PERConfig

        config = PERConfig(capacity=100)
        buffer = PrioritizedReplayBuffer(config)

        for i in range(32):
            buffer.add(("state", i, 0.1, "next_state", False))

        transitions, indices, is_weights = buffer.sample(8)
        td_errors = np.random.rand(len(indices))
        buffer.update_priorities(indices, td_errors)

    def test_empty_buffer(self):
        from app.services.prioritized_replay import PrioritizedReplayBuffer, PERConfig

        buffer = PrioritizedReplayBuffer(PERConfig(capacity=10))
        transitions, indices, is_weights = buffer.sample(4)
        assert transitions == []


# ---------------------------------------------------------------------------
# Enhanced DQN tests
# ---------------------------------------------------------------------------

@requires_torch
class TestDQNEnhanced:
    """Test enhanced DQN with Dueling, Double DQN, PER, soft update."""

    @pytest.fixture
    def dqn_config(self, tmp_path):
        from app.services.dqn_strategy_selector import DQNConfig
        return DQNConfig(
            hidden_dims=[32, 16],
            dueling=True,
            double_dqn=True,
            use_per=True,
            use_soft_update=True,
            tau=0.1,
            batch_size=8,
            replay_capacity=100,
            model_save_path=str(tmp_path / "test_dqn.pth"),
            replay_save_path=str(tmp_path / "test_replay.pkl"),
        )

    def test_dueling_network(self):
        """Test Dueling Q-Network architecture."""
        import torch
        from app.services.dqn_strategy_selector import DuelingQNetwork, DQNConfig

        config = DQNConfig(hidden_dims=[32, 16], dueling=True)
        net = DuelingQNetwork(state_dim=15, action_dim=4, config=config)

        state = torch.randn(4, 15)
        q_values = net(state)

        assert q_values.shape == (4, 4)

    def test_standard_network(self):
        """Test standard Q-Network."""
        import torch
        from app.services.dqn_strategy_selector import QNetwork, DQNConfig

        config = DQNConfig(hidden_dims=[32, 16], dueling=False)
        net = QNetwork(state_dim=15, action_dim=4, config=config)

        state = torch.randn(4, 15)
        q_values = net(state)
        assert q_values.shape == (4, 4)

    def test_dqn_agent_init(self, dqn_config):
        from app.services.dqn_strategy_selector import DQNAgent

        agent = DQNAgent(state_dim=15, action_dim=4, config=dqn_config)
        assert agent.buffer_size == 0
        assert agent.epsilon == 1.0

    def test_action_selection(self, dqn_config):
        from app.services.dqn_strategy_selector import DQNAgent

        agent = DQNAgent(state_dim=15, action_dim=4, config=dqn_config)
        state = np.random.randn(15).astype(np.float32)

        action = agent.select_action(state, explore=False)
        assert 0 <= action < 4

    def test_q_values(self, dqn_config):
        from app.services.dqn_strategy_selector import DQNAgent

        agent = DQNAgent(state_dim=15, action_dim=4, config=dqn_config)
        state = np.random.randn(15).astype(np.float32)

        q_vals = agent.get_q_values(state)
        assert q_vals.shape == (4,)

    def test_store_and_train(self, dqn_config):
        """Test full training loop with PER."""
        from app.services.dqn_strategy_selector import DQNAgent

        agent = DQNAgent(state_dim=15, action_dim=4, config=dqn_config)

        # Fill buffer with transitions
        for _ in range(20):
            state = np.random.randn(15).astype(np.float32)
            next_state = np.random.randn(15).astype(np.float32)
            agent.store_transition(state, np.random.randint(4), 0.5, next_state, False)

        assert agent.buffer_size == 20

        # Train step should succeed
        loss = agent.train_step()
        assert loss is not None
        assert loss >= 0

    def test_soft_update(self, dqn_config):
        """Test soft target network update."""
        import torch
        from app.services.dqn_strategy_selector import DQNAgent

        agent = DQNAgent(state_dim=15, action_dim=4, config=dqn_config)

        # Get initial target params
        target_params_before = [
            p.clone() for p in agent.target_network.parameters()
        ]

        # Fill buffer and train (which triggers soft update)
        for _ in range(20):
            state = np.random.randn(15).astype(np.float32)
            next_state = np.random.randn(15).astype(np.float32)
            agent.store_transition(state, np.random.randint(4), 0.5, next_state, False)

        agent.train_step()

        # Target should have changed (soft update)
        target_params_after = list(agent.target_network.parameters())
        any_changed = any(
            not torch.equal(before, after)
            for before, after in zip(target_params_before, target_params_after)
        )
        assert any_changed

    def test_save_load(self, dqn_config):
        """Test saving and loading DQN with PER."""
        from app.services.dqn_strategy_selector import DQNAgent

        agent1 = DQNAgent(state_dim=15, action_dim=4, config=dqn_config)

        # Add some transitions
        for _ in range(10):
            state = np.random.randn(15).astype(np.float32)
            agent1.store_transition(state, 1, 0.5, state, False)

        agent1.epsilon = 0.5
        agent1.steps = 100
        agent1.save()

        # Load into new agent
        agent2 = DQNAgent(state_dim=15, action_dim=4, config=dqn_config)
        agent2.load()

        assert agent2.epsilon == 0.5
        assert agent2.steps == 100

    def test_dqn_selector_confidence(self):
        """Test DQNStrategySelector confidence estimation."""
        from app.services.dqn_strategy_selector import DQNStrategySelector, DQNConfig

        config = DQNConfig(hidden_dims=[16, 8], batch_size=4, use_per=False)
        selector = DQNStrategySelector(config)

        snapshot = make_test_snapshot(round_num=5, max_rounds=10)
        diagnostics = compute_diagnostics(snapshot)

        confidence = selector.get_confidence(snapshot, diagnostics)
        assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# Strategy Router tests
# ---------------------------------------------------------------------------

class TestStrategyRouter:
    """Test StrategyRouter routing logic."""

    def test_rule_based_mode(self):
        """Router in rule_based mode always uses rule-based selector."""
        from app.services.strategy_router import StrategyRouter, RouterConfig

        router = StrategyRouter(RouterConfig(mode="rule_based"))
        snapshot = make_test_snapshot(round_num=3, max_rounds=10)

        decision = router.select_strategy(snapshot, "test-campaign-1")
        assert decision is not None
        assert decision.backend_name is not None
        assert "[RL-" not in (decision.reason or "")

    def test_rl_mode_with_fallback(self):
        """Router in rl mode falls back to rule-based on low confidence."""
        from app.services.strategy_router import StrategyRouter, RouterConfig

        router = StrategyRouter(RouterConfig(
            mode="rl",
            rl_backend="dqn",
            confidence_threshold=0.99,  # Very high → always fallback
            fallback_on_error=True,
        ))

        snapshot = make_test_snapshot(round_num=3, max_rounds=10)
        decision = router.select_strategy(snapshot, "test-campaign-2")

        # Should fall back to rule-based (confidence < 0.99 for untrained model)
        assert decision is not None

    def test_ab_test_deterministic(self):
        """A/B test assignment is deterministic for same campaign_id."""
        from app.services.strategy_router import StrategyRouter, RouterConfig

        router = StrategyRouter(RouterConfig(mode="ab_test", ab_test_rl_fraction=0.5))

        assignments = []
        for _ in range(10):
            assignment = router._ab_test_assignment("test-campaign-42")
            assignments.append(assignment)

        # All assignments should be identical
        assert len(set(assignments)) == 1

    def test_ab_test_distribution(self):
        """A/B test produces roughly expected distribution."""
        from app.services.strategy_router import StrategyRouter, RouterConfig

        router = StrategyRouter(RouterConfig(mode="ab_test", ab_test_rl_fraction=0.5))

        rl_count = sum(
            1 for i in range(1000)
            if router._ab_test_assignment(f"campaign-{i}") == "rl"
        )

        # Should be roughly 50% (within reasonable margin)
        assert 350 < rl_count < 650

    def test_on_round_complete_no_crash(self):
        """on_round_complete should never raise, even if not set up."""
        from app.services.strategy_router import StrategyRouter, RouterConfig

        router = StrategyRouter(RouterConfig(mode="rule_based"))
        snapshot = make_test_snapshot()
        diagnostics = compute_diagnostics(snapshot)

        # Should not raise
        router.on_round_complete(
            campaign_id="test",
            snapshot=snapshot,
            diagnostics=diagnostics,
            action=0,
            kpi_prev=90.0,
            kpi_curr=91.0,
        )

    def test_on_campaign_complete_no_crash(self):
        """on_campaign_complete should never raise."""
        from app.services.strategy_router import StrategyRouter, RouterConfig

        router = StrategyRouter(RouterConfig(mode="rule_based"))
        router.on_campaign_complete("nonexistent-campaign")  # Should not raise


# ---------------------------------------------------------------------------
# A/B Test Logger tests
# ---------------------------------------------------------------------------

class TestABTestLogger:
    """Test A/B test logging."""

    def test_log_and_query(self, tmp_path):
        from app.services.ab_test_logger import ABTestLogger, ABTestRecord

        db_path = str(tmp_path / "test_ab.db")
        logger_inst = ABTestLogger(db_path=db_path)

        record = ABTestRecord(
            campaign_id="test-1",
            treatment="rule_based",
            n_rounds=10,
            final_kpi=92.5,
            converged=True,
            target_reached=True,
        )
        logger_inst.log_result(record)

        results = logger_inst.get_results()
        assert len(results) == 1
        assert results[0].campaign_id == "test-1"
        assert results[0].treatment == "rule_based"

    def test_compute_summary(self, tmp_path):
        from app.services.ab_test_logger import ABTestLogger, ABTestRecord

        db_path = str(tmp_path / "test_ab2.db")
        logger_inst = ABTestLogger(db_path=db_path)

        for i in range(5):
            logger_inst.log_result(ABTestRecord(
                campaign_id=f"rb-{i}", treatment="rule_based",
                n_rounds=10, final_kpi=90.0 + i, converged=True, target_reached=True,
            ))
            logger_inst.log_result(ABTestRecord(
                campaign_id=f"rl-{i}", treatment="rl_dqn",
                n_rounds=8, final_kpi=91.0 + i, converged=True, target_reached=True,
            ))

        summary = logger_inst.compute_summary()
        assert "rule_based" in summary
        assert "rl_dqn" in summary
        assert summary["rule_based"]["n"] == 5
        assert summary["rl_dqn"]["n"] == 5


# ---------------------------------------------------------------------------
# Reward Weight Learner tests
# ---------------------------------------------------------------------------

class TestRewardWeightLearner:
    """Test meta-learning for reward weights."""

    def test_init_default_weights(self, tmp_path):
        from app.services.reward_weight_learner import RewardWeightLearner, WeightLearnerConfig

        config = WeightLearnerConfig(save_path=str(tmp_path / "weights.json"))
        learner = RewardWeightLearner(config)

        assert learner.weights.alpha == 1.0
        assert learner.weights.beta == 0.01
        assert learner.weights.gamma == 0.5
        assert learner.weights.delta == 0.1

    def test_update_weights(self, tmp_path):
        from app.services.reward_weight_learner import (
            RewardWeightLearner, WeightLearnerConfig, CampaignOutcome,
        )

        config = WeightLearnerConfig(
            save_path=str(tmp_path / "weights.json"),
            min_campaigns_before_learning=1,
        )
        learner = RewardWeightLearner(config)

        # First campaign (warmup)
        outcome1 = CampaignOutcome(
            campaign_id="c1", n_rounds=10, final_kpi=92.0, best_kpi=92.0,
            target_value=95.0, direction="maximize",
            target_reached=False, converged=True,
            reward_trace=[0.1] * 10,
        )
        learner.update_weights(outcome1)

        # Second campaign (actual learning)
        outcome2 = CampaignOutcome(
            campaign_id="c2", n_rounds=8, final_kpi=95.0, best_kpi=95.0,
            target_value=95.0, direction="maximize",
            target_reached=True, converged=True,
            reward_trace=[0.5] * 8,
        )
        new_config = learner.update_weights(outcome2)
        assert new_config is not None

    def test_persistence(self, tmp_path):
        from app.services.reward_weight_learner import (
            RewardWeightLearner, WeightLearnerConfig,
        )

        config = WeightLearnerConfig(save_path=str(tmp_path / "weights.json"))
        learner1 = RewardWeightLearner(config)
        learner1.weights.alpha = 2.5
        learner1._save()

        learner2 = RewardWeightLearner(config)
        assert learner2.weights.alpha == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Neural Reward Model tests
# ---------------------------------------------------------------------------

@requires_torch
class TestRewardModel:
    """Test neural reward model."""

    def test_predict_untrained(self, tmp_path):
        from app.services.reward_model import RewardModel, RewardModelConfig

        config = RewardModelConfig(model_save_path=str(tmp_path / "rm.pth"))
        model = RewardModel(config)

        # Untrained model should return 0.0
        state = np.random.randn(15).astype(np.float32)
        reward = model.predict_reward(state, 0, state)
        assert reward == 0.0

    def test_blend_untrained(self, tmp_path):
        from app.services.reward_model import RewardModel, RewardModelConfig

        config = RewardModelConfig(model_save_path=str(tmp_path / "rm.pth"))
        model = RewardModel(config)

        # lambda=0 → should return handcrafted exactly
        state = np.random.randn(15).astype(np.float32)
        blended = model.blend_reward(0.5, state, 1, state)
        assert blended == pytest.approx(0.5)

    def test_trainer_basic(self, tmp_path):
        from app.services.reward_model import (
            RewardModel, RewardModelTrainer, RewardModelConfig,
        )

        config = RewardModelConfig(
            model_save_path=str(tmp_path / "rm.pth"),
            batch_size=4,
            n_epochs_per_campaign=2,
        )
        model = RewardModel(config)
        trainer = RewardModelTrainer(model, config)

        # Add transitions from a successful campaign
        transitions = [
            (np.random.randn(15).astype(np.float32), i % 4, 0.1 * i,
             np.random.randn(15).astype(np.float32), i == 9)
            for i in range(10)
        ]

        trainer.add_campaign_transitions(transitions, target_reached=True, converged=True)
        assert len(trainer._buffer) == 10

        loss = trainer.train()
        assert loss >= 0
        assert model._campaigns_trained == 1

    def test_lambda_blend_increases(self, tmp_path):
        from app.services.reward_model import RewardModel, RewardModelConfig

        config = RewardModelConfig(
            model_save_path=str(tmp_path / "rm.pth"),
            lambda_warmup_campaigns=10,
            lambda_max=0.5,
        )
        model = RewardModel(config)

        model._campaigns_trained = 0
        assert model.lambda_blend == 0.0

        model._campaigns_trained = 5
        assert model.lambda_blend == pytest.approx(0.25, abs=0.01)

        model._campaigns_trained = 10
        assert model.lambda_blend == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# PPO tests
# ---------------------------------------------------------------------------

@requires_torch
class TestPPO:
    """Test PPO strategy selector."""

    @pytest.fixture
    def ppo_config(self, tmp_path):
        from app.services.ppo_strategy_selector import PPOConfig
        return PPOConfig(
            hidden_dims=[32, 16],
            mini_batch_size=4,
            n_epochs=2,
            model_save_path=str(tmp_path / "test_ppo.pth"),
        )

    def test_actor_critic_network(self, ppo_config):
        import torch
        from app.services.ppo_strategy_selector import ActorCritic

        net = ActorCritic(state_dim=15, action_dim=4, config=ppo_config)
        state = torch.randn(8, 15)

        logits, value = net(state)
        assert logits.shape == (8, 4)
        assert value.shape == (8,)

    def test_get_action_and_value(self, ppo_config):
        import torch
        from app.services.ppo_strategy_selector import ActorCritic

        net = ActorCritic(state_dim=15, action_dim=4, config=ppo_config)
        state = torch.randn(4, 15)

        action, log_prob, entropy, value = net.get_action_and_value(state)
        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)

    def test_rollout_buffer_gae(self):
        from app.services.ppo_strategy_selector import RolloutBuffer

        buffer = RolloutBuffer()
        for i in range(5):
            buffer.add(
                state=np.random.randn(15).astype(np.float32),
                action=i % 4,
                reward=0.1 * (i + 1),
                log_prob=-0.5,
                value=0.5,
                done=(i == 4),
            )

        advantages, returns = buffer.compute_gae(
            last_value=0.0, gamma=0.99, gae_lambda=0.95,
        )
        assert len(advantages) == 5
        assert len(returns) == 5

    def test_ppo_agent_train(self, ppo_config):
        from app.services.ppo_strategy_selector import PPOAgent

        agent = PPOAgent(state_dim=15, action_dim=4, config=ppo_config)

        # Collect rollout
        for i in range(10):
            state = np.random.randn(15).astype(np.float32)
            action, log_prob, value = agent.select_action(state, explore=True)
            agent.store_transition(state, action, 0.1, log_prob, value, i == 9)

        stats = agent.train_on_rollout(last_value=0.0)
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert stats["n_updates"] > 0

    def test_ppo_selector_api(self, ppo_config):
        """Test PPOStrategySelector matches DQN API."""
        from app.services.ppo_strategy_selector import PPOStrategySelector

        selector = PPOStrategySelector(ppo_config)
        snapshot = make_test_snapshot(round_num=5, max_rounds=10)
        diagnostics = compute_diagnostics(snapshot)

        # Select action
        action_id, backend_name = selector.select_action(snapshot, diagnostics)
        assert 0 <= action_id < 4
        assert backend_name in {"explore", "exploit", "refine", "stabilize"}

        # Confidence
        conf = selector.get_confidence(snapshot, diagnostics)
        assert 0.0 <= conf <= 1.0

    def test_ppo_save_load(self, ppo_config):
        from app.services.ppo_strategy_selector import PPOStrategySelector

        selector1 = PPOStrategySelector(ppo_config)
        selector1.agent.steps = 42
        selector1.save()

        selector2 = PPOStrategySelector(ppo_config)
        selector2.load()
        assert selector2.agent.steps == 42

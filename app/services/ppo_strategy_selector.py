"""PPO (Proximal Policy Optimization) strategy selector.

Actor-Critic architecture with clipped surrogate objective for stable
on-policy learning. Better than DQN for:
- Continuous policy improvement (no hard ε-greedy)
- Natural exploration via stochastic policy
- Direct optimization of expected return

Architecture:
    Actor π(a|s):  state(15) → shared(64) → policy logits(4) → softmax
    Critic V(s):   state(15) → shared(64) → value(1)

Training:
    Uses GAE (Generalized Advantage Estimation) for low-variance
    advantage computation, with PPO clip for trust-region updates.

References:
    Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
    Schulman et al., "High-Dimensional Continuous Control Using GAE" (2015)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["PPOStrategySelector", "PPOConfig", "ActorCritic"]

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Provide stub so class definitions don't crash at import time
    class _StubModule:
        pass
    class _StubNN:
        Module = _StubModule
    nn = _StubNN()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# PPO Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """Hyperparameters for PPO agent."""

    # Network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    activation: str = "tanh"  # tanh works better for PPO than relu

    # PPO objective
    clip_epsilon: float = 0.2  # PPO clipping parameter
    n_epochs: int = 4  # Number of update epochs per rollout
    mini_batch_size: int = 8  # Mini-batch size for PPO updates

    # Learning rates
    actor_lr: float = 0.0003
    critic_lr: float = 0.001

    # GAE (Generalized Advantage Estimation)
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda

    # Regularization
    entropy_coeff: float = 0.01  # Entropy bonus to encourage exploration
    value_loss_coeff: float = 0.5  # Value loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    # Rollout
    rollout_length: int = 24  # Max steps per rollout (~ one campaign)

    # Persistence
    model_save_path: str = "models/ppo_selector.pth"


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

def _build_activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    return nn.Tanh()


class ActorCritic(nn.Module):
    """Actor-Critic network with shared trunk.

    Actor: π(a|s) — categorical distribution over 4 actions
    Critic: V(s) — scalar value estimate
    """

    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig):
        super().__init__()
        self.action_dim = action_dim

        # Shared feature trunk
        trunk_layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in config.hidden_dims:
            trunk_layers.append(nn.Linear(input_dim, hidden_dim))
            trunk_layers.append(_build_activation(config.activation))
            input_dim = hidden_dim

        self.trunk = nn.Sequential(*trunk_layers)

        # Actor head: logits → Categorical distribution
        self.actor_head = nn.Linear(input_dim, action_dim)

        # Critic head: state → value
        self.critic_head = nn.Linear(input_dim, 1)

        # Initialize weights with smaller values for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01 if m is self.actor_head else 1.0)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: state → (action_logits, value)."""
        features = self.trunk(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value.squeeze(-1)

    def get_action_and_value(
        self, state: torch.Tensor, action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value.

        If action is provided, evaluate that action's log_prob.
        Otherwise, sample from the policy.
        """
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores one rollout (episode) for PPO training."""

    def __init__(self):
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)

    def compute_gae(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns.

        Args:
            last_value: V(s_T) for bootstrapping
            gamma: Discount factor
            gae_lambda: GAE lambda

        Returns:
            (advantages, returns)
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            if self.dones[t]:
                delta = self.rewards[t] - self.values[t]
                gae = delta
            else:
                delta = self.rewards[t] + gamma * next_value - self.values[t]
                gae = delta + gamma * gae_lambda * gae

            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
            next_value = self.values[t]

        return advantages, returns


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """PPO agent with clipped surrogate objective and GAE."""

    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for PPO. Install with: pip install torch")

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor-Critic network
        self.network = ActorCritic(state_dim, action_dim, config)

        # Optimizer (separate lr for actor and critic is handled via param groups)
        self.optimizer = optim.Adam([
            {"params": self.network.trunk.parameters(), "lr": config.actor_lr},
            {"params": self.network.actor_head.parameters(), "lr": config.actor_lr},
            {"params": self.network.critic_head.parameters(), "lr": config.critic_lr},
        ])

        # Rollout buffer
        self.rollout = RolloutBuffer()

        # Training state
        self.steps = 0
        self.episodes = 0

    def select_action(
        self, state: np.ndarray, explore: bool = True,
    ) -> tuple[int, float, float]:
        """Select action from policy.

        Returns:
            (action_id, log_prob, value_estimate)
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)

            if explore:
                action, log_prob, entropy, value = self.network.get_action_and_value(state_t)
                return action.item(), log_prob.item(), value.item()
            else:
                logits, value = self.network(state_t)
                return logits.argmax().item(), 0.0, value.item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        """Store transition in rollout buffer."""
        self.rollout.add(state, action, reward, log_prob, value, done)

    def train_on_rollout(self, last_value: float = 0.0) -> dict[str, float]:
        """Train on collected rollout using PPO.

        Called at end of campaign (episode) or when rollout is full.

        Returns:
            Dict with loss statistics
        """
        if len(self.rollout) == 0:
            return {}

        # Compute GAE advantages
        advantages, returns = self.rollout.compute_gae(
            last_value, self.config.gamma, self.config.gae_lambda,
        )

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(self.rollout.states))
        actions_t = torch.LongTensor(self.rollout.actions)
        old_log_probs_t = torch.FloatTensor(self.rollout.log_probs)
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = len(self.rollout)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.config.n_epochs):
            # Shuffle indices for mini-batching
            indices = torch.randperm(n)

            for start in range(0, n, self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, n)
                mb_indices = indices[start:end]

                mb_states = states_t[mb_indices]
                mb_actions = actions_t[mb_indices]
                mb_old_log_probs = old_log_probs_t[mb_indices]
                mb_advantages = advantages_t[mb_indices]
                mb_returns = returns_t[mb_indices]

                # Evaluate current policy
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    mb_states, mb_actions,
                )

                # PPO Clipped Surrogate
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon,
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, mb_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_loss_coeff * value_loss
                    + self.config.entropy_coeff * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # Clear rollout
        self.rollout.clear()
        self.episodes += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "n_updates": n_updates,
        }

    def save(self, path: Path | str | None = None) -> None:
        """Save model checkpoint."""
        save_path = Path(path) if path else Path(self.config.model_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "steps": self.steps,
            "episodes": self.episodes,
            "config": self.config,
        }, save_path)
        logger.info("Saved PPO model to %s", save_path)

    def load(self, path: Path | str | None = None) -> None:
        """Load model checkpoint."""
        load_path = Path(path) if path else Path(self.config.model_save_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")

        checkpoint = torch.load(load_path, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.steps = checkpoint["steps"]
        self.episodes = checkpoint.get("episodes", 0)
        logger.info("Loaded PPO model from %s", load_path)


# ---------------------------------------------------------------------------
# PPO Strategy Selector (wrapper matching DQNStrategySelector API)
# ---------------------------------------------------------------------------

class PPOStrategySelector:
    """PPO-based strategy selector matching DQNStrategySelector API.

    Differences from DQN:
    - On-policy (no replay buffer)
    - Stochastic policy with natural exploration
    - Trains at end of episode (campaign), not per-step
    """

    def __init__(self, config: PPOConfig | None = None):
        from app.services.rl_strategy_selector import RLState, ACTIONS

        if config is None:
            config = PPOConfig()

        self.config = config
        self.ACTIONS = ACTIONS

        self.state_dim = 15
        self.action_dim = len(ACTIONS)

        self.agent = PPOAgent(self.state_dim, self.action_dim, config)

        # Track per-campaign rollout for on-policy learning
        self._current_log_prob: float = 0.0
        self._current_value: float = 0.0

        logger.info(
            "Initialized PPO selector: state_dim=%d, action_dim=%d",
            self.state_dim, self.action_dim,
        )

    def select_action(
        self,
        snapshot: Any,
        diagnostics: Any,
        explore: bool = True,
    ) -> tuple[int, str]:
        """Select action using PPO policy.

        Returns:
            (action_id, backend_name)
        """
        from app.services.rl_strategy_selector import RLState

        state = RLState.from_snapshot(snapshot, diagnostics)
        state_array = state.to_array()

        action_id, log_prob, value = self.agent.select_action(state_array, explore=explore)
        self._current_log_prob = log_prob
        self._current_value = value

        backend_name = self.ACTIONS[action_id]
        return action_id, backend_name

    def get_confidence(
        self,
        snapshot: Any,
        diagnostics: Any,
    ) -> float:
        """Estimate confidence based on policy entropy and training maturity."""
        from app.services.rl_strategy_selector import RLState

        state = RLState.from_snapshot(snapshot, diagnostics)
        state_array = state.to_array()

        with torch.no_grad():
            state_t = torch.FloatTensor(state_array).unsqueeze(0)
            logits, value = self.agent.network(state_t)
            dist = torch.distributions.Categorical(logits=logits)

            # Low entropy = high confidence (policy is decisive)
            max_entropy = np.log(self.action_dim)
            entropy_ratio = dist.entropy().item() / max_entropy
            entropy_conf = 1.0 - entropy_ratio

            # Training maturity
            maturity = min(1.0, self.agent.episodes / 20.0)

        confidence = 0.5 * entropy_conf + 0.5 * maturity
        return float(max(0.0, min(1.0, confidence)))

    def learn_from_experience(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any | None,
        done: bool,
    ) -> float | None:
        """Store transition in rollout buffer. Train on episode end.

        PPO is on-policy: we collect a full rollout, then train.
        """
        state_array = state.to_array() if hasattr(state, "to_array") else state

        self.agent.store_transition(
            state=state_array,
            action=action,
            reward=reward,
            log_prob=self._current_log_prob,
            value=self._current_value,
            done=done,
        )

        self.agent.steps += 1

        # Train at end of episode (campaign)
        if done:
            stats = self.agent.train_on_rollout(last_value=0.0)
            if stats:
                logger.info(
                    "PPO training: policy_loss=%.4f value_loss=%.4f entropy=%.4f",
                    stats["policy_loss"], stats["value_loss"], stats["entropy"],
                )
                return stats.get("policy_loss")

        # Or train when rollout is full (mid-campaign)
        elif len(self.agent.rollout) >= self.config.rollout_length:
            # Bootstrap with current value estimate
            next_state_array = (
                next_state.to_array() if next_state and hasattr(next_state, "to_array") else None
            )
            if next_state_array is not None:
                with torch.no_grad():
                    _, last_value = self.agent.network(
                        torch.FloatTensor(next_state_array).unsqueeze(0),
                    )
                    last_val = last_value.item()
            else:
                last_val = 0.0

            stats = self.agent.train_on_rollout(last_value=last_val)
            if stats:
                return stats.get("policy_loss")

        return None

    def save(self, path: Path | str | None = None) -> None:
        """Save PPO model."""
        self.agent.save(path or self.config.model_save_path)

    def load(self, path: Path | str | None = None) -> None:
        """Load PPO model."""
        self.agent.load(path or self.config.model_save_path)

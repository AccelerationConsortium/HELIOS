"""Deep Q-Network (DQN) strategy selector with PyTorch.

Enhanced DQN implementation with:
- **Dueling Network**: Separate value and advantage streams (Wang et al., 2016)
- **Double DQN**: Decoupled action selection and evaluation (van Hasselt et al., 2016)
- **Prioritized Experience Replay (PER)**: Sample important transitions more often (Schaul et al., 2015)
- **Soft target updates**: Polyak averaging instead of hard copy (Lillicrap et al., 2016)

Key improvements over tabular Q-learning:
- No state discretization (handles continuous states directly)
- Better generalization across similar states
- Stable learning through target network + PER IS-weights
- Scalable to larger state spaces
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import random
import pickle
from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Provide a stub nn.Module so class definitions don't crash at import time
    class _StubModule:
        pass
    class _StubNN:
        Module = _StubModule
    nn = _StubNN()  # type: ignore[assignment]

logger = logging.getLogger(__name__)

if not TORCH_AVAILABLE:
    logger.warning("PyTorch not available. DQN selector will not work. Install with: pip install torch")


# ---------------------------------------------------------------------------
# DQN Config
# ---------------------------------------------------------------------------

@dataclass
class DQNConfig:
    """Hyperparameters for DQN agent."""

    # Q-network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    activation: str = "relu"  # relu, tanh, leaky_relu
    dropout: float = 0.0  # Dropout rate (0 = no dropout)
    dueling: bool = True  # Use dueling architecture

    # Learning
    learning_rate: float = 0.001
    gamma: float = 0.95  # Discount factor
    epsilon: float = 1.0  # Initial exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01

    # Training
    batch_size: int = 32
    replay_capacity: int = 10000
    target_update_freq: int = 10  # Update target network every N episodes (fallback for hard update)
    grad_clip: float = 1.0  # Gradient clipping threshold

    # Soft target update (Polyak averaging)
    tau: float = 0.005  # Soft update coefficient: target = tau*online + (1-tau)*target
    use_soft_update: bool = True  # Use soft updates instead of hard copy

    # Double DQN
    double_dqn: bool = True  # Use Double DQN for action evaluation

    # Prioritized Experience Replay
    use_per: bool = True  # Use PER instead of uniform replay
    per_alpha: float = 0.6  # Priority exponent
    per_beta: float = 0.4  # IS weight exponent
    per_beta_increment: float = 0.001
    per_epsilon: float = 1e-6

    # Persistence
    model_save_path: str = "models/dqn_selector.pth"
    replay_save_path: str = "models/dqn_replay.pkl"


# ---------------------------------------------------------------------------
# Q-Network Architectures
# ---------------------------------------------------------------------------

def _build_activation(name: str) -> nn.Module:
    """Build activation layer by name."""
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    else:
        return nn.ReLU()


class QNetwork(nn.Module):
    """Standard Deep Q-Network: maps state → Q-values for each action."""

    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig):
        super().__init__()
        self.config = config

        layers = []
        input_dim = state_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(_build_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: state → Q-values."""
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """Dueling Deep Q-Network (Wang et al., 2016).

    Separates Q-value estimation into:
    - Value stream V(s): how good is this state overall
    - Advantage stream A(s,a): relative benefit of each action

    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    This decomposition helps the network learn state values
    independently from action advantages, improving learning
    when many actions have similar values.
    """

    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig):
        super().__init__()
        self.config = config
        self.action_dim = action_dim

        # Shared feature trunk
        trunk_layers: list[nn.Module] = []
        input_dim = state_dim

        # Use all hidden dims except last for shared trunk
        trunk_dims = config.hidden_dims[:-1] if len(config.hidden_dims) > 1 else []
        for hidden_dim in trunk_dims:
            trunk_layers.append(nn.Linear(input_dim, hidden_dim))
            trunk_layers.append(_build_activation(config.activation))
            if config.dropout > 0:
                trunk_layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim

        self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()
        trunk_out_dim = input_dim

        # Stream hidden dimension (last hidden dim or default)
        stream_dim = config.hidden_dims[-1] if config.hidden_dims else 32

        # Value stream: V(s) → scalar
        self.value_stream = nn.Sequential(
            nn.Linear(trunk_out_dim, stream_dim),
            _build_activation(config.activation),
            nn.Linear(stream_dim, 1),
        )

        # Advantage stream: A(s,a) → per-action advantage
        self.advantage_stream = nn.Sequential(
            nn.Linear(trunk_out_dim, stream_dim),
            _build_activation(config.activation),
            nn.Linear(stream_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: state → Q-values via value + advantage decomposition."""
        features = self.trunk(state)

        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        # Subtracting mean ensures identifiability (unique V and A)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """Enhanced DQN agent with Double DQN, Dueling, PER, and soft target updates."""

    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DQN. Install with: pip install torch")

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Select network architecture
        NetworkClass = DuelingQNetwork if config.dueling else QNetwork

        # Q-network (online)
        self.q_network = NetworkClass(state_dim, action_dim, config)

        # Target network (for stability)
        self.target_network = NetworkClass(state_dim, action_dim, config)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Always in eval mode

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

        # Experience replay (PER or uniform)
        self._use_per = config.use_per
        if self._use_per:
            from app.services.prioritized_replay import PrioritizedReplayBuffer, PERConfig
            per_config = PERConfig(
                capacity=config.replay_capacity,
                alpha=config.per_alpha,
                beta=config.per_beta,
                beta_increment=config.per_beta_increment,
                epsilon=config.per_epsilon,
            )
            self.per_buffer = PrioritizedReplayBuffer(per_config)
            self.replay_buffer = None  # Not used with PER
        else:
            self.per_buffer = None
            self.replay_buffer = deque(maxlen=config.replay_capacity)

        # Training state
        self.epsilon = config.epsilon
        self.steps = 0
        self.episodes = 0

        logger.info(
            "DQN agent initialized: dueling=%s double=%s per=%s soft_update=%s",
            config.dueling, config.double_dqn, config.use_per, config.use_soft_update,
        )

    @property
    def buffer_size(self) -> int:
        """Current replay buffer size."""
        if self._use_per:
            return len(self.per_buffer)
        return len(self.replay_buffer)

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """ε-greedy action selection."""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        # Greedy action from Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state (for confidence estimation)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.squeeze(0).numpy()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray | None,
        done: bool,
    ) -> None:
        """Add transition to replay buffer."""
        transition = (state, action, reward, next_state, done)
        if self._use_per:
            # New transitions get max priority (sampled at least once)
            self.per_buffer.add(transition, priority=None)
        else:
            self.replay_buffer.append(transition)

    def train_step(self) -> float | None:
        """Sample batch and perform one gradient descent step.

        Uses Double DQN for target computation and PER IS-weights if enabled.

        Returns:
            Average loss or None if insufficient data
        """
        buf_size = self.buffer_size
        if buf_size < self.config.batch_size:
            return None

        # --- Sample mini-batch ---
        if self._use_per:
            transitions, per_indices, is_weights = self.per_buffer.sample(
                self.config.batch_size
            )
            if not transitions:
                return None
            is_weights_t = torch.FloatTensor(is_weights).unsqueeze(1)
        else:
            transitions = random.sample(self.replay_buffer, self.config.batch_size)
            per_indices = None
            is_weights_t = None

        states, actions, rewards, next_states, dones = zip(*transitions)

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        non_final_mask = torch.BoolTensor([s is not None for s in next_states])
        non_final_next_states = torch.FloatTensor(
            np.array([s for s in next_states if s is not None])
        ) if non_final_mask.sum() > 0 else torch.FloatTensor()

        # --- Current Q-values: Q(s, a) ---
        q_values = self.q_network(states_t).gather(1, actions_t)

        # --- Target Q-values ---
        next_q_values = torch.zeros(len(transitions), 1)

        if non_final_mask.sum() > 0:
            with torch.no_grad():
                if self.config.double_dqn:
                    # Double DQN: online network selects action, target evaluates
                    # best_a = argmax_a Q_online(s', a)
                    online_q = self.q_network(non_final_next_states)
                    best_actions = online_q.argmax(dim=1, keepdim=True)

                    # Q_target = Q_target(s', best_a)
                    target_q = self.target_network(non_final_next_states)
                    next_q_values[non_final_mask] = target_q.gather(1, best_actions)
                else:
                    # Vanilla DQN: target network selects and evaluates
                    next_q_values[non_final_mask] = (
                        self.target_network(non_final_next_states)
                        .max(1)[0]
                        .unsqueeze(1)
                    )

        target_q_values = rewards_t + self.config.gamma * next_q_values

        # --- Compute TD errors (for PER priority update) ---
        td_errors = (q_values - target_q_values).detach().abs().squeeze(1)

        # --- Compute loss ---
        # Element-wise Huber loss
        element_loss = nn.functional.smooth_l1_loss(
            q_values, target_q_values, reduction="none"
        )

        if self._use_per and is_weights_t is not None:
            # Weight loss by importance sampling weights to correct PER bias
            weighted_loss = (element_loss * is_weights_t).mean()
        else:
            weighted_loss = element_loss.mean()

        # --- Gradient descent ---
        self.optimizer.zero_grad()
        weighted_loss.backward()

        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip)

        self.optimizer.step()

        # --- Update PER priorities with new TD errors ---
        if self._use_per and per_indices is not None:
            self.per_buffer.update_priorities(per_indices, td_errors.numpy())

        # --- Soft target update (Polyak averaging) ---
        if self.config.use_soft_update:
            self._soft_update_target()

        self.steps += 1

        return weighted_loss.item()

    def _soft_update_target(self) -> None:
        """Soft update target network: θ_target = τ·θ_online + (1-τ)·θ_target."""
        tau = self.config.tau
        for target_param, online_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

    def update_target_network(self) -> None:
        """Hard copy weights from Q-network to target network (fallback)."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )

    def save(self, path: Path | str | None = None) -> None:
        """Save Q-network and training state."""
        save_path = Path(path) if path else Path(self.config.model_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "episodes": self.episodes,
            "config": self.config,
        }
        torch.save(checkpoint, save_path)
        logger.info("Saved DQN model to %s", save_path)

        # Save replay buffer separately (can be large)
        replay_path = Path(self.config.replay_save_path)
        replay_path.parent.mkdir(parents=True, exist_ok=True)

        if self._use_per:
            # Save PER buffer state
            with open(replay_path, "wb") as f:
                pickle.dump({
                    "type": "per",
                    "data": [
                        self.per_buffer.tree.data[i]
                        for i in range(self.per_buffer.tree.size)
                    ],
                    "priorities": [
                        float(self.per_buffer.tree.tree[i + self.per_buffer.tree.capacity - 1])
                        for i in range(self.per_buffer.tree.size)
                    ],
                }, f)
        else:
            with open(replay_path, "wb") as f:
                pickle.dump({"type": "uniform", "data": list(self.replay_buffer)}, f)

        logger.info("Saved replay buffer to %s", replay_path)

    def load(self, path: Path | str | None = None) -> None:
        """Load Q-network and training state."""
        load_path = Path(path) if path else Path(self.config.model_save_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")

        checkpoint = torch.load(load_path, weights_only=False)

        self.q_network.load_state_dict(checkpoint["q_network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self.episodes = checkpoint.get("episodes", 0)

        logger.info("Loaded DQN model from %s", load_path)

        # Load replay buffer if available
        replay_path = Path(self.config.replay_save_path)
        if replay_path.exists():
            try:
                with open(replay_path, "rb") as f:
                    replay_data = pickle.load(f)

                if isinstance(replay_data, dict):
                    if replay_data["type"] == "per" and self._use_per:
                        for transition, priority in zip(
                            replay_data["data"], replay_data["priorities"]
                        ):
                            if transition is not None:
                                self.per_buffer.add(transition, priority=priority)
                    elif replay_data["type"] == "uniform" and not self._use_per:
                        self.replay_buffer.clear()
                        self.replay_buffer.extend(replay_data["data"])
                    else:
                        logger.warning(
                            "Replay buffer type mismatch (saved=%s, current=%s), skipping",
                            replay_data["type"],
                            "per" if self._use_per else "uniform",
                        )
                elif isinstance(replay_data, list):
                    # Legacy format: plain list of transitions
                    if self._use_per:
                        for transition in replay_data:
                            self.per_buffer.add(transition, priority=None)
                    else:
                        self.replay_buffer.clear()
                        self.replay_buffer.extend(replay_data)

                logger.info("Loaded replay buffer (%d transitions)", self.buffer_size)
            except Exception:
                logger.debug("Failed to load replay buffer", exc_info=True)


# ---------------------------------------------------------------------------
# DQN Strategy Selector (wrapper matching RLStrategySelector API)
# ---------------------------------------------------------------------------

class DQNStrategySelector:
    """DQN-based strategy selector with same API as RLStrategySelector.

    Wraps DQNAgent with:
    - State conversion from CampaignSnapshot → RLState → numpy array
    - Action mapping to strategy backend names
    - Online learning interface (store + train per step)
    - Q-value confidence estimation for safety fallback
    """

    def __init__(self, config: DQNConfig | None = None):
        from app.services.rl_strategy_selector import RLState, ACTIONS

        if config is None:
            config = DQNConfig()

        self.config = config
        self.ACTIONS = ACTIONS

        # State dimension from RLState (15 features)
        self.state_dim = 15
        self.action_dim = len(ACTIONS)

        # Create DQN agent
        self.agent = DQNAgent(self.state_dim, self.action_dim, config)

        logger.info(
            "Initialized DQN selector: state_dim=%d, action_dim=%d, "
            "dueling=%s, double=%s, per=%s",
            self.state_dim, self.action_dim,
            config.dueling, config.double_dqn, config.use_per,
        )

    def select_action(
        self,
        snapshot: Any,
        diagnostics: Any,
        explore: bool = True,
    ) -> tuple[int, str]:
        """Select action using DQN policy.

        Returns:
            (action_id, backend_name)
        """
        from app.services.rl_strategy_selector import RLState

        state = RLState.from_snapshot(snapshot, diagnostics)
        state_array = state.to_array()

        action_id = self.agent.select_action(state_array, explore=explore)
        backend_name = self.ACTIONS[action_id]

        return action_id, backend_name

    def get_confidence(
        self,
        snapshot: Any,
        diagnostics: Any,
    ) -> float:
        """Estimate confidence in current policy for this state.

        Returns a confidence score in [0, 1] based on:
        - Q-value spread (high spread = high confidence)
        - Training maturity (more steps = more confident)
        - Epsilon level (lower epsilon = more exploiting)

        Used by StrategyRouter to decide RL vs rule-based fallback.
        """
        from app.services.rl_strategy_selector import RLState

        state = RLState.from_snapshot(snapshot, diagnostics)
        state_array = state.to_array()
        q_values = self.agent.get_q_values(state_array)

        # Q-value spread: normalized difference between best and worst
        q_range = q_values.max() - q_values.min()
        q_spread_conf = min(1.0, q_range / 2.0)  # Normalize to [0, 1]

        # Training maturity: sigmoid-like ramp-up
        maturity = min(1.0, self.agent.steps / 500.0)

        # Exploitation level: inverse of epsilon
        exploit_conf = 1.0 - self.agent.epsilon

        # Weighted combination
        confidence = 0.4 * q_spread_conf + 0.3 * maturity + 0.3 * exploit_conf
        return float(max(0.0, min(1.0, confidence)))

    def learn_from_experience(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any | None,
        done: bool,
    ) -> float | None:
        """Store transition and perform training step.

        Returns:
            Loss value or None if not enough data to train
        """
        # Convert states to arrays
        state_array = state.to_array() if hasattr(state, "to_array") else state
        next_state_array = (
            next_state.to_array() if next_state and hasattr(next_state, "to_array") else next_state
        )

        # Store transition
        self.agent.store_transition(state_array, action, reward, next_state_array, done)

        # Train if enough data
        loss = self.agent.train_step()

        if done:
            self.agent.episodes += 1

            # Hard target update (fallback if soft update disabled)
            if not self.config.use_soft_update:
                if self.agent.episodes % self.config.target_update_freq == 0:
                    self.agent.update_target_network()

            # Decay epsilon
            self.agent.decay_epsilon()

        return loss

    def save(self, path: Path | str | None = None) -> None:
        """Save DQN model."""
        self.agent.save(path)

    def load(self, path: Path | str | None = None) -> None:
        """Load DQN model."""
        self.agent.load(path)

"""Deep Q-Network (DQN) strategy selector with PyTorch.

Advanced RL implementation using neural network function approximation
to overcome tabular Q-learning limitations.

Key improvements over tabular Q-learning:
- No state discretization (handles continuous states directly)
- Better generalization across similar states
- Experience replay for stable learning
- Target network for training stability
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

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch not available. DQN selector will not work. Install with: pip install torch")

logger = logging.getLogger(__name__)


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

    # Learning
    learning_rate: float = 0.001
    gamma: float = 0.95  # Discount factor
    epsilon: float = 1.0  # Initial exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01

    # Training
    batch_size: int = 32
    replay_capacity: int = 10000
    target_update_freq: int = 10  # Update target network every N episodes
    grad_clip: float = 1.0  # Gradient clipping threshold

    # Persistence
    model_save_path: str = "models/dqn_selector.pth"
    replay_save_path: str = "models/dqn_replay.pkl"


# ---------------------------------------------------------------------------
# Q-Network Architecture
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Deep Q-Network: maps state → Q-values for each action."""

    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig):
        super().__init__()
        self.config = config

        # Build network layers
        layers = []
        input_dim = state_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))

            # Activation
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "tanh":
                layers.append(nn.Tanh())
            elif config.activation == "leaky_relu":
                layers.append(nn.LeakyReLU())

            # Dropout
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))

            input_dim = hidden_dim

        # Output layer: Q-values for each action
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: state → Q-values."""
        return self.network(state)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """Deep Q-Network agent with experience replay and target network."""

    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DQN. Install with: pip install torch")

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q-network (online)
        self.q_network = QNetwork(state_dim, action_dim, config)

        # Target network (for stability)
        self.target_network = QNetwork(state_dim, action_dim, config)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Always in eval mode

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

        # Experience replay
        self.replay_buffer = deque(maxlen=config.replay_capacity)

        # Training state
        self.epsilon = config.epsilon
        self.steps = 0
        self.episodes = 0

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """ε-greedy action selection."""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        # Greedy action from Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray | None,
        done: bool,
    ) -> None:
        """Add transition to replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self) -> float | None:
        """Sample batch and perform one gradient descent step.

        Returns:
            Average TD error (loss) or None if insufficient data
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        # Sample mini-batch
        batch = random.sample(self.replay_buffer, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        non_final_mask = torch.BoolTensor([s is not None for s in next_states])
        non_final_next_states = torch.FloatTensor(
            np.array([s for s in next_states if s is not None])
        )

        # Current Q-values: Q(s, a)
        q_values = self.q_network(states_t).gather(1, actions_t)

        # Target Q-values: r + γ * max_a' Q_target(s', a')
        next_q_values = torch.zeros(self.config.batch_size, 1)
        with torch.no_grad():
            if non_final_mask.sum() > 0:
                next_q_values[non_final_mask] = (
                    self.target_network(non_final_next_states).max(1)[0].unsqueeze(1)
                )

        target_q_values = rewards_t + self.config.gamma * next_q_values

        # Compute loss (Huber loss for stability)
        loss = nn.SmoothL1Loss()(q_values, target_q_values)

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip)

        self.optimizer.step()

        self.steps += 1

        return loss.item()

    def update_target_network(self) -> None:
        """Copy weights from Q-network to target network."""
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

        torch.save({
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "episodes": self.episodes,
            "config": self.config,
        }, save_path)

        logger.info(f"Saved DQN model to {save_path}")

        # Save replay buffer separately (can be large)
        replay_path = Path(self.config.replay_save_path)
        with open(replay_path, "wb") as f:
            pickle.dump(list(self.replay_buffer), f)
        logger.info(f"Saved replay buffer to {replay_path}")

    def load(self, path: Path | str | None = None) -> None:
        """Load Q-network and training state."""
        load_path = Path(path) if path else Path(self.config.model_save_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")

        checkpoint = torch.load(load_path)

        self.q_network.load_state_dict(checkpoint["q_network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self.episodes = checkpoint.get("episodes", 0)

        logger.info(f"Loaded DQN model from {load_path}")

        # Load replay buffer if available
        replay_path = Path(self.config.replay_save_path)
        if replay_path.exists():
            with open(replay_path, "rb") as f:
                transitions = pickle.load(f)
                self.replay_buffer.clear()
                self.replay_buffer.extend(transitions)
            logger.info(f"Loaded replay buffer ({len(self.replay_buffer)} transitions)")


# ---------------------------------------------------------------------------
# DQN Strategy Selector (wrapper matching RLStrategySelector API)
# ---------------------------------------------------------------------------

class DQNStrategySelector:
    """DQN-based strategy selector with same API as RLStrategySelector."""

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

        logger.info(f"Initialized DQN selector: state_dim={self.state_dim}, action_dim={self.action_dim}")
        logger.info(f"Network architecture: {config.hidden_dims}")

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

        # Convert to RLState
        state = RLState.from_snapshot(snapshot, diagnostics)
        state_array = state.to_array()

        # Select action
        action_id = self.agent.select_action(state_array, explore=explore)
        backend_name = self.ACTIONS[action_id]

        return action_id, backend_name

    def learn_from_experience(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any | None,
        done: bool,
    ) -> None:
        """Store transition and perform training step."""
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

            # Update target network periodically
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

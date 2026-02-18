"""Neural Reward Model — learns reward function from campaign outcomes.

Instead of relying solely on hand-crafted reward shaping (rl_reward.py),
this module trains a small neural network to predict rewards that better
correlate with actual campaign success.

Architecture:
    Input: [state(15) + action_onehot(4) + next_state(15)] = 34 dims
    Hidden: 64 → 32
    Output: scalar reward prediction

Training:
    After each campaign, all transitions are hindsight-relabeled:
    - Successful campaigns get positive reward scaling
    - Failed campaigns get negative reward scaling
    - The model learns to assign higher rewards to (state, action, next_state)
      transitions that lead to good outcomes

Usage:
    The predicted reward is blended with the handcrafted reward:
        final_reward = (1 - λ) * handcrafted + λ * learned
    where λ gradually increases as more campaigns are seen.

References:
    Inspired by reward learning in RLHF and outcome-conditioned reward models.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["RewardModel", "RewardModelTrainer", "RewardModelConfig"]

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
# Config
# ---------------------------------------------------------------------------

@dataclass
class RewardModelConfig:
    """Configuration for the neural reward model."""

    # Network architecture
    state_dim: int = 15
    action_dim: int = 4
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])

    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    n_epochs_per_campaign: int = 5  # Training epochs after each campaign
    max_buffer_size: int = 5000  # Max transitions to keep

    # Blending (how much to trust learned vs handcrafted reward)
    lambda_initial: float = 0.0  # Start fully trusting handcrafted
    lambda_max: float = 0.5  # Never exceed 50% learned reward
    lambda_warmup_campaigns: int = 50  # Campaigns to reach lambda_max

    # Persistence
    model_save_path: str = "models/reward_model.pth"

    # Hindsight relabeling
    success_bonus: float = 1.0  # Extra reward for transitions in successful campaigns
    failure_penalty: float = -0.5  # Penalty for transitions in failed campaigns


# ---------------------------------------------------------------------------
# Reward Model Network
# ---------------------------------------------------------------------------

class RewardNetwork(nn.Module):
    """Neural network that predicts reward from (state, action, next_state).

    Input: [state(15) + action_onehot(4) + next_state(15)] = 34
    Output: scalar reward
    """

    def __init__(self, config: RewardModelConfig):
        super().__init__()
        input_dim = config.state_dim * 2 + config.action_dim  # state + action + next_state

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (state, action, next_state) → reward."""
        return self.network(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Transition Buffer for Training
# ---------------------------------------------------------------------------

@dataclass
class LabeledTransition:
    """A transition labeled with hindsight reward."""
    state: Any  # np.ndarray (15,)
    action: int
    next_state: Any  # np.ndarray (15,) or None
    handcrafted_reward: float
    outcome_label: float  # hindsight: success_bonus or failure_penalty
    target_reward: float  # handcrafted_reward + outcome_label


# ---------------------------------------------------------------------------
# Reward Model (high-level API)
# ---------------------------------------------------------------------------

class RewardModel:
    """Neural reward model with predict and blend capabilities.

    Usage:
        model = RewardModel()
        predicted = model.predict_reward(state, action, next_state)
        blended = model.blend_reward(handcrafted_reward, state, action, next_state)
    """

    def __init__(self, config: RewardModelConfig | None = None):
        if config is None:
            config = RewardModelConfig()
        self.config = config
        self._campaigns_trained: int = 0

        if not TORCH_AVAILABLE:
            self._network = None
            logger.warning("PyTorch not available. Neural reward model disabled.")
            return

        self._network = RewardNetwork(config)

        # Try loading existing model
        try:
            path = Path(config.model_save_path)
            if path.exists():
                checkpoint = torch.load(path, weights_only=False)
                self._network.load_state_dict(checkpoint["model_state"])
                self._campaigns_trained = checkpoint.get("campaigns_trained", 0)
                logger.info(
                    "Loaded reward model from %s (trained on %d campaigns)",
                    path, self._campaigns_trained,
                )
        except Exception:
            logger.debug("No existing reward model found, starting fresh", exc_info=True)

    @property
    def lambda_blend(self) -> float:
        """Current blending factor: how much to trust the learned reward."""
        if self._campaigns_trained < 1:
            return self.config.lambda_initial

        progress = min(1.0, self._campaigns_trained / self.config.lambda_warmup_campaigns)
        return self.config.lambda_initial + progress * (
            self.config.lambda_max - self.config.lambda_initial
        )

    def predict_reward(
        self,
        state: Any,
        action: int,
        next_state: Any | None,
    ) -> float:
        """Predict reward for a (state, action, next_state) transition.

        Returns 0.0 if model is not available or not trained.
        """
        if self._network is None or self._campaigns_trained < 1:
            return 0.0

        try:
            x = self._build_input(state, action, next_state)
            with torch.no_grad():
                return float(self._network(x))
        except Exception:
            logger.debug("Reward model prediction failed", exc_info=True)
            return 0.0

    def blend_reward(
        self,
        handcrafted_reward: float,
        state: Any,
        action: int,
        next_state: Any | None,
    ) -> float:
        """Blend handcrafted and learned rewards.

        final = (1 - λ) * handcrafted + λ * learned
        """
        lam = self.lambda_blend
        if lam <= 0.0:
            return handcrafted_reward

        learned = self.predict_reward(state, action, next_state)
        return (1 - lam) * handcrafted_reward + lam * learned

    def _build_input(self, state: Any, action: int, next_state: Any | None) -> torch.Tensor:
        """Build input tensor: [state, action_onehot, next_state]."""
        state_arr = state.to_array() if hasattr(state, "to_array") else np.array(state, dtype=np.float32)

        if next_state is not None:
            next_arr = next_state.to_array() if hasattr(next_state, "to_array") else np.array(next_state, dtype=np.float32)
        else:
            next_arr = np.zeros(self.config.state_dim, dtype=np.float32)

        # Action one-hot
        action_oh = np.zeros(self.config.action_dim, dtype=np.float32)
        action_oh[action] = 1.0

        x = np.concatenate([state_arr, action_oh, next_arr])
        return torch.FloatTensor(x).unsqueeze(0)

    def save(self, path: str | None = None) -> None:
        """Save model checkpoint."""
        if self._network is None:
            return
        save_path = Path(path or self.config.model_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self._network.state_dict(),
            "campaigns_trained": self._campaigns_trained,
            "config": self.config,
        }, save_path)
        logger.info("Saved reward model to %s", save_path)


# ---------------------------------------------------------------------------
# Reward Model Trainer
# ---------------------------------------------------------------------------

class RewardModelTrainer:
    """Trains the neural reward model from campaign traces.

    After each campaign completes:
    1. Collect all transitions from the campaign
    2. Hindsight-relabel with outcome quality
    3. Train for N epochs on the accumulated buffer
    4. Save checkpoint
    """

    def __init__(
        self,
        model: RewardModel,
        config: RewardModelConfig | None = None,
    ):
        self.model = model
        self.config = config or model.config
        self._buffer: list[LabeledTransition] = []

    def add_campaign_transitions(
        self,
        transitions: list[tuple[Any, int, float, Any | None, bool]],
        target_reached: bool,
        converged: bool,
    ) -> None:
        """Add transitions from a completed campaign with hindsight labels.

        Args:
            transitions: List of (state, action, reward, next_state, done)
            target_reached: Whether the campaign reached its target
            converged: Whether the campaign converged
        """
        # Compute outcome label
        if target_reached:
            outcome_label = self.config.success_bonus
        elif converged:
            outcome_label = self.config.success_bonus * 0.5
        else:
            outcome_label = self.config.failure_penalty

        for state, action, reward, next_state, done in transitions:
            labeled = LabeledTransition(
                state=state,
                action=action,
                next_state=next_state,
                handcrafted_reward=reward,
                outcome_label=outcome_label,
                target_reward=reward + outcome_label,
            )
            self._buffer.append(labeled)

        # Trim buffer if too large
        if len(self._buffer) > self.config.max_buffer_size:
            self._buffer = self._buffer[-self.config.max_buffer_size:]

        logger.info(
            "Added %d transitions (outcome_label=%.2f), buffer size=%d",
            len(transitions), outcome_label, len(self._buffer),
        )

    def train(self) -> float:
        """Train the reward model on accumulated buffer.

        Returns:
            Average training loss
        """
        if self.model._network is None:
            return 0.0

        if len(self._buffer) < self.config.batch_size:
            logger.debug("Not enough data to train reward model (%d < %d)",
                         len(self._buffer), self.config.batch_size)
            return 0.0

        import random

        optimizer = optim.Adam(
            self.model._network.parameters(),
            lr=self.config.learning_rate,
        )

        total_loss = 0.0
        n_batches = 0

        for epoch in range(self.config.n_epochs_per_campaign):
            random.shuffle(self._buffer)

            for i in range(0, len(self._buffer) - self.config.batch_size + 1, self.config.batch_size):
                batch = self._buffer[i:i + self.config.batch_size]

                # Build input tensors
                inputs = []
                targets = []

                for t in batch:
                    state_arr = t.state.to_array() if hasattr(t.state, "to_array") else np.array(t.state, dtype=np.float32)
                    if t.next_state is not None:
                        next_arr = t.next_state.to_array() if hasattr(t.next_state, "to_array") else np.array(t.next_state, dtype=np.float32)
                    else:
                        next_arr = np.zeros(self.config.state_dim, dtype=np.float32)

                    action_oh = np.zeros(self.config.action_dim, dtype=np.float32)
                    action_oh[t.action] = 1.0

                    x = np.concatenate([state_arr, action_oh, next_arr])
                    inputs.append(x)
                    targets.append(t.target_reward)

                inputs_t = torch.FloatTensor(np.array(inputs))
                targets_t = torch.FloatTensor(targets)

                # Forward pass
                predictions = self.model._network(inputs_t)
                loss = nn.functional.mse_loss(predictions, targets_t)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model._network.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        self.model._campaigns_trained += 1

        # Save
        self.model.save()

        logger.info(
            "Trained reward model: avg_loss=%.4f, campaigns_trained=%d, λ=%.3f",
            avg_loss, self.model._campaigns_trained, self.model.lambda_blend,
        )

        return avg_loss

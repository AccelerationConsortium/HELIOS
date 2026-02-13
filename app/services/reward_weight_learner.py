"""Meta-learning for RL reward weights (α, β, γ, δ).

Instead of static reward hyperparameters, this module learns optimal
reward weight configurations from campaign outcomes. After each campaign,
it evaluates whether the current weights led to good outcomes and adjusts
using finite-difference gradient estimation.

The intuition:
- If high total_reward → good_outcome (target reached, fast convergence),
  the weights are good — keep them.
- If high total_reward → bad_outcome, the reward signal is misleading —
  adjust weights so reward better correlates with true performance.

Learning algorithm:
1. Compute total reward under current weights for the campaign trace
2. Compute finite-difference gradient: ∂quality/∂weight for each weight
3. Update: weight += lr * (outcome_quality - baseline) * gradient
4. EMA smoothing across campaigns for stability
5. Clip to valid ranges

Persistence: saves learned weights to JSON for cross-session learning.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["RewardWeightLearner", "LearnedWeights"]


# ---------------------------------------------------------------------------
# Learned Weights State
# ---------------------------------------------------------------------------

@dataclass
class LearnedWeights:
    """Current state of learned reward weights."""

    # Core weights (map to RewardConfig)
    alpha: float = 1.0  # KPI improvement weight
    beta: float = 0.01  # Cost penalty weight
    gamma: float = 0.5  # Convergence bonus weight
    delta: float = 0.1  # Exploration bonus weight

    # Meta-learning state
    n_updates: int = 0
    ema_quality: float = 0.0  # Exponential moving average of outcome quality
    ema_reward: float = 0.0  # EMA of total reward

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearnedWeights:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Campaign Trace (lightweight summary for weight learning)
# ---------------------------------------------------------------------------

@dataclass
class CampaignOutcome:
    """Summary of a campaign for reward weight learning."""

    campaign_id: str
    n_rounds: int
    final_kpi: float | None
    best_kpi: float | None
    target_value: float | None
    direction: str  # "maximize" | "minimize"
    target_reached: bool
    converged: bool
    reward_trace: list[float] = field(default_factory=list)  # per-round rewards


# ---------------------------------------------------------------------------
# Reward Weight Learner
# ---------------------------------------------------------------------------

@dataclass
class WeightLearnerConfig:
    """Configuration for the reward weight meta-learner."""

    # Learning
    learning_rate: float = 0.01
    ema_decay: float = 0.9  # EMA smoothing factor
    perturbation: float = 0.05  # Finite-difference step size

    # Weight bounds (prevent degenerate reward signals)
    alpha_range: tuple[float, float] = (0.1, 5.0)
    beta_range: tuple[float, float] = (0.001, 0.1)
    gamma_range: tuple[float, float] = (0.05, 2.0)
    delta_range: tuple[float, float] = (0.01, 0.5)

    # Persistence
    save_path: str = "models/reward_weights.json"

    # Warmup (don't adjust weights until we have enough campaigns)
    min_campaigns_before_learning: int = 3


class RewardWeightLearner:
    """Learns optimal reward weights from campaign outcomes.

    Usage:
        learner = RewardWeightLearner()

        # After each campaign:
        outcome = CampaignOutcome(...)
        new_config = learner.update_weights(outcome)

        # Use new_config for next campaign's reward computation
    """

    def __init__(self, config: WeightLearnerConfig | None = None):
        if config is None:
            config = WeightLearnerConfig()
        self.config = config
        self.weights = LearnedWeights()

        # Try to load existing weights
        self._load()

        logger.info(
            "RewardWeightLearner initialized: α=%.3f β=%.4f γ=%.3f δ=%.3f (updates=%d)",
            self.weights.alpha, self.weights.beta,
            self.weights.gamma, self.weights.delta,
            self.weights.n_updates,
        )

    def get_reward_config(self) -> Any:
        """Get a RewardConfig with current learned weights."""
        from app.services.rl_reward import RewardConfig
        return RewardConfig(
            alpha=self.weights.alpha,
            beta=self.weights.beta,
            gamma=self.weights.gamma,
            delta=self.weights.delta,
        )

    def update_weights(self, outcome: CampaignOutcome) -> Any:
        """Update weights based on campaign outcome.

        Args:
            outcome: Summary of completed campaign

        Returns:
            Updated RewardConfig for next campaign
        """
        # Compute outcome quality score
        quality = self._compute_outcome_quality(outcome)

        # Compute total reward under current weights
        total_reward = sum(outcome.reward_trace) if outcome.reward_trace else 0.0

        # Update EMA baselines
        decay = self.config.ema_decay
        self.weights.ema_quality = decay * self.weights.ema_quality + (1 - decay) * quality
        self.weights.ema_reward = decay * self.weights.ema_reward + (1 - decay) * total_reward

        # Skip actual weight update during warmup
        self.weights.n_updates += 1
        if self.weights.n_updates < self.config.min_campaigns_before_learning:
            logger.info(
                "Warmup phase (%d/%d), collecting baselines...",
                self.weights.n_updates, self.config.min_campaigns_before_learning,
            )
            self._save()
            return self.get_reward_config()

        # Compute advantage: how much better/worse than baseline
        advantage = quality - self.weights.ema_quality

        # Finite-difference gradient estimation for each weight
        # Perturb each weight slightly, see how total reward changes,
        # then adjust in direction that correlates quality with reward
        weight_names = ["alpha", "beta", "gamma", "delta"]
        weight_ranges = {
            "alpha": self.config.alpha_range,
            "beta": self.config.beta_range,
            "gamma": self.config.gamma_range,
            "delta": self.config.delta_range,
        }

        for name in weight_names:
            current_val = getattr(self.weights, name)
            eps = self.config.perturbation * current_val  # Relative perturbation

            if eps < 1e-8:
                continue

            # Estimate gradient: if reward was high but quality was low,
            # we want to decrease this weight (it's misleading)
            # If reward was high and quality was high, keep/increase
            reward_deviation = total_reward - self.weights.ema_reward

            # Gradient approximation: adjust weight in direction of
            # (advantage * sign(correlation between reward and quality))
            if abs(reward_deviation) > 1e-8:
                correlation_sign = math.copysign(1.0, advantage * reward_deviation)
            else:
                correlation_sign = math.copysign(1.0, advantage)

            # Update
            delta = self.config.learning_rate * advantage * correlation_sign * eps
            new_val = current_val + delta

            # Clip to valid range
            lo, hi = weight_ranges[name]
            new_val = max(lo, min(hi, new_val))

            setattr(self.weights, name, new_val)

        logger.info(
            "Updated reward weights: α=%.3f β=%.4f γ=%.3f δ=%.3f "
            "(quality=%.3f advantage=%.3f)",
            self.weights.alpha, self.weights.beta,
            self.weights.gamma, self.weights.delta,
            quality, advantage,
        )

        self._save()
        return self.get_reward_config()

    def _compute_outcome_quality(self, outcome: CampaignOutcome) -> float:
        """Compute a scalar quality score for a campaign outcome.

        Quality = weighted sum of:
        - Target reached (binary, high weight)
        - Convergence (binary, moderate weight)
        - Efficiency (fewer rounds = better)
        - KPI improvement ratio
        """
        quality = 0.0

        # Target reached: big bonus
        if outcome.target_reached:
            quality += 1.0

        # Converged: moderate bonus
        if outcome.converged:
            quality += 0.5

        # Efficiency: bonus for using fewer rounds
        if outcome.n_rounds > 0:
            efficiency = 1.0 - (outcome.n_rounds / max(outcome.n_rounds * 2, 24))
            quality += 0.3 * max(0.0, efficiency)

        # KPI quality: normalize by target if available
        if outcome.best_kpi is not None and outcome.target_value is not None:
            if outcome.direction == "maximize":
                kpi_ratio = outcome.best_kpi / max(abs(outcome.target_value), 1e-8)
            else:
                kpi_ratio = outcome.target_value / max(abs(outcome.best_kpi), 1e-8)
            quality += 0.2 * min(1.0, max(0.0, kpi_ratio))

        return quality

    def _save(self) -> None:
        """Save weights to JSON."""
        try:
            path = Path(self.config.save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.weights.to_dict(), f, indent=2)
        except Exception:
            logger.debug("Failed to save reward weights", exc_info=True)

    def _load(self) -> None:
        """Load weights from JSON if available."""
        try:
            path = Path(self.config.save_path)
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                self.weights = LearnedWeights.from_dict(data)
                logger.info("Loaded reward weights from %s", path)
        except Exception:
            logger.debug("Failed to load reward weights, using defaults", exc_info=True)

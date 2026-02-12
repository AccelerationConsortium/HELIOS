"""RL-Based Strategy Selector — learns optimal策略选择 from experience.

Upgrades the hand-crafted utility-based selector (v3) to a reinforcement
learning agent that learns from historical campaigns.

Architecture:
    State: DiagnosticSignals (15+ features)
    Action: Backend selection (explore/exploit/refine/stabilize)
    Reward: KPI improvement / cost (rounds consumed)

Training:
    - Offline: Pre-train on historical campaigns
    - Online: Fine-tune during live campaigns with ε-greedy exploration

Implementation:
    - Phase 1: Simple Q-learning baseline
    - Phase 2: Deep Q-Network (DQN) with experience replay
    - Phase 3: Policy gradient methods (PPO)
    - Phase 4: Meta-learning across campaign types

Current version: Phase 1 (Q-learning baseline)
"""
from __future__ import annotations

import json
import logging
import math
import pickle
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.services.strategy_selector import (
    CampaignSnapshot,
    DiagnosticSignals,
    StrategyDecision,
    compute_diagnostics,
    select_strategy as rule_based_select_strategy,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RLStrategySelector",
    "RLState",
    "RLConfig",
    "ExperienceReplay",
    "train_rl_selector_offline",
    "select_strategy_rl",
]


# ---------------------------------------------------------------------------
# RL State Representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RLState:
    """State representation for RL agent.

    Extracted from DiagnosticSignals + CampaignSnapshot.
    Features are normalized to [0, 1] for stable learning.
    """

    # Campaign context (4 features)
    progress: float  # round / max_rounds
    n_obs_ratio: float  # n_observations / expected_total
    has_categorical: float  # 0 or 1
    has_log_scale: float  # 0 or 1

    # Epistemic signals (2 features)
    space_coverage: float  # 0-1
    model_uncertainty: float  # normalized; 0 if None

    # Aleatoric signals (3 features)
    noise_ratio: float  # 0-1; 0 if None
    replicate_need_score: float  # 0-1; 0 if None
    batch_kpi_cv: float  # 0-1; 0 if None

    # Saturation signals (4 features)
    improvement_velocity: float  # 0-1; 0 if None
    ei_decay_proxy: float  # 0-1; 0 if None
    convergence_confidence: float  # 0-1
    convergence_plateau: float  # 1 if plateau, else 0

    # Landscape signals (2 features)
    local_smoothness: float  # 0-1; 0 if None
    batch_param_spread: float  # 0-1; 0 if None

    # Total: 16 features

    @classmethod
    def from_snapshot(
        cls,
        snapshot: CampaignSnapshot,
        diagnostics: DiagnosticSignals,
    ) -> RLState:
        """Extract state from campaign snapshot + diagnostics."""
        # Campaign context
        progress = snapshot.round_number / snapshot.max_rounds
        expected_total = snapshot.max_rounds * 10  # assume 10 candidates/round
        n_obs_ratio = min(1.0, snapshot.n_observations / expected_total)

        # Convergence
        convergence_plateau = 1.0 if diagnostics.convergence_status == "plateau" else 0.0

        # Helper: clamp to [0, 1] and handle None
        def norm(x: float | None, default: float = 0.0) -> float:
            if x is None:
                return default
            return max(0.0, min(1.0, x))

        return cls(
            progress=progress,
            n_obs_ratio=n_obs_ratio,
            has_categorical=1.0 if snapshot.has_categorical else 0.0,
            has_log_scale=1.0 if snapshot.has_log_scale else 0.0,
            space_coverage=diagnostics.space_coverage,
            model_uncertainty=norm(diagnostics.model_uncertainty, 0.0),
            noise_ratio=norm(diagnostics.noise_ratio, 0.0),
            replicate_need_score=norm(diagnostics.replicate_need_score, 0.0),
            batch_kpi_cv=norm(diagnostics.batch_kpi_cv, 0.0),
            improvement_velocity=norm(diagnostics.improvement_velocity, 0.0),
            ei_decay_proxy=norm(diagnostics.ei_decay_proxy, 0.0),
            convergence_confidence=diagnostics.convergence_confidence,
            convergence_plateau=convergence_plateau,
            local_smoothness=norm(diagnostics.local_smoothness, 0.0),
            batch_param_spread=norm(diagnostics.batch_param_spread, 0.0),
        )

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for neural network input."""
        return np.array([
            self.progress,
            self.n_obs_ratio,
            self.has_categorical,
            self.has_log_scale,
            self.space_coverage,
            self.model_uncertainty,
            self.noise_ratio,
            self.replicate_need_score,
            self.batch_kpi_cv,
            self.improvement_velocity,
            self.ei_decay_proxy,
            self.convergence_confidence,
            self.convergence_plateau,
            self.local_smoothness,
            self.batch_param_spread,
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

# Action space: 4 discrete actions mapping to backend strategies
ACTIONS = {
    0: "explore",     # lhs, random
    1: "exploit",     # bayesian, optuna_tpe
    2: "refine",      # optuna_cmaes, scipy_de
    3: "stabilize",   # replicate best points
}

ACTION_TO_BACKEND = {
    "explore": "lhs",
    "exploit": "built_in",  # Bayesian
    "refine": "optuna_cmaes",
    "stabilize": "built_in",  # will trigger StabilizeSpec
}


# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    """Single transition (s, a, r, s', done)."""
    state: RLState
    action: int  # 0-3
    reward: float
    next_state: RLState | None
    done: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperienceReplay:
    """Replay buffer for offline training and online learning."""

    def __init__(self, capacity: int = 10000):
        self.buffer: deque[Experience] = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, exp: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample random batch."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: Path) -> None:
        """Save buffer to disk."""
        with open(path, "wb") as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path: Path) -> None:
        """Load buffer from disk."""
        with open(path, "rb") as f:
            experiences = pickle.load(f)
            self.buffer.clear()
            self.buffer.extend(experiences)


# ---------------------------------------------------------------------------
# RL Config
# ---------------------------------------------------------------------------

@dataclass
class RLConfig:
    """Hyperparameters for RL agent."""

    # Q-learning
    learning_rate: float = 0.01
    gamma: float = 0.95  # discount factor
    epsilon: float = 0.1  # exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01

    # State discretization
    n_bins: int = 3  # Number of bins per feature (2=binary, 3=low/mid/high, 5=fine-grained)
    adaptive_binning: bool = True  # Use feature-specific binning strategies

    # Experience replay
    replay_capacity: int = 10000
    batch_size: int = 32
    train_frequency: int = 4  # train every N steps

    # Reward shaping
    kpi_improvement_scale: float = 10.0  # scale KPI improvement to ~[0, 1]
    round_cost: float = -0.01  # penalty per round
    convergence_bonus: float = 0.5  # bonus for reaching convergence
    failure_penalty: float = -0.1  # penalty for QC failures

    # Model persistence
    model_save_path: str = "models/rl_strategy_selector.pkl"
    replay_save_path: str = "models/rl_replay_buffer.pkl"

    # Training
    n_epochs: int = 100
    target_update_frequency: int = 10  # for DQN (future)


# ---------------------------------------------------------------------------
# Q-Learning Agent (Phase 1: Simple Baseline)
# ---------------------------------------------------------------------------

class QLearningAgent:
    """Tabular Q-learning with state discretization.

    Uses simple lookup table Q(s, a) with ε-greedy exploration.
    Good baseline to establish feasibility before deep RL.
    """

    def __init__(self, config: RLConfig):
        self.config = config
        # Q-table: state_hash -> {action: Q-value}
        self.q_table: dict[str, dict[int, float]] = defaultdict(
            lambda: {a: 0.0 for a in ACTIONS.keys()}
        )
        self.epsilon = config.epsilon
        self.steps = 0

    def _discretize_state(self, state: RLState) -> str:
        """Discretize continuous state to hash key with adaptive binning.

        Strategy:
        - Binary features (0/1): Keep as-is
        - Progress features (0-1): Use quantile-based binning
        - Confidence features (0-1): Use threshold-based binning
        - Rate features (-inf to inf): Use adaptive thresholds

        Returns:
            Hash key string for Q-table lookup
        """
        arr = state.to_array()
        n_bins = self.config.n_bins

        if not self.config.adaptive_binning:
            # Simple uniform binning
            binned = []
            for x in arr:
                x_clipped = max(0.0, min(1.0, x))  # Clip to [0,1]
                bin_idx = min(int(x_clipped * n_bins), n_bins - 1)
                binned.append(bin_idx)
            return str(tuple(binned))

        # Adaptive binning with feature-specific strategies
        binned = []

        # Unpack features (matching RLState.to_array() order - 15 features)
        (
            f1_progress,                 # Round progress (0-1)
            f2_n_obs_ratio,             # Observations ratio (0-1)
            f3_has_categorical,         # Binary (0/1)
            f4_has_log_scale,           # Binary (0/1)
            f5_space_coverage,          # Search space coverage (0-1)
            f6_model_uncertainty,       # Model uncertainty (0-1)
            f7_noise_ratio,             # Noise ratio (0-1)
            f8_replicate_need_score,    # Replicate need (0-1)
            f9_batch_kpi_cv,            # Batch KPI CV (0-1)
            f10_improvement_velocity,   # Improvement velocity (0-1)
            f11_ei_decay_proxy,         # EI decay (0-1)
            f12_convergence_confidence, # Convergence confidence (0-1)
            f13_convergence_plateau,    # Plateau indicator (0-1)
            f14_local_smoothness,       # Local smoothness (0-1)
            f15_batch_param_spread,     # Parameter spread (0-1)
        ) = arr

        # Progress features: quantile binning
        binned.append(self._bin_progress(f1_progress, n_bins))
        binned.append(self._bin_progress(f2_n_obs_ratio, n_bins))

        # Binary features: keep as-is (0/1 → 0/1)
        binned.append(int(f3_has_categorical))
        binned.append(int(f4_has_log_scale))

        # Coverage and uncertainty
        binned.append(self._bin_coverage(f5_space_coverage, n_bins))
        binned.append(self._bin_uncertainty(f6_model_uncertainty, n_bins))
        binned.append(self._bin_uncertainty(f7_noise_ratio, n_bins))

        # Replicate need and batch variability
        binned.append(self._bin_confidence(f8_replicate_need_score, n_bins))
        binned.append(self._bin_uncertainty(f9_batch_kpi_cv, n_bins))

        # Improvement and decay signals
        binned.append(self._bin_centered_rate(f10_improvement_velocity, n_bins))
        binned.append(self._bin_confidence(f11_ei_decay_proxy, n_bins))

        # Convergence signals
        binned.append(self._bin_confidence(f12_convergence_confidence, n_bins))
        binned.append(self._bin_confidence(f13_convergence_plateau, n_bins))

        # Landscape features
        binned.append(self._bin_confidence(f14_local_smoothness, n_bins))
        binned.append(self._bin_confidence(f15_batch_param_spread, n_bins))

        return str(tuple(binned))

    def _bin_progress(self, x: float, n_bins: int) -> int:
        """Bin progress features (0-1) with focus on early/late stages."""
        x = max(0.0, min(1.0, x))
        if n_bins == 2:
            return int(x >= 0.5)
        elif n_bins == 3:
            # Early (0-0.3), Mid (0.3-0.7), Late (0.7-1.0)
            if x < 0.3:
                return 0
            elif x < 0.7:
                return 1
            else:
                return 2
        else:  # n_bins >= 5
            return min(int(x * n_bins), n_bins - 1)

    def _bin_kpi(self, x: float, n_bins: int) -> int:
        """Bin KPI features (0-1 normalized) with focus on high values."""
        x = max(0.0, min(1.0, x))
        if n_bins == 2:
            return int(x >= 0.7)  # Low vs high performance
        elif n_bins == 3:
            # Poor (<0.6), Good (0.6-0.8), Excellent (>0.8)
            if x < 0.6:
                return 0
            elif x < 0.8:
                return 1
            else:
                return 2
        else:  # n_bins >= 5
            # Focus resolution on high-performance region
            if x < 0.5:
                return 0
            elif x < 0.7:
                return 1
            elif x < 0.8:
                return 2
            elif x < 0.9:
                return 3
            else:
                return 4

    def _bin_uncertainty(self, x: float, n_bins: int) -> int:
        """Bin uncertainty features (0-1) with focus on low uncertainty."""
        x = max(0.0, min(1.0, x))
        if n_bins == 2:
            return int(x >= 0.1)  # Low vs high uncertainty
        elif n_bins == 3:
            # Low (<0.05), Medium (0.05-0.15), High (>0.15)
            if x < 0.05:
                return 0
            elif x < 0.15:
                return 1
            else:
                return 2
        else:  # n_bins >= 5
            thresholds = [0.03, 0.08, 0.15, 0.25]
            for i, threshold in enumerate(thresholds):
                if x < threshold:
                    return i
            return n_bins - 1

    def _bin_confidence(self, x: float, n_bins: int) -> int:
        """Bin confidence features (0-1) with focus on high confidence."""
        x = max(0.0, min(1.0, x))
        if n_bins == 2:
            return int(x >= 0.7)
        elif n_bins == 3:
            # Low (<0.5), Medium (0.5-0.8), High (>0.8)
            if x < 0.5:
                return 0
            elif x < 0.8:
                return 1
            else:
                return 2
        else:  # n_bins >= 5
            return min(int(x * n_bins), n_bins - 1)

    def _bin_stagnation(self, x: float, n_bins: int) -> int:
        """Bin runs_since_improvement (0-1) with focus on recent stagnation."""
        x = max(0.0, min(1.0, x))
        if n_bins == 2:
            return int(x >= 0.3)  # Recent vs stagnant
        elif n_bins == 3:
            # Recent (<0.2), Some (0.2-0.5), Stagnant (>0.5)
            if x < 0.2:
                return 0
            elif x < 0.5:
                return 1
            else:
                return 2
        else:  # n_bins >= 5
            return min(int(x * n_bins), n_bins - 1)

    def _bin_coverage(self, x: float, n_bins: int) -> int:
        """Bin search space coverage (0-1) with focus on low coverage."""
        x = max(0.0, min(1.0, x))
        if n_bins == 2:
            return int(x >= 0.5)
        elif n_bins == 3:
            # Low (<0.3), Medium (0.3-0.7), High (>0.7)
            if x < 0.3:
                return 0
            elif x < 0.7:
                return 1
            else:
                return 2
        else:  # n_bins >= 5
            return min(int(x * n_bins), n_bins - 1)

    def _bin_rate(self, x: float, n_bins: int) -> int:
        """Bin rate features (0-1) with linear binning."""
        x = max(0.0, min(1.0, x))
        return min(int(x * n_bins), n_bins - 1)

    def _bin_centered_rate(self, x: float, n_bins: int) -> int:
        """Bin rate features centered at 0 (e.g., improvement_rate, gradient).

        Maps negative/zero/positive to bins.
        """
        # Clip to [-1, 1]
        x = max(-1.0, min(1.0, x))

        if n_bins == 2:
            return int(x >= 0)  # Negative vs positive
        elif n_bins == 3:
            # Negative, Near-zero, Positive
            if x < -0.1:
                return 0
            elif x < 0.1:
                return 1
            else:
                return 2
        else:  # n_bins >= 5
            # Map [-1, 1] to [0, n_bins-1]
            normalized = (x + 1.0) / 2.0  # Map to [0, 1]
            return min(int(normalized * n_bins), n_bins - 1)

    def select_action(self, state: RLState, explore: bool = True) -> int:
        """ε-greedy action selection."""
        if explore and random.random() < self.epsilon:
            return random.choice(list(ACTIONS.keys()))

        state_key = self._discretize_state(state)
        q_values = self.q_table[state_key]
        return max(q_values, key=q_values.get)  # type: ignore[arg-type]

    def update(
        self,
        state: RLState,
        action: int,
        reward: float,
        next_state: RLState | None,
        done: bool,
    ) -> None:
        """Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]."""
        state_key = self._discretize_state(state)

        current_q = self.q_table[state_key][action]

        if done or next_state is None:
            target_q = reward
        else:
            next_state_key = self._discretize_state(next_state)
            max_next_q = max(self.q_table[next_state_key].values())
            target_q = reward + self.config.gamma * max_next_q

        # Q-learning update
        self.q_table[state_key][action] += self.config.learning_rate * (
            target_q - current_q
        )

        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )
        self.steps += 1

    def save(self, path: Path) -> None:
        """Save Q-table to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "q_table": dict(self.q_table),
                "epsilon": self.epsilon,
                "steps": self.steps,
            }, f)

    def load(self, path: Path) -> None:
        """Load Q-table from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.q_table = defaultdict(
                lambda: {a: 0.0 for a in ACTIONS.keys()},
                data["q_table"],
            )
            self.epsilon = data.get("epsilon", self.config.epsilon)
            self.steps = data.get("steps", 0)


# ---------------------------------------------------------------------------
# Main RL Strategy Selector
# ---------------------------------------------------------------------------

class RLStrategySelector:
    """RL-based strategy selector with offline pre-training + online learning."""

    def __init__(self, config: RLConfig | None = None):
        self.config = config or RLConfig()
        self.agent = QLearningAgent(self.config)
        self.replay_buffer = ExperienceReplay(self.config.replay_capacity)

        # Try to load pre-trained model
        model_path = Path(self.config.model_save_path)
        if model_path.exists():
            logger.info("Loading pre-trained RL model from %s", model_path)
            self.agent.load(model_path)

        # Try to load replay buffer
        replay_path = Path(self.config.replay_save_path)
        if replay_path.exists():
            logger.info("Loading replay buffer from %s", replay_path)
            self.replay_buffer.load(replay_path)

    def select_action(
        self,
        snapshot: CampaignSnapshot,
        diagnostics: DiagnosticSignals,
        explore: bool = True,
    ) -> tuple[int, str]:
        """Select action (backend strategy) for current state.

        Returns:
            (action_index, backend_name)
        """
        state = RLState.from_snapshot(snapshot, diagnostics)
        action = self.agent.select_action(state, explore=explore)
        action_name = ACTIONS[action]
        backend = ACTION_TO_BACKEND[action_name]
        return action, backend

    def learn_from_experience(
        self,
        state: RLState,
        action: int,
        reward: float,
        next_state: RLState | None,
        done: bool,
    ) -> None:
        """Online learning: update Q-values from single transition."""
        # Store in replay buffer
        exp = Experience(state, action, reward, next_state, done)
        self.replay_buffer.add(exp)

        # Q-learning update
        self.agent.update(state, action, reward, next_state, done)

        # Optionally: batch update from replay buffer
        if self.agent.steps % self.config.train_frequency == 0:
            self._train_from_replay(batch_size=self.config.batch_size)

    def _train_from_replay(self, batch_size: int) -> None:
        """Train from random batch of experiences (experience replay)."""
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        for exp in batch:
            self.agent.update(
                exp.state,
                exp.action,
                exp.reward,
                exp.next_state,
                exp.done,
            )

    def save(self) -> None:
        """Save model and replay buffer."""
        model_path = Path(self.config.model_save_path)
        self.agent.save(model_path)
        logger.info("Saved RL model to %s", model_path)

        replay_path = Path(self.config.replay_save_path)
        self.replay_buffer.save(replay_path)
        logger.info("Saved replay buffer to %s", replay_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Global singleton (lazy initialization)
_rl_selector: RLStrategySelector | None = None


def get_rl_selector() -> RLStrategySelector:
    """Get or create global RL selector."""
    global _rl_selector
    if _rl_selector is None:
        _rl_selector = RLStrategySelector()
    return _rl_selector


def select_strategy_rl(
    snapshot: CampaignSnapshot,
    explore: bool = True,
    fallback_to_rule_based: bool = True,
) -> StrategyDecision:
    """RL-based strategy selection with rule-based fallback.

    Args:
        snapshot: Current campaign state
        explore: Whether to use ε-greedy exploration
        fallback_to_rule_based: If True, use rule-based selector as backup

    Returns:
        StrategyDecision with backend recommendation
    """
    try:
        selector = get_rl_selector()

        # Compute diagnostics (needed for state extraction)
        diagnostics = compute_diagnostics(snapshot)

        # RL action selection
        action, backend = selector.select_action(snapshot, diagnostics, explore=explore)
        action_name = ACTIONS[action]

        # Build decision
        return StrategyDecision(
            backend_name=backend,
            phase=action_name,
            reason=f"RL-selected: {action_name} (action={action}, ε={selector.agent.epsilon:.3f})",
            confidence=1.0 - selector.agent.epsilon,  # higher confidence = less exploration
            diagnostics=diagnostics,
            explanation=f"RL agent chose '{action_name}' strategy (explore={explore})",
        )

    except Exception as exc:
        logger.error("RL selector failed: %s", exc, exc_info=True)
        if fallback_to_rule_based:
            logger.info("Falling back to rule-based selector")
            return rule_based_select_strategy(snapshot)
        raise


# ---------------------------------------------------------------------------
# Offline Training
# ---------------------------------------------------------------------------

def train_rl_selector_offline(
    historical_campaigns: list[dict[str, Any]],
    config: RLConfig | None = None,
    save_path: str | None = None,
) -> RLStrategySelector:
    """Train RL selector offline from historical campaign data.

    Args:
        historical_campaigns: List of campaign traces with:
            {
                "snapshots": [CampaignSnapshot, ...],
                "actions": [action_index, ...],
                "rewards": [reward, ...],
                "kpi_history": [kpi, ...],
            }
        config: RL hyperparameters
        save_path: Where to save trained model

    Returns:
        Trained RLStrategySelector
    """
    selector = RLStrategySelector(config)

    logger.info("Starting offline training on %d campaigns", len(historical_campaigns))

    total_transitions = 0
    for camp_idx, campaign in enumerate(historical_campaigns):
        snapshots = campaign["snapshots"]
        actions = campaign["actions"]
        rewards = campaign["rewards"]

        # Convert snapshots to states
        states = []
        for snap in snapshots:
            diag = compute_diagnostic_signals(snap)
            state = RLState.from_snapshot(snap, diag)
            states.append(state)

        # Create transitions
        for i in range(len(states) - 1):
            next_state = states[i + 1] if i + 1 < len(states) else None
            done = (i == len(states) - 1)

            selector.learn_from_experience(
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                next_state=next_state,
                done=done,
            )
            total_transitions += 1

        if (camp_idx + 1) % 10 == 0:
            logger.info("Processed %d/%d campaigns", camp_idx + 1, len(historical_campaigns))

    logger.info("Offline training complete: %d transitions", total_transitions)

    # Save trained model
    if save_path:
        selector.config.model_save_path = save_path
    selector.save()

    return selector

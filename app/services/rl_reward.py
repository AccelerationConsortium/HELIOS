"""Reward shaping for RL-based strategy selector.

Design principles:
1. **Immediate reward**: KPI improvement in this round
2. **Delayed reward**: Contribution to final convergence
3. **Cost penalty**: Resource consumption (rounds, QC failures)
4. **Exploration bonus**: Encourage diverse strategies early

Reward equation:
    R(t) = α·ΔKP(t) - β·cost(t) + γ·convergence_bonus(t) + δ·exploration_bonus(t)

Where:
    - ΔKP(t): Normalized KPI improvement
    - cost(t): Round cost + QC failure penalty
    - convergence_bonus: Reward for reaching target or plateau
    - exploration_bonus: Encourage trying different actions early
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from app.services.strategy_selector import CampaignSnapshot, DiagnosticSignals

logger = logging.getLogger(__name__)

__all__ = [
    "RewardConfig",
    "RewardComponents",
    "compute_reward",
    "compute_immediate_reward",
    "compute_terminal_reward",
]


# ---------------------------------------------------------------------------
# Reward Config
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Hyperparameters for reward shaping."""

    # Weight factors
    alpha: float = 1.0  # KPI improvement weight
    beta: float = 0.01  # Cost penalty weight
    gamma: float = 0.5  # Convergence bonus weight
    delta: float = 0.1  # Exploration bonus weight

    # Normalization
    kpi_scale: float = 10.0  # Scale KPI to ~[0, 1]
    round_cost: float = -0.01  # Fixed cost per round
    qc_failure_penalty: float = -0.1  # Penalty per QC failure

    # Terminal rewards
    convergence_bonus: float = 1.0  # Bonus for reaching target
    plateau_bonus: float = 0.5  # Bonus for stable convergence
    early_convergence_bonus: float = 0.3  # Extra for converging early

    # Exploration bonuses
    exploration_bonus_early: float = 0.05  # Bonus in first 20% of rounds
    action_diversity_bonus: float = 0.02  # Bonus for trying new actions


@dataclass(frozen=True)
class RewardComponents:
    """Breakdown of reward components for explainability."""

    kpi_improvement: float
    round_cost: float
    qc_penalty: float
    convergence_bonus: float
    exploration_bonus: float
    total: float

    # Metadata
    kpi_prev: float | None
    kpi_curr: float | None
    round_num: int
    n_qc_failures: int = 0


# ---------------------------------------------------------------------------
# Immediate Reward (per round)
# ---------------------------------------------------------------------------

def compute_immediate_reward(
    kpi_prev: float | None,
    kpi_curr: float | None,
    direction: str,
    round_num: int,
    max_rounds: int,
    n_qc_failures: int = 0,
    action: int | None = None,
    prev_actions: list[int] | None = None,
    config: RewardConfig | None = None,
) -> RewardComponents:
    """Compute reward for a single round transition.

    Args:
        kpi_prev: KPI before this round (None if first round)
        kpi_curr: KPI after this round (None if failed)
        direction: "maximize" or "minimize"
        round_num: Current round number (1-based)
        max_rounds: Total rounds in campaign
        n_qc_failures: Number of QC failures in this round
        action: Action taken this round (for exploration bonus)
        prev_actions: Previous actions (for diversity bonus)
        config: Reward hyperparameters

    Returns:
        RewardComponents with breakdown
    """
    if config is None:
        config = RewardConfig()

    # --- 1. KPI Improvement ---
    kpi_improvement = 0.0
    if kpi_prev is not None and kpi_curr is not None:
        delta = kpi_curr - kpi_prev
        if direction == "minimize":
            delta = -delta  # Flip sign for minimization
        # Normalize to ~[0, 1]
        kpi_improvement = delta / config.kpi_scale
        # Clip to [-1, 1]
        kpi_improvement = max(-1.0, min(1.0, kpi_improvement))

    # --- 2. Round Cost ---
    round_cost = config.round_cost

    # --- 3. QC Penalty ---
    qc_penalty = n_qc_failures * config.qc_failure_penalty

    # --- 4. Convergence Bonus (handled separately in terminal reward) ---
    convergence_bonus = 0.0

    # --- 5. Exploration Bonus ---
    exploration_bonus = 0.0

    # Early exploration bonus (first 20% of campaign)
    progress = round_num / max_rounds
    if progress < 0.2:
        exploration_bonus += config.exploration_bonus_early

    # Action diversity bonus
    if action is not None and prev_actions is not None:
        # Bonus if this action is different from recent actions
        recent = prev_actions[-3:] if len(prev_actions) >= 3 else prev_actions
        if action not in recent:
            exploration_bonus += config.action_diversity_bonus

    # --- Total Reward ---
    total = (
        config.alpha * kpi_improvement
        + config.beta * round_cost
        + qc_penalty
        + config.delta * exploration_bonus
    )

    return RewardComponents(
        kpi_improvement=kpi_improvement,
        round_cost=round_cost,
        qc_penalty=qc_penalty,
        convergence_bonus=convergence_bonus,
        exploration_bonus=exploration_bonus,
        total=total,
        kpi_prev=kpi_prev,
        kpi_curr=kpi_curr,
        round_num=round_num,
        n_qc_failures=n_qc_failures,
    )


# ---------------------------------------------------------------------------
# Terminal Reward (end of campaign)
# ---------------------------------------------------------------------------

def compute_terminal_reward(
    snapshot: CampaignSnapshot,
    diagnostics: DiagnosticSignals,
    target_reached: bool,
    config: RewardConfig | None = None,
) -> float:
    """Compute terminal reward at end of campaign.

    Bonuses for:
    - Reaching target KPI
    - Converging stably (plateau)
    - Converging early (using fewer rounds)

    Args:
        snapshot: Final campaign state
        diagnostics: Final diagnostics
        target_reached: Whether target KPI was reached
        config: Reward hyperparameters

    Returns:
        Terminal reward (added to last round's reward)
    """
    if config is None:
        config = RewardConfig()

    terminal_reward = 0.0

    # --- Convergence Bonus ---
    if target_reached:
        terminal_reward += config.convergence_bonus

    # --- Plateau Bonus ---
    if diagnostics.convergence_status == "plateau":
        terminal_reward += config.plateau_bonus

    # --- Early Convergence Bonus ---
    progress = snapshot.round_number / snapshot.max_rounds
    if progress < 0.8 and (target_reached or diagnostics.convergence_status == "plateau"):
        # Extra bonus for converging before 80% of budget
        early_factor = 1.0 - progress
        terminal_reward += config.early_convergence_bonus * early_factor

    return terminal_reward * config.gamma


# ---------------------------------------------------------------------------
# Unified Reward Computation
# ---------------------------------------------------------------------------

def compute_reward(
    snapshot: CampaignSnapshot,
    diagnostics: DiagnosticSignals,
    kpi_prev: float | None,
    kpi_curr: float | None,
    n_qc_failures: int = 0,
    action: int | None = None,
    prev_actions: list[int] | None = None,
    is_terminal: bool = False,
    target_reached: bool = False,
    config: RewardConfig | None = None,
) -> RewardComponents:
    """Unified reward computation for RL training.

    Combines immediate reward + terminal reward (if applicable).

    Args:
        snapshot: Current campaign state
        diagnostics: Current diagnostics
        kpi_prev: Previous best KPI
        kpi_curr: Current best KPI
        n_qc_failures: Number of QC failures this round
        action: Action taken this round
        prev_actions: Previous actions (for diversity bonus)
        is_terminal: Whether this is the last round
        target_reached: Whether target KPI was reached
        config: Reward hyperparameters

    Returns:
        RewardComponents with total reward
    """
    if config is None:
        config = RewardConfig()

    # Compute immediate reward
    immediate = compute_immediate_reward(
        kpi_prev=kpi_prev,
        kpi_curr=kpi_curr,
        direction=snapshot.direction,
        round_num=snapshot.round_number,
        max_rounds=snapshot.max_rounds,
        n_qc_failures=n_qc_failures,
        action=action,
        prev_actions=prev_actions,
        config=config,
    )

    # Add terminal reward if final round
    if is_terminal:
        terminal = compute_terminal_reward(
            snapshot=snapshot,
            diagnostics=diagnostics,
            target_reached=target_reached,
            config=config,
        )
        total = immediate.total + terminal
        return RewardComponents(
            kpi_improvement=immediate.kpi_improvement,
            round_cost=immediate.round_cost,
            qc_penalty=immediate.qc_penalty,
            convergence_bonus=terminal,
            exploration_bonus=immediate.exploration_bonus,
            total=total,
            kpi_prev=kpi_prev,
            kpi_curr=kpi_curr,
            round_num=snapshot.round_number,
            n_qc_failures=n_qc_failures,
        )

    return immediate


# ---------------------------------------------------------------------------
# Reward Analysis Utilities
# ---------------------------------------------------------------------------

def analyze_reward_trace(
    rewards: list[RewardComponents],
    direction: str = "maximize",
) -> dict[str, float]:
    """Analyze reward trace for debugging.

    Returns:
        Dict with statistics:
        - total_reward: Sum of all rewards
        - avg_reward: Mean reward per round
        - kpi_improvement_total: Total KPI improvement
        - cost_total: Total cost penalty
        - convergence_bonus_total: Total convergence bonus
    """
    if not rewards:
        return {}

    total = sum(r.total for r in rewards)
    avg = total / len(rewards)

    kpi_improvements = [r.kpi_improvement for r in rewards if r.kpi_improvement != 0.0]
    kpi_improvement_total = sum(kpi_improvements)

    cost_total = sum(r.round_cost + r.qc_penalty for r in rewards)
    convergence_bonus_total = sum(r.convergence_bonus for r in rewards)

    return {
        "total_reward": total,
        "avg_reward": avg,
        "kpi_improvement_total": kpi_improvement_total,
        "cost_total": cost_total,
        "convergence_bonus_total": convergence_bonus_total,
        "n_rounds": len(rewards),
        "direction": direction,
    }

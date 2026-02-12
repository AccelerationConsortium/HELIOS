"""Historical campaign data collector for RL training.

Extracts training data from completed campaigns stored in campaign_state.db:
- Campaign snapshots at each round
- Actions taken (strategy decisions)
- Rewards (KPI improvements)
- Terminal outcomes

Data format for offline training:
    {
        "campaign_id": "camp-abc123",
        "snapshots": [CampaignSnapshot, ...],  # one per round
        "actions": [0, 1, 2, ...],  # RL action indices
        "rewards": [0.05, -0.01, 0.12, ...],  # per-round rewards
        "kpi_history": [98.5, 98.7, 99.1, ...],
        "final_kpi": 99.3,
        "converged": True,
        "target_reached": False,
    }
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from app.services.strategy_selector import CampaignSnapshot, compute_diagnostics
from app.services.rl_strategy_selector import ACTIONS, ACTION_TO_BACKEND
from app.services.rl_reward import RewardConfig, compute_reward

logger = logging.getLogger(__name__)

__all__ = [
    "collect_historical_campaigns",
    "extract_campaign_trace",
    "action_from_backend_name",
    "save_training_dataset",
    "load_training_dataset",
]


# ---------------------------------------------------------------------------
# Backend name → RL action mapping
# ---------------------------------------------------------------------------

def action_from_backend_name(backend: str) -> int:
    """Map backend name to RL action index.

    Args:
        backend: Backend name (e.g., "lhs", "built_in", "optuna_tpe")

    Returns:
        Action index (0-3)
    """
    # Mapping logic (heuristic, can be improved)
    backend_lower = backend.lower()

    if "lhs" in backend_lower or "random" in backend_lower or "grid" in backend_lower:
        return 0  # explore
    elif "bayesian" in backend_lower or "tpe" in backend_lower or "built_in" in backend_lower:
        return 1  # exploit
    elif "cmaes" in backend_lower or "de" in backend_lower or "refine" in backend_lower:
        return 2  # refine
    elif "stabilize" in backend_lower or "replicate" in backend_lower:
        return 3  # stabilize
    else:
        logger.warning("Unknown backend '%s', defaulting to action=1 (exploit)", backend)
        return 1


# ---------------------------------------------------------------------------
# Extract single campaign trace
# ---------------------------------------------------------------------------

def extract_campaign_trace(
    campaign_id: str,
    db_path: str = "otbot.db",
    reward_config: RewardConfig | None = None,
) -> dict[str, Any] | None:
    """Extract training data from a single campaign.

    Args:
        campaign_id: Campaign ID
        db_path: Path to SQLite database
        reward_config: Reward hyperparameters

    Returns:
        Campaign trace dict, or None if campaign incomplete/invalid
    """
    if reward_config is None:
        reward_config = RewardConfig()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        # Load campaign metadata
        cur.execute(
            "SELECT * FROM campaign_state WHERE campaign_id = ?",
            (campaign_id,),
        )
        row = cur.fetchone()
        if not row:
            logger.warning("Campaign %s not found", campaign_id)
            return None

        status = row["status"]
        if status not in ("completed", "failed"):
            logger.debug("Campaign %s not finished (status=%s)", campaign_id, status)
            return None

        input_json = json.loads(row["input_json"])
        direction = input_json.get("direction", "maximize")
        max_rounds = input_json.get("max_rounds", 10)
        target_value = input_json.get("target_value")

        # Load round data
        cur.execute(
            """
            SELECT round_number, strategy, n_candidates, kpis, params
            FROM campaign_rounds
            WHERE campaign_id = ?
            ORDER BY round_number
            """,
            (campaign_id,),
        )
        rounds = cur.fetchall()

        if not rounds:
            logger.warning("Campaign %s has no rounds", campaign_id)
            return None

        # Build snapshots, actions, rewards
        snapshots: list[CampaignSnapshot] = []
        actions: list[int] = []
        rewards: list[float] = []
        kpi_history: list[float] = []

        cumulative_kpis: list[float] = []
        cumulative_params: list[dict[str, Any]] = []
        prev_best_kpi: float | None = None

        for i, round_row in enumerate(rounds):
            round_num = round_row["round_number"]
            strategy = round_row["strategy"] or "built_in"
            n_candidates = round_row["n_candidates"]

            # Parse KPIs and params
            round_kpis_json = round_row["kpis"]
            round_params_json = round_row["params"]

            round_kpis = json.loads(round_kpis_json) if round_kpis_json else []
            round_params = json.loads(round_params_json) if round_params_json else []

            # Accumulate history
            cumulative_kpis.extend(round_kpis)
            cumulative_params.extend(round_params)

            # Best KPI so far
            if cumulative_kpis:
                if direction == "maximize":
                    best_kpi = max(cumulative_kpis)
                else:
                    best_kpi = min(cumulative_kpis)
            else:
                best_kpi = None

            # Build snapshot
            snapshot = CampaignSnapshot(
                round_number=round_num,
                max_rounds=max_rounds,
                n_observations=len(cumulative_kpis),
                n_dimensions=len(input_json.get("dimensions", [])),
                has_categorical=any(
                    d.get("choices") is not None
                    for d in input_json.get("dimensions", [])
                ),
                has_log_scale=any(
                    d.get("log_scale", False)
                    for d in input_json.get("dimensions", [])
                ),
                kpi_history=tuple(kpi_history),
                direction=direction,
                last_batch_kpis=tuple(round_kpis),
                last_batch_params=tuple(round_params),
                best_kpi_so_far=best_kpi,
                all_params=tuple(cumulative_params),
                all_kpis=tuple(cumulative_kpis),
            )
            snapshots.append(snapshot)

            # Map strategy to action
            action = action_from_backend_name(strategy)
            actions.append(action)

            # Compute reward
            # (simplified: use KPI improvement as reward proxy)
            if best_kpi is not None:
                if prev_best_kpi is not None:
                    delta = best_kpi - prev_best_kpi
                    if direction == "minimize":
                        delta = -delta
                    reward = delta / reward_config.kpi_scale
                else:
                    reward = 0.0  # First round has no improvement
                prev_best_kpi = best_kpi
            else:
                reward = reward_config.round_cost  # Failed round

            rewards.append(reward)
            kpi_history.extend(round_kpis)

        # Terminal reward
        final_kpi = best_kpi
        target_reached = False
        if target_value is not None and final_kpi is not None:
            if direction == "maximize":
                target_reached = final_kpi >= target_value
            else:
                target_reached = final_kpi <= target_value

        # Add terminal bonus to last reward
        if target_reached:
            rewards[-1] += reward_config.convergence_bonus * reward_config.gamma

        return {
            "campaign_id": campaign_id,
            "snapshots": snapshots,
            "actions": actions,
            "rewards": rewards,
            "kpi_history": kpi_history,
            "final_kpi": final_kpi,
            "converged": status == "completed",
            "target_reached": target_reached,
            "direction": direction,
            "n_rounds": len(rounds),
        }

    except Exception as exc:
        logger.error("Failed to extract campaign %s: %s", campaign_id, exc, exc_info=True)
        return None

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Collect all historical campaigns
# ---------------------------------------------------------------------------

def collect_historical_campaigns(
    db_path: str = "otbot.db",
    min_rounds: int = 3,
    reward_config: RewardConfig | None = None,
) -> list[dict[str, Any]]:
    """Collect training data from all completed campaigns.

    Args:
        db_path: Path to SQLite database
        min_rounds: Minimum rounds to include campaign
        reward_config: Reward hyperparameters

    Returns:
        List of campaign trace dicts
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Find all completed campaigns
    cur.execute(
        """
        SELECT campaign_id, status, created_at
        FROM campaign_state
        WHERE status IN ('completed', 'failed')
        ORDER BY created_at DESC
        """
    )
    campaign_rows = cur.fetchall()
    conn.close()

    logger.info("Found %d completed campaigns", len(campaign_rows))

    # Extract traces
    traces: list[dict[str, Any]] = []
    for row in campaign_rows:
        campaign_id = row[0]
        trace = extract_campaign_trace(campaign_id, db_path, reward_config)

        if trace is not None and trace["n_rounds"] >= min_rounds:
            traces.append(trace)

    logger.info("Extracted %d valid campaign traces (min_rounds=%d)", len(traces), min_rounds)

    return traces


# ---------------------------------------------------------------------------
# Save/Load training dataset
# ---------------------------------------------------------------------------

def save_training_dataset(
    traces: list[dict[str, Any]],
    output_path: str = "models/rl_training_data.json",
) -> None:
    """Save training dataset to JSON.

    Snapshots are serialized to dicts for JSON compatibility.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Serialize snapshots to dicts
    serialized_traces = []
    for trace in traces:
        serialized = dict(trace)
        # Convert CampaignSnapshot objects to dicts
        serialized["snapshots"] = [
            {
                "round_number": s.round_number,
                "max_rounds": s.max_rounds,
                "n_observations": s.n_observations,
                "n_dimensions": s.n_dimensions,
                "has_categorical": s.has_categorical,
                "has_log_scale": s.has_log_scale,
                "kpi_history": list(s.kpi_history),
                "direction": s.direction,
                "last_batch_kpis": list(s.last_batch_kpis),
                "last_batch_params": list(s.last_batch_params),
                "best_kpi_so_far": s.best_kpi_so_far,
                "all_params": list(s.all_params),
                "all_kpis": list(s.all_kpis),
            }
            for s in trace["snapshots"]
        ]
        serialized_traces.append(serialized)

    with open(output, "w") as f:
        json.dump(serialized_traces, f, indent=2)

    logger.info("Saved %d campaign traces to %s", len(traces), output)


def load_training_dataset(
    input_path: str = "models/rl_training_data.json",
) -> list[dict[str, Any]]:
    """Load training dataset from JSON.

    Deserializes snapshot dicts back to CampaignSnapshot objects.
    """
    with open(input_path, "r") as f:
        serialized_traces = json.load(f)

    traces = []
    for serialized in serialized_traces:
        trace = dict(serialized)
        # Deserialize snapshots
        trace["snapshots"] = [
            CampaignSnapshot(
                round_number=s["round_number"],
                max_rounds=s["max_rounds"],
                n_observations=s["n_observations"],
                n_dimensions=s["n_dimensions"],
                has_categorical=s["has_categorical"],
                has_log_scale=s["has_log_scale"],
                kpi_history=tuple(s["kpi_history"]),
                direction=s["direction"],
                last_batch_kpis=tuple(s["last_batch_kpis"]),
                last_batch_params=tuple(s["last_batch_params"]),
                best_kpi_so_far=s["best_kpi_so_far"],
                all_params=tuple(s["all_params"]),
                all_kpis=tuple(s["all_kpis"]),
            )
            for s in serialized["snapshots"]
        ]
        traces.append(trace)

    logger.info("Loaded %d campaign traces from %s", len(traces), input_path)

    return traces

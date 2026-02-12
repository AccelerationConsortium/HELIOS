#!/usr/bin/env python3
"""Generate synthetic campaign data for RL strategy selector training.

Usage:
    python3 scripts/generate_synthetic_rl_data.py --output models/synthetic_rl_data.json --campaigns 50
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_campaign_trace(
    campaign_id: int,
    strategy_profile: str,
    target_kpi: float = 90.0,
    max_rounds: int = 10,
    n_dimensions: int = 3,
    batch_size: int = 10,
) -> dict:
    """Generate a synthetic campaign trace with realistic dynamics.

    Args:
        campaign_id: Unique campaign identifier
        strategy_profile: One of "explorer", "exploiter", "balanced", "adaptive"
        target_kpi: Target KPI value
        max_rounds: Maximum rounds to simulate
        n_dimensions: Number of search dimensions
        batch_size: Batch size per round

    Returns:
        Campaign trace dict with snapshots, actions, rewards
    """
    snapshots = []
    actions = []
    rewards = []

    # Initialize campaign state
    current_kpi = random.uniform(40.0, 60.0)  # Start with mediocre KPI
    n_runs = 0
    best_kpi = current_kpi
    rounds_since_improvement = 0

    # Track full history
    kpi_history = []
    all_kpis = []
    all_params = []

    # Strategy profiles define action preferences
    strategy_probs = {
        "explorer": [0.5, 0.2, 0.2, 0.1],  # explore, exploit, refine, stabilize
        "exploiter": [0.1, 0.5, 0.3, 0.1],
        "balanced": [0.25, 0.25, 0.25, 0.25],
        "adaptive": [0.3, 0.3, 0.2, 0.2],  # Will adapt based on state
    }

    probs = strategy_probs.get(strategy_profile, strategy_probs["balanced"])

    for round_num in range(max_rounds):
        # Compute 16-dim state vector (matching RLState)
        kpi_improvement_rate = (current_kpi - best_kpi) / (best_kpi + 1e-6) if round_num > 0 else 0.0
        runs_per_round = batch_size

        # Generate batch KPIs for this round
        batch_kpis = []
        batch_params = []
        for _ in range(batch_size):
            # Simulate run params and KPI
            params = {f"param_{i}": random.uniform(0.0, 1.0) for i in range(n_dimensions)}
            kpi = current_kpi + random.uniform(-5.0, 5.0)
            batch_kpis.append(kpi)
            batch_params.append(params)
            all_kpis.append(kpi)
            all_params.append(params)

        # Update KPI history
        round_avg_kpi = sum(batch_kpis) / len(batch_kpis)
        kpi_history.append(round_avg_kpi)

        # Campaign context (4 features)
        f1_round_progress = round_num / max_rounds
        f2_runs_so_far = n_runs / (max_rounds * runs_per_round)
        f3_current_kpi = current_kpi / 100.0  # Normalize to [0,1]
        f4_best_kpi = best_kpi / 100.0

        # Epistemic signals (4 features)
        f5_kpi_stddev = random.uniform(2.0, 8.0) / 100.0  # Uncertainty
        f6_model_confidence = max(0.0, 1.0 - f5_kpi_stddev * 5)  # Higher stddev → lower confidence
        f7_runs_since_improvement = min(1.0, rounds_since_improvement / 3.0)
        f8_search_space_coverage = min(1.0, n_runs / 50.0)

        # Aleatoric signals (4 features)
        f9_kpi_variance_recent = random.uniform(1.0, 5.0) / 100.0
        f10_noise_level = f9_kpi_variance_recent * 2.0
        f11_outlier_rate = random.uniform(0.0, 0.1)
        f12_repeatability = max(0.0, 1.0 - f10_noise_level * 2)

        # Saturation signals (2 features)
        f13_improvement_rate = max(-0.5, min(0.5, kpi_improvement_rate))
        f14_convergence_indicator = 1.0 if rounds_since_improvement >= 3 else 0.0

        # Landscape signals (2 features)
        f15_local_gradient = random.uniform(-0.2, 0.3)  # Search space gradient
        f16_flatness = 1.0 - abs(f15_local_gradient)

        # Create CampaignSnapshot-compatible dict
        snapshot = {
            "round_number": round_num,
            "max_rounds": max_rounds,
            "n_observations": n_runs,
            "n_dimensions": n_dimensions,
            "has_categorical": False,
            "has_log_scale": False,
            "kpi_history": list(kpi_history),
            "direction": "maximize",
            "last_batch_kpis": batch_kpis,
            "last_batch_params": batch_params,
            "best_kpi_so_far": best_kpi,
            "all_params": list(all_params),
            "all_kpis": list(all_kpis),
        }
        snapshots.append(snapshot)

        # Select action based on strategy profile (with some adaptation)
        if strategy_profile == "adaptive":
            # Adapt based on state features computed above
            convergence_indicator = f14_convergence_indicator
            search_space_coverage = f8_search_space_coverage
            model_confidence = f6_model_confidence

            # Adapt based on state
            if convergence_indicator > 0.5:
                # Converged → stabilize or refine
                probs = [0.1, 0.1, 0.3, 0.5]
            elif search_space_coverage < 0.3:
                # Early exploration needed
                probs = [0.5, 0.2, 0.2, 0.1]
            elif model_confidence > 0.7:
                # Confident → exploit
                probs = [0.1, 0.5, 0.3, 0.1]
            else:
                # Default adaptive
                probs = [0.3, 0.3, 0.2, 0.2]

        action = random.choices([0, 1, 2, 3], weights=probs)[0]
        actions.append(action)

        # Simulate round execution and compute reward
        # Action effects: 0=explore (high variance), 1=exploit (improve KPI), 2=refine (moderate), 3=stabilize (low variance)
        if action == 0:  # explore
            kpi_delta = random.uniform(-5.0, 10.0)  # High variance
            cost = 1.0
        elif action == 1:  # exploit
            kpi_delta = random.uniform(0.0, 8.0)  # Positive bias
            cost = 1.2
        elif action == 2:  # refine
            kpi_delta = random.uniform(-2.0, 6.0)  # Moderate
            cost = 1.0
        else:  # stabilize
            kpi_delta = random.uniform(-1.0, 2.0)  # Low variance
            cost = 0.8

        # Update state
        new_kpi = min(100.0, max(0.0, current_kpi + kpi_delta))
        kpi_improvement = new_kpi - current_kpi

        if new_kpi > best_kpi:
            best_kpi = new_kpi
            rounds_since_improvement = 0
        else:
            rounds_since_improvement += 1

        current_kpi = new_kpi
        n_runs += runs_per_round

        # Compute reward (matching RL reward function)
        kpi_reward = kpi_improvement / 100.0  # Normalize
        cost_penalty = -cost * 0.1

        # Convergence bonus
        convergence_bonus = 0.0
        if rounds_since_improvement == 0 and best_kpi >= target_kpi:
            convergence_bonus = 1.0

        # Exploration bonus
        exploration_bonus = 0.0
        if action == 0 and f8_search_space_coverage < 0.5:
            exploration_bonus = 0.1

        reward = kpi_reward + cost_penalty + convergence_bonus + exploration_bonus
        rewards.append(reward)

        # Early stopping if target reached and stable
        if best_kpi >= target_kpi and rounds_since_improvement >= 2:
            break

    # Campaign metadata
    converged = rounds_since_improvement >= 2
    target_reached = best_kpi >= target_kpi

    return {
        "campaign_id": f"synthetic_{campaign_id:03d}",
        "strategy_profile": strategy_profile,
        "snapshots": snapshots,
        "actions": actions,
        "rewards": rewards,
        "n_rounds": len(snapshots),
        "final_kpi": current_kpi,
        "best_kpi": best_kpi,
        "converged": converged,
        "target_reached": target_reached,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic RL training data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/synthetic_rl_data.json",
        help="Output path for synthetic dataset (default: models/synthetic_rl_data.json)",
    )
    parser.add_argument(
        "--campaigns",
        type=int,
        default=50,
        help="Number of campaigns to generate (default: 50)",
    )
    parser.add_argument(
        "--target-kpi",
        type=float,
        default=90.0,
        help="Target KPI for campaigns (default: 90.0)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum rounds per campaign (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    logger.info(f"Generating {args.campaigns} synthetic campaigns...")
    logger.info(f"Target KPI: {args.target_kpi}, Max rounds: {args.max_rounds}")

    # Generate campaigns with different strategy profiles
    strategy_profiles = ["explorer", "exploiter", "balanced", "adaptive"]
    campaigns = []

    for i in range(args.campaigns):
        profile = strategy_profiles[i % len(strategy_profiles)]
        campaign = generate_campaign_trace(
            campaign_id=i,
            strategy_profile=profile,
            target_kpi=args.target_kpi,
            max_rounds=args.max_rounds,
        )
        campaigns.append(campaign)

        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i + 1}/{args.campaigns} campaigns")

    # Compute statistics
    total_rounds = sum(c["n_rounds"] for c in campaigns)
    avg_rounds = total_rounds / len(campaigns)
    converged_count = sum(1 for c in campaigns if c["converged"])
    target_reached_count = sum(1 for c in campaigns if c["target_reached"])
    avg_final_kpi = sum(c["final_kpi"] for c in campaigns) / len(campaigns)

    logger.info(f"\n{'='*60}")
    logger.info("Dataset Statistics:")
    logger.info(f"  Total campaigns: {len(campaigns)}")
    logger.info(f"  Total rounds: {total_rounds}")
    logger.info(f"  Avg rounds per campaign: {avg_rounds:.2f}")
    logger.info(f"  Avg final KPI: {avg_final_kpi:.2f}")
    logger.info(f"  Converged: {converged_count}/{len(campaigns)} ({converged_count/len(campaigns)*100:.1f}%)")
    logger.info(f"  Target reached: {target_reached_count}/{len(campaigns)} ({target_reached_count/len(campaigns)*100:.1f}%)")

    # Strategy profile breakdown
    logger.info(f"\n  Strategy profiles:")
    for profile in strategy_profiles:
        profile_campaigns = [c for c in campaigns if c["strategy_profile"] == profile]
        avg_kpi = sum(c["final_kpi"] for c in profile_campaigns) / len(profile_campaigns)
        logger.info(f"    {profile}: {len(profile_campaigns)} campaigns, avg KPI={avg_kpi:.2f}")

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving dataset to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(campaigns, f, indent=2)

    logger.info("✅ Synthetic data generation complete!")
    logger.info(f"\nNext step: python3 scripts/train_rl_selector.py --data {output_path}")


if __name__ == "__main__":
    main()

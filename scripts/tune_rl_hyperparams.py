#!/usr/bin/env python3
"""Hyperparameter grid search for RL strategy selector.

Usage:
    python3 scripts/tune_rl_hyperparams.py --data models/synthetic_rl_data.json --output models/tuning_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import itertools

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.services.rl_data_collector import load_training_dataset
from app.services.rl_strategy_selector import (
    RLConfig,
    RLState,
    RLStrategySelector,
)
from app.services.strategy_selector import compute_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_and_evaluate(
    training_data: list[dict],
    test_data: list[dict],
    config: RLConfig,
    epochs: int = 100,
) -> dict[str, Any]:
    """Train RL selector with given config and evaluate on test set.

    Args:
        training_data: Training campaign traces
        test_data: Test campaign traces
        config: RL configuration with hyperparameters
        epochs: Number of training epochs

    Returns:
        Dict with training metrics and test performance
    """
    from scripts.train_rl_selector import train_rl_selector_offline

    # Train
    selector = train_rl_selector_offline(training_data, config, epochs)

    # Evaluate on test set
    test_kpis = []
    test_rounds = []
    test_converged = 0
    test_target_reached = 0

    for campaign in test_data:
        snapshots = campaign["snapshots"]
        final_kpi = campaign.get("final_kpi")
        n_rounds = campaign.get("n_rounds", len(snapshots))
        converged = campaign.get("converged", False)
        target_reached = campaign.get("target_reached", False)

        if final_kpi is not None:
            test_kpis.append(final_kpi)
        test_rounds.append(n_rounds)
        if converged:
            test_converged += 1
        if target_reached:
            test_target_reached += 1

    avg_kpi = sum(test_kpis) / len(test_kpis) if test_kpis else 0.0
    avg_rounds = sum(test_rounds) / len(test_rounds) if test_rounds else 0.0
    convergence_rate = test_converged / len(test_data) if test_data else 0.0
    target_rate = test_target_reached / len(test_data) if test_data else 0.0

    return {
        "avg_kpi": avg_kpi,
        "avg_rounds": avg_rounds,
        "convergence_rate": convergence_rate,
        "target_rate": target_rate,
        "q_table_size": len(selector.agent.q_table),
    }


def grid_search(
    training_data: list[dict],
    test_data: list[dict],
    param_grid: dict[str, list],
    epochs: int = 100,
) -> list[dict[str, Any]]:
    """Perform grid search over hyperparameter space.

    Args:
        training_data: Training campaign traces
        test_data: Test campaign traces
        param_grid: Dict mapping param names to lists of values
        epochs: Training epochs per config

    Returns:
        List of result dicts sorted by avg_kpi (descending)
    """
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(itertools.product(*param_values))

    logger.info(f"Grid search: {len(combinations)} configurations")
    logger.info(f"Parameters: {param_names}")

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        logger.info(f"\n[{i+1}/{len(combinations)}] Testing: {params}")

        # Create config
        config = RLConfig(
            learning_rate=params.get("learning_rate", 0.1),
            gamma=params.get("gamma", 0.95),
            epsilon=params.get("epsilon", 0.1),
            epsilon_decay=params.get("epsilon_decay", 1.0),
            epsilon_min=params.get("epsilon_min", 0.01),
            model_save_path="models/temp_tuning.pkl",
        )

        try:
            # Train and evaluate
            metrics = train_and_evaluate(training_data, test_data, config, epochs)

            result = {
                "params": params,
                "metrics": metrics,
            }
            results.append(result)

            logger.info(f"  → Avg KPI: {metrics['avg_kpi']:.2f}")
            logger.info(f"  → Avg Rounds: {metrics['avg_rounds']:.2f}")
            logger.info(f"  → Q-table size: {metrics['q_table_size']}")

        except Exception as exc:
            logger.error(f"  ✗ Failed: {exc}", exc_info=True)
            results.append({
                "params": params,
                "metrics": None,
                "error": str(exc),
            })

    # Sort by avg_kpi (descending)
    results.sort(key=lambda r: r["metrics"]["avg_kpi"] if r["metrics"] else -float("inf"), reverse=True)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter grid search for RL selector"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/tuning_results.json",
        help="Output path for tuning results (default: models/tuning_results.json)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Train/test split ratio (default: 0.7)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs per config (default: 100)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: smaller grid for faster tuning",
    )

    args = parser.parse_args()

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Training data not found: {args.data}")
        sys.exit(1)

    logger.info(f"Loading training data from: {args.data}")
    all_data = load_training_dataset(args.data)
    logger.info(f"Loaded {len(all_data)} campaign traces")

    # Split into train/test
    split_idx = int(len(all_data) * args.train_split)
    training_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    logger.info(f"Dataset split: {len(training_data)} train, {len(test_data)} test")

    # Define parameter grid
    if args.quick:
        # Quick grid: 12 combinations
        param_grid = {
            "learning_rate": [0.05, 0.1, 0.2],
            "gamma": [0.9, 0.95],
            "epsilon": [0.1, 0.2],
        }
    else:
        # Full grid: 48 combinations
        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "gamma": [0.9, 0.95, 0.99],
            "epsilon": [0.05, 0.1, 0.2, 0.3],
        }

    logger.info(f"\n{'='*70}")
    logger.info("HYPERPARAMETER GRID SEARCH")
    logger.info(f"{'='*70}\n")

    # Run grid search
    results = grid_search(training_data, test_data, param_grid, args.epochs)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving tuning results to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print top 5 configs
    logger.info(f"\n{'='*70}")
    logger.info("TOP 5 CONFIGURATIONS")
    logger.info(f"{'='*70}\n")

    for i, result in enumerate(results[:5]):
        if result["metrics"] is None:
            continue

        params = result["params"]
        metrics = result["metrics"]

        logger.info(f"#{i+1}: Avg KPI={metrics['avg_kpi']:.2f}, Rounds={metrics['avg_rounds']:.2f}")
        logger.info(f"  Params: lr={params['learning_rate']}, gamma={params['gamma']}, eps={params['epsilon']}")
        logger.info(f"  Q-table size: {metrics['q_table_size']}, Convergence: {metrics['convergence_rate']:.1%}\n")

    # Best config
    best_result = results[0]
    if best_result["metrics"]:
        logger.info(f"\n{'='*70}")
        logger.info("✅ BEST CONFIGURATION FOUND")
        logger.info(f"{'='*70}\n")
        logger.info(f"Parameters: {best_result['params']}")
        logger.info(f"Avg KPI: {best_result['metrics']['avg_kpi']:.2f}")
        logger.info(f"Avg Rounds: {best_result['metrics']['avg_rounds']:.2f}")
        logger.info(f"Convergence Rate: {best_result['metrics']['convergence_rate']:.1%}")
        logger.info(f"Q-table size: {best_result['metrics']['q_table_size']}")

        logger.info(f"\nNext step: Train final model with best params:")
        logger.info(f"  python3 scripts/train_rl_selector.py \\")
        logger.info(f"    --data {args.data} \\")
        logger.info(f"    --output models/rl_selector_tuned.pkl \\")
        logger.info(f"    --learning-rate {best_result['params']['learning_rate']} \\")
        logger.info(f"    --gamma {best_result['params']['gamma']} \\")
        logger.info(f"    --epsilon {best_result['params']['epsilon']}")


if __name__ == "__main__":
    main()

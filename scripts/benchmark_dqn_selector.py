#!/usr/bin/env python3
"""Benchmark DQN strategy selector vs rule-based selector.

Usage:
    python3 scripts/benchmark_dqn_selector.py --model models/dqn_selector_v1.pth --data models/synthetic_rl_data_large.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.services.rl_data_collector import load_training_dataset
from app.services.dqn_strategy_selector import DQNConfig, DQNStrategySelector
from app.services.strategy_selector import (
    compute_diagnostics,
    select_strategy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def benchmark_selector(
    selector_fn: callable,
    test_data: list[dict],
    name: str = "Selector",
) -> dict[str, Any]:
    """Benchmark a strategy selector on test data.

    Args:
        selector_fn: Function that takes (snapshot, diagnostics) and returns (action, backend)
        test_data: List of campaign traces
        name: Name for logging

    Returns:
        Dict with metrics: avg_kpi, avg_rounds, strategy_switches, etc.
    """
    logger.info(f"Benchmarking {name} on {len(test_data)} campaigns...")

    final_kpis = []
    total_rounds = []
    strategy_switches = []
    converged_count = 0
    target_reached_count = 0

    for campaign in test_data:
        snapshots = campaign["snapshots"]
        final_kpi = campaign.get("final_kpi")
        n_rounds = campaign.get("n_rounds", len(snapshots))
        converged = campaign.get("converged", False)
        target_reached = campaign.get("target_reached", False)

        if final_kpi is not None:
            final_kpis.append(final_kpi)
        total_rounds.append(n_rounds)
        if converged:
            converged_count += 1
        if target_reached:
            target_reached_count += 1

        # Count strategy switches (using selector predictions)
        predicted_backends = []
        for snapshot in snapshots:
            diagnostics = compute_diagnostics(snapshot)
            try:
                _, backend = selector_fn(snapshot, diagnostics)
                predicted_backends.append(backend)
            except Exception:
                predicted_backends.append("unknown")

        # Count switches
        switches = sum(
            1 for i in range(1, len(predicted_backends))
            if predicted_backends[i] != predicted_backends[i - 1]
        )
        strategy_switches.append(switches)

    # Compute metrics
    avg_kpi = sum(final_kpis) / len(final_kpis) if final_kpis else 0.0
    avg_rounds = sum(total_rounds) / len(total_rounds) if total_rounds else 0.0
    avg_switches = sum(strategy_switches) / len(strategy_switches) if strategy_switches else 0.0
    convergence_rate = converged_count / len(test_data) if test_data else 0.0
    target_rate = target_reached_count / len(test_data) if test_data else 0.0

    return {
        "name": name,
        "n_campaigns": len(test_data),
        "avg_final_kpi": avg_kpi,
        "avg_rounds": avg_rounds,
        "avg_strategy_switches": avg_switches,
        "convergence_rate": convergence_rate,
        "target_reached_rate": target_rate,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DQN selector vs rule-based selector"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained DQN model (.pth)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to test dataset JSON",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Train/test split ratio (default: 0.7 = 70%% train, 30%% test)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {args.model}")
        logger.info("Train a model first with train_dqn_selector.py")
        sys.exit(1)

    logger.info(f"Loading trained model from: {args.model}")

    try:
        config = DQNConfig(model_save_path=str(model_path))
        dqn_selector = DQNStrategySelector(config)
        dqn_selector.load()
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}", exc_info=True)
        sys.exit(1)

    # Load test data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Test data not found: {args.data}")
        sys.exit(1)

    logger.info(f"Loading test data from: {args.data}")

    try:
        all_data = load_training_dataset(args.data)
    except Exception as exc:
        logger.error(f"Failed to load test data: {exc}", exc_info=True)
        sys.exit(1)

    # Split into train/test
    split_idx = int(len(all_data) * args.train_split)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    logger.info(f"Dataset split: {len(train_data)} train, {len(test_data)} test")

    if not test_data:
        logger.warning("No test data available (dataset too small)")
        logger.info("Using all data for evaluation")
        test_data = all_data

    # Define selector functions
    def dqn_selector_fn(snapshot, diagnostics):
        action, backend = dqn_selector.select_action(snapshot, diagnostics, explore=False)
        return action, backend

    def rule_based_selector_fn(snapshot, diagnostics):
        decision = select_strategy(snapshot, diagnostics)
        # Map backend name to action (approximate)
        action = 1  # Default: exploit
        if "lhs" in decision.backend_name.lower() or "random" in decision.backend_name.lower():
            action = 0  # explore
        elif "cmaes" in decision.backend_name.lower() or "de" in decision.backend_name.lower():
            action = 2  # refine
        return action, decision.backend_name

    # Benchmark both selectors
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 70)

    dqn_metrics = benchmark_selector(dqn_selector_fn, test_data, name="DQN Selector")
    rule_metrics = benchmark_selector(rule_based_selector_fn, test_data, name="Rule-Based Selector")

    # Print comparison
    print("\n┌─────────────────────────────────┬───────────────────┬───────────────────┐")
    print("│ Metric                          │ DQN Selector      │ Rule-Based        │")
    print("├─────────────────────────────────┼───────────────────┼───────────────────┤")
    print(f"│ Avg Final KPI                   │ {dqn_metrics['avg_final_kpi']:17.4f} │ {rule_metrics['avg_final_kpi']:17.4f} │")
    print(f"│ Avg Rounds                      │ {dqn_metrics['avg_rounds']:17.2f} │ {rule_metrics['avg_rounds']:17.2f} │")
    print(f"│ Avg Strategy Switches           │ {dqn_metrics['avg_strategy_switches']:17.2f} │ {rule_metrics['avg_strategy_switches']:17.2f} │")
    print(f"│ Convergence Rate                │ {dqn_metrics['convergence_rate']:16.1%} │ {rule_metrics['convergence_rate']:16.1%} │")
    print(f"│ Target Reached Rate             │ {dqn_metrics['target_reached_rate']:16.1%} │ {rule_metrics['target_reached_rate']:16.1%} │")
    print("└─────────────────────────────────┴───────────────────┴───────────────────┘")

    # Compute improvements
    kpi_improvement = (dqn_metrics['avg_final_kpi'] - rule_metrics['avg_final_kpi']) / abs(rule_metrics['avg_final_kpi']) * 100 if rule_metrics['avg_final_kpi'] != 0 else 0
    rounds_reduction = (rule_metrics['avg_rounds'] - dqn_metrics['avg_rounds']) / rule_metrics['avg_rounds'] * 100 if rule_metrics['avg_rounds'] != 0 else 0

    print(f"\n📊 Improvement Summary:")
    print(f"  • KPI improvement: {kpi_improvement:+.2f}%")
    print(f"  • Rounds reduction: {rounds_reduction:+.2f}%")

    if kpi_improvement > 5 or rounds_reduction > 10:
        print("\n✅ DQN selector shows significant improvement!")
        print("   Ready for production deployment consideration")
    elif kpi_improvement > 0 or rounds_reduction > 0:
        print("\n⚠️  DQN selector shows marginal improvement")
        print("   Consider more training or hyperparameter tuning")
    else:
        print("\n⚠️  DQN selector needs more training or tuning")
        print("   Try: larger dataset, different architecture, reward shaping")

    logger.info("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()

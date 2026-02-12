#!/usr/bin/env python3
"""Train RL strategy selector offline from historical campaign data.

Usage:
    python3 scripts/train_rl_selector.py --data models/rl_training_data.json --output models/rl_selector_v1.pkl
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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


def train_rl_selector_offline(
    training_data: list[dict],
    config: RLConfig | None = None,
    epochs: int = 100,
) -> RLStrategySelector:
    """Train RL selector offline from historical campaigns.

    Args:
        training_data: List of campaign traces from collect_historical_campaigns()
        config: RL configuration (uses defaults if None)
        epochs: Number of training epochs (passes through dataset)

    Returns:
        Trained RLStrategySelector
    """
    if config is None:
        config = RLConfig()

    selector = RLStrategySelector(config)

    logger.info(f"Starting offline training with {len(training_data)} campaigns")
    logger.info(f"Training for {epochs} epochs")

    # Collect all transitions from all campaigns
    all_transitions = []

    for campaign in training_data:
        snapshots = campaign["snapshots"]
        actions = campaign["actions"]
        rewards = campaign["rewards"]

        # Build transitions
        for i in range(len(snapshots) - 1):
            snapshot = snapshots[i]
            diagnostics = compute_diagnostics(snapshot)
            state = RLState.from_snapshot(snapshot, diagnostics)

            next_snapshot = snapshots[i + 1]
            next_diagnostics = compute_diagnostics(next_snapshot)
            next_state = RLState.from_snapshot(next_snapshot, next_diagnostics)

            action = actions[i]
            reward = rewards[i]
            done = (i == len(snapshots) - 2)  # Last transition

            all_transitions.append((state, action, reward, next_state, done))

    logger.info(f"Collected {len(all_transitions)} transitions for training")

    # Train for multiple epochs
    for epoch in range(epochs):
        # Shuffle transitions each epoch (simple random shuffle)
        import random
        random.shuffle(all_transitions)

        # Train on all transitions
        epoch_loss = 0.0
        for state, action, reward, next_state, done in all_transitions:
            # Get Q-value before update
            state_key = selector.agent._discretize_state(state)
            q_before = selector.agent.q_table[state_key][action]

            # Update
            selector.agent.update(state, action, reward, next_state, done)

            # Get Q-value after update
            q_after = selector.agent.q_table[state_key][action]

            # Track loss (TD error)
            epoch_loss += abs(q_after - q_before)

        avg_loss = epoch_loss / len(all_transitions) if all_transitions else 0.0

        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}: avg_loss={avg_loss:.6f}")

    logger.info("Training complete!")
    logger.info(f"Final Q-table size: {len(selector.agent.q_table)} states")

    # Add all transitions to replay buffer (for potential future use)
    for state, action, reward, next_state, done in all_transitions[:1000]:  # Keep last 1000
        selector.learn_from_experience(state, action, reward, next_state, done)

    return selector


def main():
    parser = argparse.ArgumentParser(
        description="Train RL strategy selector from historical data"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training dataset JSON (from collect_rl_data.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/rl_selector_v1.pkl",
        help="Output path for trained model (default: models/rl_selector_v1.pkl)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor (default: 0.95)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate (default: 0.1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load training data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Training data not found: {args.data}")
        logger.info("Run collect_rl_data.py first to collect training data")
        sys.exit(1)

    logger.info(f"Loading training data from: {args.data}")

    try:
        training_data = load_training_dataset(args.data)
    except Exception as exc:
        logger.error(f"Failed to load training data: {exc}", exc_info=True)
        sys.exit(1)

    logger.info(f"Loaded {len(training_data)} campaign traces")

    # Create config with custom hyperparameters
    output_path = Path(args.output)
    config = RLConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        model_save_path=str(output_path),
        replay_save_path=str(output_path.parent / f"{output_path.stem}_replay.pkl"),
    )

    logger.info("Training configuration:")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Gamma: {config.gamma}")
    logger.info(f"  Epsilon: {config.epsilon}")

    # Train
    try:
        selector = train_rl_selector_offline(
            training_data=training_data,
            config=config,
            epochs=args.epochs,
        )
    except Exception as exc:
        logger.error(f"Training failed: {exc}", exc_info=True)
        sys.exit(1)

    # Save trained model
    logger.info(f"Saving trained model to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        selector.save()
    except Exception as exc:
        logger.error(f"Failed to save model: {exc}", exc_info=True)
        sys.exit(1)

    logger.info("✅ Training complete!")
    logger.info(f"Model saved to: {output_path}")
    logger.info(f"Replay buffer saved to: {config.replay_save_path}")
    logger.info(f"\nNext step: python3 scripts/benchmark_rl_selector.py --model {output_path}")


if __name__ == "__main__":
    main()

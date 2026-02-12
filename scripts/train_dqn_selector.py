#!/usr/bin/env python3
"""Train DQN strategy selector offline from historical campaign data.

Usage:
    python3 scripts/train_dqn_selector.py --data models/synthetic_rl_data_large.json --output models/dqn_selector_v1.pth
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
from app.services.dqn_strategy_selector import DQNConfig, DQNStrategySelector
from app.services.rl_strategy_selector import RLState
from app.services.strategy_selector import compute_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_dqn_selector_offline(
    training_data: list[dict],
    config: DQNConfig | None = None,
    epochs: int = 100,
) -> DQNStrategySelector:
    """Train DQN selector offline from historical campaigns.

    Args:
        training_data: List of campaign traces from collect_historical_campaigns()
        config: DQN configuration (uses defaults if None)
        epochs: Number of training epochs (passes through dataset)

    Returns:
        Trained DQNStrategySelector
    """
    if config is None:
        config = DQNConfig()

    selector = DQNStrategySelector(config)

    logger.info(f"Starting offline DQN training with {len(training_data)} campaigns")
    logger.info(f"Training for {epochs} epochs")
    logger.info(f"Network architecture: {config.hidden_dims}")

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

    # Pre-fill replay buffer with all transitions
    logger.info("Filling replay buffer...")
    for state, action, reward, next_state, done in all_transitions:
        selector.learn_from_experience(state, action, reward, next_state, done)

    logger.info(f"Replay buffer size: {len(selector.agent.replay_buffer)}")

    # Train for multiple epochs
    total_loss = 0.0
    total_steps = 0

    for epoch in range(epochs):
        # Shuffle transitions each epoch
        import random
        random.shuffle(all_transitions)

        epoch_loss = 0.0
        epoch_steps = 0

        # Train on all transitions
        for state, action, reward, next_state, done in all_transitions:
            loss = selector.learn_from_experience(state, action, reward, next_state, done)

            if loss is not None:
                epoch_loss += loss
                epoch_steps += 1
                total_steps += 1

        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0

        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"avg_loss={avg_epoch_loss:.6f}, "
                f"epsilon={selector.agent.epsilon:.4f}"
            )

        total_loss += epoch_loss

    avg_total_loss = total_loss / total_steps if total_steps > 0 else 0.0

    logger.info("Training complete!")
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Final avg loss: {avg_total_loss:.6f}")
    logger.info(f"Final epsilon: {selector.agent.epsilon:.4f}")
    logger.info(f"Replay buffer size: {len(selector.agent.replay_buffer)}")

    return selector


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN strategy selector from historical data"
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
        default="models/dqn_selector_v1.pth",
        help="Output path for trained model (default: models/dqn_selector_v1.pth)",
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
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor (default: 0.95)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="64,32",
        help="Hidden layer dimensions (default: 64,32)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
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

    # Parse hidden dims
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]

    # Create config with custom hyperparameters
    output_path = Path(args.output)
    config = DQNConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        hidden_dims=hidden_dims,
        batch_size=args.batch_size,
        model_save_path=str(output_path),
        replay_save_path=str(output_path.parent / f"{output_path.stem}_replay.pkl"),
    )

    logger.info("Training configuration:")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Gamma: {config.gamma}")
    logger.info(f"  Hidden dims: {config.hidden_dims}")
    logger.info(f"  Batch size: {config.batch_size}")

    # Train
    try:
        selector = train_dqn_selector_offline(
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
    logger.info(f"\nNext step: python3 scripts/benchmark_dqn_selector.py --model {output_path}")


if __name__ == "__main__":
    main()

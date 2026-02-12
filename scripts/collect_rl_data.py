#!/usr/bin/env python3
"""Collect training data from historical campaigns for RL strategy selector.

Usage:
    python3 scripts/collect_rl_data.py --db otbot.db --output models/rl_training_data.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.services.rl_data_collector import (
    collect_historical_campaigns,
    save_training_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Collect RL training data from campaign database"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="otbot.db",
        help="Path to campaign database (default: otbot.db)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/rl_training_data.json",
        help="Output path for training dataset (default: models/rl_training_data.json)",
    )
    parser.add_argument(
        "--min-rounds",
        type=int,
        default=3,
        help="Minimum rounds required for campaign to be included (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check if database exists
    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database not found: {args.db}")
        logger.info("Please provide a valid path to campaign_state.db or otbot.db")
        sys.exit(1)

    logger.info(f"Collecting campaigns from: {args.db}")
    logger.info(f"Minimum rounds required: {args.min_rounds}")

    # Collect historical campaigns
    try:
        traces = collect_historical_campaigns(
            db_path=args.db,
            min_rounds=args.min_rounds,
        )
    except Exception as exc:
        logger.error(f"Failed to collect campaigns: {exc}", exc_info=True)
        sys.exit(1)

    if not traces:
        logger.warning("No valid campaigns found in database")
        logger.info("Make sure campaigns table has completed campaigns with rounds data")
        sys.exit(1)

    logger.info(f"Collected {len(traces)} valid campaign traces")

    # Print statistics
    total_rounds = sum(t["n_rounds"] for t in traces)
    avg_rounds = total_rounds / len(traces) if traces else 0
    converged_count = sum(1 for t in traces if t.get("converged", False))
    target_reached_count = sum(1 for t in traces if t.get("target_reached", False))

    logger.info(f"Total rounds: {total_rounds}")
    logger.info(f"Average rounds per campaign: {avg_rounds:.2f}")
    logger.info(f"Converged campaigns: {converged_count}/{len(traces)} ({converged_count/len(traces)*100:.1f}%)")
    logger.info(f"Target reached: {target_reached_count}/{len(traces)} ({target_reached_count/len(traces)*100:.1f}%)")

    # Save to file
    output_path = Path(args.output)
    logger.info(f"Saving training dataset to: {output_path}")

    try:
        save_training_dataset(traces, output_path=str(output_path))
    except Exception as exc:
        logger.error(f"Failed to save dataset: {exc}", exc_info=True)
        sys.exit(1)

    logger.info("✅ Data collection complete!")
    logger.info(f"Next step: python3 scripts/train_rl_selector.py --data {output_path}")


if __name__ == "__main__":
    main()

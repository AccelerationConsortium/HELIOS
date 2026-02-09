"""CLI entry point for OTbot Offline Benchmark Framework.

Usage:
    python -m benchmarks                          # Run all scenarios
    python -m benchmarks --category c2            # Run C2 scenarios only
    python -m benchmarks --scenario c2_latency    # Run specific scenario
    python -m benchmarks --seed 123 --verbose     # Custom seed + verbose
"""
from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OTbot Offline Benchmark Framework",
    )
    parser.add_argument(
        "--category",
        choices=["c2", "c3", "c4", "c5", "fault", "intelligence", "all"],
        default="all",
        help="Run scenarios from a specific category (default: all)",
    )
    parser.add_argument(
        "--scenario",
        help="Run a specific scenario by ID",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for reports (default: benchmark_results)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Configure logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Import scenarios (triggers registration)
    from benchmarks.scenarios import get_scenarios
    from benchmarks.runner import BenchmarkRunner
    from benchmarks.reporter import Reporter

    # Select scenarios
    category = args.category if args.category != "all" else None
    scenarios = get_scenarios(category)

    if args.scenario:
        scenarios = [s for s in scenarios if s.id == args.scenario]
        if not scenarios:
            print(f"Error: scenario '{args.scenario}' not found", file=sys.stderr)
            sys.exit(2)

    if not scenarios:
        print("No scenarios to run", file=sys.stderr)
        sys.exit(2)

    print(f"Running {len(scenarios)} benchmark scenario(s) with seed={args.seed}")

    # Run benchmarks
    runner = BenchmarkRunner(
        scenarios=scenarios,
        seed=args.seed,
        verbose=args.verbose,
    )
    report = runner.run_all()

    # Generate reports
    reporter = Reporter(output_dir=args.output_dir)
    json_path, md_path = reporter.generate(report)

    # Print summary
    print(f"\nResults: {report.passed}/{report.total} passed "
          f"({report.failed} failed) in {report.duration_s:.1f}s")
    print(f"Reports: {json_path}, {md_path}")

    # Exit code: 0 if all pass, 1 if any fail
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()

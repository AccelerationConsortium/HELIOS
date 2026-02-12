#!/usr/bin/env python3
"""
Run instrumented experiment with full log-decision-recovery pipeline.

Usage:
    python -m exp_agent.cli.run_instrumented --fault-mode overshoot --target-temp 120
    python -m exp_agent.cli.run_instrumented --fault-mode sensor_fail --log-dir ./logs
    python -m exp_agent.cli.run_instrumented --fault-mode timeout --verbose
"""
import argparse
import sys
from pathlib import Path

from ..orchestrator.instrumented_supervisor import InstrumentedSupervisor
from ..logging.pipeline import LogLevel


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment with log-decision-recovery pipeline"
    )
    parser.add_argument(
        "--fault-mode",
        choices=["none", "random", "timeout", "overshoot", "sensor_fail"],
        default="overshoot",
        help="Fault injection mode (default: overshoot)"
    )
    parser.add_argument(
        "--target-temp",
        type=float,
        default=120.0,
        help="Target temperature in °C (default: 120)"
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment identifier (default: auto-generated)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for log files (optional)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only show warnings and errors"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run experiment and show detailed analysis"
    )

    args = parser.parse_args()

    # Determine log level
    if args.verbose:
        log_level = LogLevel.DEBUG
    elif args.quiet:
        log_level = LogLevel.WARNING
    else:
        log_level = LogLevel.INFO

    print("=" * 70)
    print("LOG-DECISION-RECOVERY PIPELINE")
    print("=" * 70)
    print(f"Target Temperature: {args.target_temp}°C")
    print(f"Fault Mode:         {args.fault_mode}")
    print(f"Log Level:          {log_level.value}")
    if args.log_dir:
        print(f"Log Directory:      {args.log_dir}")
    print("=" * 70)
    print()

    # Create and run supervisor
    supervisor = InstrumentedSupervisor(
        target_temp=args.target_temp,
        fault_mode=args.fault_mode,
        experiment_id=args.experiment_id,
        log_dir=args.log_dir,
        log_level=log_level,
    )

    try:
        success = supervisor.run()
    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
        supervisor.shutdown()
        success = False

    # Analysis
    print()
    print("=" * 70)
    print("DECISION ANALYSIS SUMMARY")
    print("=" * 70)

    summary = supervisor.get_decision_summary()

    if summary.get("total_decisions", 0) == 0:
        print("No recovery decisions were made (clean run).")
    else:
        print(f"Total Decisions:        {summary['total_decisions']}")
        print(f"Success Rate:           {summary['success_rate']:.1%}")
        print(f"Avg Recovery Duration:  {summary['avg_recovery_duration_ms']:.0f}ms")
        print()
        print("By Decision Kind:")
        for kind, count in summary.get("by_kind", {}).items():
            print(f"  {kind:12} {count}")
        print()
        print("By Error Type:")
        for err_type, count in summary.get("by_error_type", {}).items():
            print(f"  {err_type:20} {count}")
        print()
        print("By Fault Signature:")
        for sig, count in summary.get("by_signature", {}).items():
            print(f"  {sig:12} {count}")

    # Detailed trails
    if args.analyze_only or args.verbose:
        analyzer = supervisor.get_analyzer()
        trails = analyzer.get_all_trails()

        if trails:
            print()
            print("=" * 70)
            print("DETAILED DECISION TRAILS")
            print("=" * 70)

            for i, trail in enumerate(trails[:5]):  # Show first 5
                print(f"\n--- Trail {i+1}: {trail.correlation_id} ---")
                print(f"  Error:      {trail.error_type} ({trail.error_severity})")
                print(f"  Signature:  {trail.signature_mode} (confidence={trail.signature_confidence:.2f})")
                print(f"  Decision:   {trail.decision_kind}")
                print(f"  Rationale:  {trail.decision_rationale}")
                print(f"  Actions:    {' → '.join(trail.recovery_actions) or 'none'}")
                print(f"  Success:    {'✓' if trail.recovery_success else '✗'}")
                print(f"  Duration:   {trail.total_duration_ms:.0f}ms")

            if len(trails) > 5:
                print(f"\n... and {len(trails) - 5} more trails")

    print()
    print("=" * 70)
    print(f"Experiment Result: {'SUCCESS ✓' if success else 'FAILED ✗'}")
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

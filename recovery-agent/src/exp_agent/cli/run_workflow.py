"""
Enhanced CLI that supports workflow-aware execution.

This extends the basic agent CLI to support experiment workflows with
multiple loops and step dependencies.
"""

import argparse
import sys
from pathlib import Path

from ..orchestrator.workflow_manager import WorkflowManager, create_sample_prep_workflow
from ..recovery.workflow_policy import WorkflowRecoveryPolicy


def main():
    parser = argparse.ArgumentParser(
        description="Run the Experiment Agent with Workflow Support"
    )

    # Basic options
    parser.add_argument(
        "--simulation", action="store_true", help="Run in simulation mode (default)"
    )
    parser.add_argument(
        "--real-hardware", action="store_true", help="Run with real hardware devices"
    )

    # Workflow options
    parser.add_argument(
        "--workflow",
        type=str,
        default="sample_prep",
        choices=["sample_prep", "custom"],
        help="Predefined workflow to run",
    )
    parser.add_argument(
        "--workflow-config", type=str, help="Path to custom workflow configuration file"
    )
    parser.add_argument(
        "--max-loops", type=int, default=3, help="Maximum number of experiment loops"
    )

    # Device options (for real hardware)
    parser.add_argument(
        "--device-config", type=str, help="Path to device configuration file"
    )

    # Output options
    parser.add_argument(
        "--output-results", type=str, help="Path to save experiment results JSON"
    )

    args = parser.parse_args()

    # Set random seed if provided
    if hasattr(args, "seed") and args.seed is not None:
        import random

        random.seed(args.seed)
        print(f"Random Seed: {args.seed}")

    print("=== Experiment Agent with Workflow Support ===")
    print(f"Mode: {'Real Hardware' if args.real_hardware else 'Simulation'}")
    print(f"Workflow: {args.workflow}")
    print(f"Max Loops: {args.max_loops}")

    try:
        # Create workflow
        if args.workflow == "sample_prep":
            workflow_steps = create_sample_prep_workflow()
        elif args.workflow == "custom" and args.workflow_config:
            workflow_steps = load_custom_workflow(args.workflow_config)
        else:
            print("Error: Must specify workflow or provide custom workflow config")
            sys.exit(1)

        print(f"Workflow loaded with {len(workflow_steps)} steps:")
        for i, step in enumerate(workflow_steps):
            print(
                f"  {i + 1}. {step.name} ({step.phase.value}, {step.dependency.value})"
            )

        # Create recovery policy
        recovery_policy = WorkflowRecoveryPolicy()

        # Create workflow manager
        workflow_manager = WorkflowManager(
            workflow_steps=workflow_steps, recovery_policy=recovery_policy
        )

        # Run experiment
        print("\nStarting experiment execution...")
        results = workflow_manager.run_experiment(max_loops=args.max_loops)

        # Print results
        print("\n=== Experiment Results ===")
        for result in results:
            print(f"\nLoop {result.loop_id}:")
            print(f"  Status: {result.status}")
            print(f"  Duration: {result.duration:.1f}s")
            print(f"  Completed steps: {len(result.completed_steps)}")
            print(f"  Failed steps: {len(result.failed_steps)}")
            print(f"  Skipped steps: {len(result.skipped_steps)}")

            if result.failed_steps:
                print("  Failures:")
                for step, reason in result.failed_steps.items():
                    print(f"    - {step}: {reason}")

        # Save results if requested
        if args.output_results:
            import json

            output_data = {
                "experiment_summary": {
                    "total_loops": len(results),
                    "successful_loops": sum(
                        1 for r in results if r.status == "completed"
                    ),
                    "total_duration": sum(r.duration for r in results),
                },
                "loop_results": [
                    {
                        "loop_id": r.loop_id,
                        "status": r.status,
                        "duration": r.duration,
                        "completed_steps": r.completed_steps,
                        "failed_steps": r.failed_steps,
                        "skipped_steps": r.skipped_steps,
                    }
                    for r in results
                ],
            }

            with open(args.output_results, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output_results}")

        # Overall success check
        successful_loops = sum(1 for r in results if r.status == "completed")
        if successful_loops > 0:
            print(f"\n✓ Experiment completed with {successful_loops} successful loops!")
        else:
            print("\n✗ Experiment failed - no successful loops")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def load_custom_workflow(config_path: str):
    """Load custom workflow from configuration file."""
    import json
    from ..orchestrator.workflow_manager import (
        WorkflowStep,
        WorkflowPhase,
        StepDependency,
    )

    with open(config_path, "r") as f:
        config = json.load(f)

    steps = []
    for step_config in config["steps"]:
        step = WorkflowStep(
            name=step_config["name"],
            phase=WorkflowPhase(step_config["phase"]),
            devices=step_config.get("devices", []),
            dependency=StepDependency(step_config.get("dependency", "none")),
            can_skip=step_config.get("can_skip", False),
            max_retries=step_config.get("max_retries", 3),
            timeout=step_config.get("timeout", 300.0),
            preconditions=step_config.get("preconditions", []),
            postconditions=step_config.get("postconditions", []),
        )
        steps.append(step)

    return steps


if __name__ == "__main__":
    main()

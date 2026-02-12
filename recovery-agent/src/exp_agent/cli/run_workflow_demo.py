"""
CLI entry point: demonstrate WorkflowSupervisor with a 6-step plan.

Usage:
    python -m exp_agent.cli.run_workflow_demo --fault-mode overshoot
    python -m exp_agent.cli.run_workflow_demo --fault-mode timeout
    python -m exp_agent.cli.run_workflow_demo --fault-mode sensor_fail
    python -m exp_agent.cli.run_workflow_demo --fault-mode none
"""
import argparse
import random

from ..core.types import PlanStep, Action
from ..devices.simulated.heater import SimHeater
from ..orchestrator.workflow_supervisor import WorkflowSupervisor


def build_plan(target_temp: float = 120.0) -> list[PlanStep]:
    """
    Build a 6-step experiment plan:
      1. setup       — initialize heater (critical)
      2. preheat     — warm to target (critical)
      3. log_snapshot — take a telemetry snapshot (optional, skip on fail)
      4. hold        — maintain temperature (critical)
      5. measure     — verify measurement (optional, skip on fail)
      6. cooldown    — safe shutdown (critical)
    """
    device = "heater_1"
    return [
        PlanStep(
            step_id="setup",
            stage="setup",
            description="Initialize heater to idle state",
            action=Action(
                name="cool_down", effect="write", device=device,
                postconditions=["telemetry.heating == False"],
            ),
            criticality="critical",
            on_failure="abort",
        ),
        PlanStep(
            step_id="preheat",
            stage="heating",
            description=f"Heat to {target_temp}°C",
            action=Action(
                name="set_temperature", effect="write", device=device,
                params={"temperature": target_temp},
                postconditions=[
                    f"telemetry.target == {target_temp}",
                    f"telemetry.temperature ~= {target_temp} +/- 2.0 within 20s",
                ],
            ),
            criticality="critical",
            on_failure="abort",
            max_retries=2,
        ),
        PlanStep(
            step_id="log_snapshot",
            stage="diagnostics",
            description="Take telemetry snapshot (optional)",
            action=Action(
                name="wait", effect="write", device=device,
                params={"duration": 1},
                postconditions=[
                    f"telemetry.temperature ~= {target_temp} +/- 5.0 within 5s",
                ],
            ),
            criticality="optional",
            on_failure="skip",
        ),
        PlanStep(
            step_id="hold",
            stage="hold",
            description=f"Hold at {target_temp}°C",
            action=Action(
                name="wait", effect="write", device=device,
                params={"duration": 3},
                postconditions=[
                    f"telemetry.temperature ~= {target_temp} +/- 2.0 within 10s",
                ],
            ),
            criticality="critical",
            on_failure="abort",
            max_retries=2,
        ),
        PlanStep(
            step_id="measure",
            stage="measure",
            description="Verify measurement reading (optional)",
            action=Action(
                name="wait", effect="write", device=device,
                params={"duration": 1},
                postconditions=[
                    f"telemetry.temperature ~= {target_temp} +/- 3.0 within 5s",
                ],
            ),
            criticality="optional",
            on_failure="skip",
            max_retries=1,
        ),
        PlanStep(
            step_id="cooldown",
            stage="cooldown",
            description="Cool down to ambient",
            action=Action(
                name="cool_down", effect="write", device=device,
                postconditions=["telemetry.heating == False"],
            ),
            criticality="critical",
            on_failure="abort",
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run Workflow Supervisor Demo")
    parser.add_argument(
        "--fault-mode", type=str, default="none",
        choices=["none", "random", "timeout", "overshoot", "sensor_fail"],
        help="Fault injection mode",
    )
    parser.add_argument("--target-temp", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random Seed: {args.seed}")

    print(f"Configuration: fault_mode={args.fault_mode}, target_temp={args.target_temp}")

    device = SimHeater(name="heater_1", fault_mode=args.fault_mode)
    plan = build_plan(args.target_temp)

    supervisor = WorkflowSupervisor(device=device, target_temp=args.target_temp)
    result = supervisor.execute_plan(plan)

    # Exit code
    exit(0 if result.success else 1)


if __name__ == "__main__":
    main()

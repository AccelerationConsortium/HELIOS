"""LogReplay — Pre-recorded log scenario replay for benchmarking.

Provides deterministic replay of recorded action/result sequences for
validating KPI extraction, reviewer accuracy, and recovery paths.

Key classes:
- ReplayStep: single recorded step (primitive, params, result/error)
- LogScenario: complete recorded scenario with expected KPIs
- ReplayAdapter: feeds recorded results to the execution engine
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReplayStep:
    """A single recorded step in a log scenario.

    If ``error`` is set, the step raises RuntimeError instead of returning.
    """

    primitive: str
    params: dict[str, Any]
    result: dict[str, Any] | None = None  # None when error is set
    error: str | None = None
    duration_s: float = 1.0


@dataclass(frozen=True)
class LogScenario:
    """A complete pre-recorded scenario for replay.

    Contains an ordered sequence of steps and expected KPI values
    for validation after replay.
    """

    name: str
    description: str
    steps: tuple[ReplayStep, ...]
    expected_kpis: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


class ReplayAdapter:
    """Feeds pre-recorded results to the execution engine.

    Steps are consumed in order; raises IndexError if more steps
    are executed than were recorded.
    """

    def __init__(self, scenario: LogScenario) -> None:
        self.scenario = scenario
        self._step_idx = 0
        self.executed: list[ReplayStep] = []

    @property
    def steps_remaining(self) -> int:
        return len(self.scenario.steps) - self._step_idx

    @property
    def steps_executed(self) -> int:
        return self._step_idx

    def execute(self, primitive: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the next recorded step.

        Verifies primitive matches the recorded step (warning if mismatch).
        Returns recorded result or raises recorded error.
        """
        if self._step_idx >= len(self.scenario.steps):
            raise IndexError(
                f"ReplayAdapter exhausted: {self._step_idx} steps recorded, "
                f"but more execution requested"
            )

        step = self.scenario.steps[self._step_idx]
        self._step_idx += 1
        self.executed.append(step)

        # Check primitive matches (allow mismatch with warning for flexibility)
        if step.primitive != primitive:
            import logging
            logging.getLogger(__name__).warning(
                "Replay primitive mismatch at step %d: expected %s, got %s",
                self._step_idx - 1,
                step.primitive,
                primitive,
            )

        if step.error is not None:
            raise RuntimeError(step.error)

        return step.result if step.result is not None else {"status": "ok"}

    def reset(self) -> None:
        """Reset replay to beginning."""
        self._step_idx = 0
        self.executed.clear()


# ---------------------------------------------------------------------------
# Built-in log scenarios for common patterns
# ---------------------------------------------------------------------------


def make_simple_pipetting_scenario(
    n_wells: int = 4,
    volume_ul: float = 100.0,
    noise_pct: float = 0.02,
) -> LogScenario:
    """Create a simple aspirate/dispense scenario for N wells."""
    import random as _rng

    r = _rng.Random(42)
    steps: list[ReplayStep] = [
        ReplayStep(primitive="robot.home", params={}, result={"status": "ok", "homed": True}),
        ReplayStep(
            primitive="robot.load_pipettes",
            params={"pipettes": ["left"]},
            result={"status": "ok", "pipettes_loaded": True},
        ),
        ReplayStep(
            primitive="robot.load_labware",
            params={"labware": "plate1", "slot": "1"},
            result={"status": "ok", "labware": "plate1", "slot": "1"},
        ),
    ]

    for i in range(n_wells):
        well = f"A{i + 1}"
        # Pick up tip
        steps.append(ReplayStep(
            primitive="robot.pick_up_tip",
            params={"pipette": "left"},
            result={"status": "ok", "pipette": "left", "tip": "on"},
        ))
        # Aspirate
        measured_asp = volume_ul * (1.0 + r.gauss(0.0, noise_pct))
        steps.append(ReplayStep(
            primitive="robot.aspirate",
            params={"pipette": "left", "volume_ul": volume_ul,
                    "labware": "plate1", "well": well},
            result={
                "status": "ok",
                "requested_volume_ul": volume_ul,
                "measured_volume_ul": measured_asp,
                "pipette": "left",
                "labware": "plate1",
                "well": well,
            },
        ))
        # Dispense
        measured_disp = volume_ul * (1.0 + r.gauss(0.0, noise_pct))
        steps.append(ReplayStep(
            primitive="robot.dispense",
            params={"pipette": "left", "volume_ul": volume_ul,
                    "labware": "plate1", "well": well},
            result={
                "status": "ok",
                "requested_volume_ul": volume_ul,
                "measured_volume_ul": measured_disp,
                "pipette": "left",
                "labware": "plate1",
                "well": well,
            },
        ))
        # Drop tip
        steps.append(ReplayStep(
            primitive="robot.drop_tip",
            params={"pipette": "left"},
            result={"status": "ok", "pipette": "left", "tip": "off"},
        ))

    return LogScenario(
        name="simple_pipetting",
        description=f"Simple aspirate/dispense across {n_wells} wells",
        steps=tuple(steps),
        expected_kpis={"run_success_rate": 1.0},
        tags=["pipetting", "basic"],
    )


def make_error_recovery_scenario() -> LogScenario:
    """Create a scenario with a failure mid-run for recovery testing."""
    steps = (
        ReplayStep(primitive="robot.home", params={},
                   result={"status": "ok", "homed": True}),
        ReplayStep(primitive="robot.load_pipettes", params={"pipettes": ["left"]},
                   result={"status": "ok", "pipettes_loaded": True}),
        ReplayStep(primitive="robot.load_labware",
                   params={"labware": "plate1", "slot": "1"},
                   result={"status": "ok", "labware": "plate1", "slot": "1"}),
        ReplayStep(primitive="robot.pick_up_tip",
                   params={"pipette": "left"},
                   result={"status": "ok", "pipette": "left", "tip": "on"}),
        # This step fails
        ReplayStep(
            primitive="robot.aspirate",
            params={"pipette": "left", "volume_ul": 100.0,
                    "labware": "plate1", "well": "A1"},
            error="instrument disconnected during aspirate",
        ),
        # Recovery: re-home
        ReplayStep(primitive="robot.home", params={},
                   result={"status": "ok", "homed": True}),
        # Retry aspirate succeeds
        ReplayStep(
            primitive="robot.aspirate",
            params={"pipette": "left", "volume_ul": 100.0,
                    "labware": "plate1", "well": "A1"},
            result={
                "status": "ok",
                "requested_volume_ul": 100.0,
                "measured_volume_ul": 99.5,
                "pipette": "left",
                "labware": "plate1",
                "well": "A1",
            },
        ),
    )
    return LogScenario(
        name="error_recovery",
        description="Aspirate fails mid-run, recovery re-homes and retries",
        steps=steps,
        expected_kpis={"recovery_count": 1},
        tags=["recovery", "fault"],
    )

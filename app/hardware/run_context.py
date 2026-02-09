"""
RunContext - Tracks live experiment state for intelligent recovery

This module maintains real-time state about the current experiment phase,
tools being held, electrode status, and other critical information needed
to determine appropriate recovery actions when failures occur.
"""
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from enum import Enum


class ExperimentPhase(str, Enum):
    """Current phase of the experiment"""
    SAMPLE_PREP = "SAMPLE_PREP"  # Sample preparation: pipetting, mixing, adding reagents
    ELECTROCHEM_RUNNING = "ELECTROCHEM_RUNNING"  # Electrochemical test in progress
    POST_ELECTROCHEM = "POST_ELECTROCHEM"  # Post-test cleanup, washing, data collection
    UNKNOWN = "UNKNOWN"  # Cannot determine phase (error state)


class Payload(str, Enum):
    """Current tool/item held by the OT2 robot"""
    TIP = "TIP"  # Regular pipette tip
    FLUSH_TOOL = "FLUSH_TOOL"  # Flusher tool
    ELECTRODE_HEAD = "ELECTRODE_HEAD"  # Electrode assembly
    NONE = "NONE"  # No tool held


@dataclass
class RunContext:
    """
    Live state tracking for experiment execution

    This context is updated by the orchestrator at every critical step
    to maintain accurate state for recovery decision-making.
    """

    # Current execution position
    current_step_id: Optional[str] = None
    current_phase: ExperimentPhase = ExperimentPhase.UNKNOWN

    # OT2 robot state
    payload: Payload = Payload.NONE
    pipette_has_tip: Dict[str, bool] = field(default_factory=lambda: {
        'left': False,
        'right': False
    })

    # Electrode state
    electrode_inserted: bool = False  # Is electrode currently in reactor
    electrode_contaminated: bool = False  # Has electrode been used (needs cleaning)

    # Loop context
    loop_num: Optional[int] = None
    reactor_well: Optional[str] = None
    vial_pos: Optional[str] = None
    tip_pos: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_step(self, step_id: str, phase: Optional[ExperimentPhase] = None):
        """Update current step and optionally phase"""
        self.current_step_id = step_id
        if phase is not None:
            self.current_phase = phase

    def acquire_payload(self, payload: Payload, pipette: Optional[str] = None):
        """Record that robot has acquired a tool/tip"""
        self.payload = payload
        if payload == Payload.TIP and pipette:
            self.pipette_has_tip[pipette] = True

    def release_payload(self, pipette: Optional[str] = None):
        """Record that robot has released current tool/tip"""
        if self.payload == Payload.TIP and pipette:
            self.pipette_has_tip[pipette] = False
        self.payload = Payload.NONE

    def insert_electrode(self):
        """Record that electrode has been inserted into reactor"""
        self.electrode_inserted = True
        self.electrode_contaminated = True  # Assume contaminated once inserted

    def remove_electrode(self):
        """Record that electrode has been removed from reactor"""
        self.electrode_inserted = False

    def clean_electrode(self):
        """Record that electrode has been cleaned"""
        self.electrode_contaminated = False

    def reset_for_new_loop(self, loop_num: int, reactor_well: str,
                           vial_pos: str = None, tip_pos: str = None):
        """Reset context for a new loop iteration"""
        self.loop_num = loop_num
        self.reactor_well = reactor_well
        self.vial_pos = vial_pos
        self.tip_pos = tip_pos

        # Reset state
        self.current_step_id = None
        self.current_phase = ExperimentPhase.UNKNOWN
        self.payload = Payload.NONE
        self.pipette_has_tip = {'left': False, 'right': False}
        self.electrode_inserted = False
        self.electrode_contaminated = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging"""
        return {
            'current_step_id': self.current_step_id,
            'current_phase': self.current_phase.value if self.current_phase else None,
            'payload': self.payload.value if self.payload else None,
            'pipette_has_tip': self.pipette_has_tip,
            'electrode_inserted': self.electrode_inserted,
            'electrode_contaminated': self.electrode_contaminated,
            'loop_num': self.loop_num,
            'reactor_well': self.reactor_well,
            'vial_pos': self.vial_pos,
            'tip_pos': self.tip_pos,
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        """Human-readable string representation"""
        return (
            f"RunContext("
            f"loop={self.loop_num}, "
            f"phase={self.current_phase.value if self.current_phase else 'UNKNOWN'}, "
            f"payload={self.payload.value if self.payload else 'NONE'}, "
            f"electrode={'IN' if self.electrode_inserted else 'OUT'}"
            f"{'(dirty)' if self.electrode_contaminated else ''}, "
            f"well={self.reactor_well})"
        )

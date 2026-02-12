"""
Workflow Definitions for Lab Automation

Defines the structure of workflows, phases, and steps that can be
executed across multiple instruments.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class StepParams:
    """Parameters for a workflow step."""
    data: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data.get(key)

    def __setitem__(self, key: str, value: Any):
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.data.copy()


@dataclass
class Step:
    """A single step in a workflow phase."""
    step_id: str
    device_type: str  # e.g., "liquid_handler", "potentiostat", "pump_controller"
    action: str  # e.g., "liquid_handler.transfer", "potentiostat.run_eis"
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    # Execution metadata
    depends_on: List[str] = field(default_factory=list)  # step_ids this step depends on
    timeout_seconds: Optional[float] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "step_id": self.step_id,
            "device_type": self.device_type,
            "action": self.action,
            "params": self.params,
            "description": self.description,
        }

        if self.depends_on:
            result["depends_on"] = self.depends_on
        if self.timeout_seconds:
            result["timeout_seconds"] = self.timeout_seconds
        if self.retry_count > 0:
            result["retry_count"] = self.retry_count

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Step':
        """Create Step from dictionary."""
        return cls(
            step_id=data["step_id"],
            device_type=data.get("device_type", "unknown"),
            action=data["action"],
            params=data.get("params", {}),
            description=data.get("description", ""),
            depends_on=data.get("depends_on", []),
            timeout_seconds=data.get("timeout_seconds"),
            retry_count=data.get("retry_count", 0),
        )


@dataclass
class ParallelThread:
    """A thread of steps that can run in parallel with other threads."""
    thread_name: str
    description: str = ""
    steps: List[Step] = field(default_factory=list)

    def add_step(self, step: Step):
        """Add a step to this thread."""
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "thread_name": self.thread_name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParallelThread':
        """Create ParallelThread from dictionary."""
        thread = cls(
            thread_name=data["thread_name"],
            description=data.get("description", ""),
        )
        for step_data in data.get("steps", []):
            thread.add_step(Step.from_dict(step_data))
        return thread


@dataclass
class Phase:
    """
    A phase in a workflow.

    A phase can contain either:
    - A list of sequential steps
    - A list of parallel threads (for concurrent execution)
    """
    phase_name: str
    description: str = ""
    steps: List[Step] = field(default_factory=list)
    parallel_threads: List[ParallelThread] = field(default_factory=list)

    @property
    def is_parallel(self) -> bool:
        """Check if this phase has parallel execution."""
        return len(self.parallel_threads) > 0

    def add_step(self, step: Step):
        """Add a sequential step."""
        if self.is_parallel:
            raise ValueError("Cannot add sequential steps to a parallel phase")
        self.steps.append(step)

    def add_parallel_thread(self, thread: ParallelThread):
        """Add a parallel thread."""
        if self.steps:
            raise ValueError("Cannot add parallel threads to a sequential phase")
        self.parallel_threads.append(thread)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "phase_name": self.phase_name,
            "description": self.description,
        }

        if self.is_parallel:
            result["parallel_threads"] = [t.to_dict() for t in self.parallel_threads]
        else:
            result["steps"] = [s.to_dict() for s in self.steps]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Phase':
        """Create Phase from dictionary."""
        phase = cls(
            phase_name=data["phase_name"],
            description=data.get("description", ""),
        )

        if "parallel_threads" in data:
            for thread_data in data["parallel_threads"]:
                phase.parallel_threads.append(ParallelThread.from_dict(thread_data))
        else:
            for step_data in data.get("steps", []):
                phase.steps.append(Step.from_dict(step_data))

        return phase


@dataclass
class DeviceConfig:
    """Configuration for a device in the workflow."""
    device_type: str  # e.g., "liquid_handler", "potentiostat"
    adapter: str  # e.g., "ot2", "squidstat"
    name: str  # Instance name
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_type": self.device_type,
            "adapter": self.adapter,
            "name": self.name,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeviceConfig':
        return cls(
            device_type=data["device_type"],
            adapter=data["adapter"],
            name=data["name"],
            config=data.get("config", {}),
        )


@dataclass
class Workflow:
    """
    A complete lab automation workflow.

    Contains all devices, phases, and steps needed to execute
    a multi-instrument experiment.
    """
    workflow_name: str
    version: str = "1.0"
    description: str = ""
    author: str = "Lab Automation Agent"
    created_at: datetime = field(default_factory=datetime.now)

    # Device configurations
    devices: List[DeviceConfig] = field(default_factory=list)

    # Workflow phases
    phases: List[Phase] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Original instructions (for traceability)
    original_instructions: List[str] = field(default_factory=list)

    def add_device(self, device: DeviceConfig):
        """Add a device configuration."""
        self.devices.append(device)

    def add_phase(self, phase: Phase):
        """Add a phase to the workflow."""
        self.phases.append(phase)

    def get_phase(self, name: str) -> Optional[Phase]:
        """Get a phase by name."""
        for phase in self.phases:
            if phase.phase_name == name:
                return phase
        return None

    def get_or_create_phase(self, name: str, description: str = "") -> Phase:
        """Get existing phase or create new one."""
        phase = self.get_phase(name)
        if phase is None:
            phase = Phase(phase_name=name, description=description)
            self.phases.append(phase)
        return phase

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflow_name": self.workflow_name,
            "version": self.version,
            "description": self.description,
            "metadata": {
                "author": self.author,
                "created_at": self.created_at.isoformat(),
                "original_instructions": self.original_instructions,
                **self.metadata,
            },
            "devices": [d.to_dict() for d in self.devices],
            "phases": [p.to_dict() for p in self.phases],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, filepath: str, indent: int = 2):
        """Save workflow to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json(indent=indent))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workflow':
        """Create Workflow from dictionary."""
        metadata = data.get("metadata", {})

        workflow = cls(
            workflow_name=data["workflow_name"],
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            author=metadata.get("author", "Unknown"),
            original_instructions=metadata.get("original_instructions", []),
        )

        # Remove processed metadata fields
        workflow.metadata = {
            k: v for k, v in metadata.items()
            if k not in ("author", "created_at", "original_instructions")
        }

        # Load devices
        for device_data in data.get("devices", []):
            workflow.devices.append(DeviceConfig.from_dict(device_data))

        # Load phases
        for phase_data in data.get("phases", []):
            workflow.phases.append(Phase.from_dict(phase_data))

        return workflow

    @classmethod
    def from_json(cls, json_str: str) -> 'Workflow':
        """Create Workflow from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load_json(cls, filepath: str) -> 'Workflow':
        """Load Workflow from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())

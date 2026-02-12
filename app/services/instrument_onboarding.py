"""Instrument Onboarding Service — auto-generates integration code for new lab instruments.

Given a minimal instrument specification (name, manufacturer, primitives),
this service generates all files required to integrate a new instrument into
the OTbot platform:

1. Hardware controller  (``app/hardware/{name}_controller.py``)
2. Skill definition     (``agent/skills/{name}.md``)
3. Dispatcher handlers  (patch for ``app/hardware/dispatcher.py``)
4. Simulated handlers   (patch for ``app/adapters/simulated_instrument.py``)
5. Adapter registration (patch for ``app/adapters/battery_lab.py``)
6. Tests                (``tests/test_{name}_onboarded.py``)

The service follows a **human-in-the-loop** design: safety-critical decisions
(error/safety classification, preconditions, KPI formulas) are surfaced as
confirmation prompts that must be approved before code is finalised.

Usage (programmatic)::

    spec = InstrumentSpec(
        name="uv_vis",
        manufacturer="Ocean Insight",
        model="Flame-S",
        communication="usb",
        primitives=[
            PrimitiveInput(
                name="measure_spectrum",
                description="Capture UV-Vis absorption spectrum",
                params={"wavelength_start_nm": ParamInput(type="number", default=200), ...},
                hazardous=False,
                generates_data=True,
            ),
        ],
    )
    svc = InstrumentOnboardingService()
    result = svc.generate(spec)
    # result.pending_confirmations  — items needing human review
    # result.files                  — generated file contents
    # svc.confirm(result, confirmations={...})
    # svc.write_files(result)       — write to disk

Usage (agent/chat)::

    The orchestrator agent calls ``svc.generate(spec)`` and, when
    ``result.pending_confirmations`` is non-empty, emits them as chat
    messages for the user to approve before calling ``svc.write_files()``.
"""
from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models — user input
# ---------------------------------------------------------------------------


class CommunicationType(str, Enum):
    """How the instrument physically connects."""

    USB = "usb"
    SERIAL = "serial"
    TCP = "tcp"
    MODBUS = "modbus"
    GPIB = "gpib"
    SIMULATED = "simulated"


@dataclass
class ParamInput:
    """One parameter of a primitive."""

    type: str = "number"  # string | number | integer | boolean | array
    description: str = ""
    default: Any = None
    optional: bool = False


@dataclass
class PrimitiveInput:
    """User-supplied description of one instrument primitive."""

    name: str  # e.g. "measure_spectrum" (without prefix)
    description: str = ""
    params: dict[str, ParamInput] = field(default_factory=dict)

    # Safety hints — human confirms final classification
    hazardous: bool = False  # involves physical motion / high voltage / irreversible
    generates_data: bool = False  # produces measurement data for KPI

    # Auto-filled during processing
    error_class: str = ""  # CRITICAL | BYPASS — auto-inferred, user confirms
    safety_class: str = ""  # HAZARDOUS | CAREFUL | REVERSIBLE | INFORMATIONAL
    timeout_seconds: int = 30
    retries: int = 1

    # Preconditions/effects — user may add
    preconditions: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)


@dataclass
class InstrumentSpec:
    """Complete specification for onboarding a new instrument."""

    name: str  # snake_case identifier, e.g. "uv_vis_spectrometer"
    manufacturer: str = ""
    model: str = ""
    communication: CommunicationType = CommunicationType.USB
    description: str = ""
    primitives: list[PrimitiveInput] = field(default_factory=list)

    # Optional: Python package for the SDK (e.g. "seabreeze")
    sdk_package: str = ""

    # Resource locking
    resource_id: str = ""  # defaults to name if empty

    @property
    def prefix(self) -> str:
        """Action prefix for this instrument (e.g. 'uv_vis')."""
        return self.name

    @property
    def class_name(self) -> str:
        """PascalCase class name (e.g. 'UvVisController')."""
        parts = self.name.split("_")
        return "".join(p.capitalize() for p in parts) + "Controller"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        if self.manufacturer and self.model:
            return f"{self.manufacturer} {self.model}"
        return self.name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Data models — confirmation prompts (human-in-the-loop)
# ---------------------------------------------------------------------------


class ConfirmationType(str, Enum):
    """Categories of items requiring human confirmation."""

    SAFETY_CLASSIFICATION = "safety_classification"
    PRECONDITIONS = "preconditions"
    KPI_EXTRACTION = "kpi_extraction"
    COMMUNICATION_DETAILS = "communication_details"


@dataclass
class ConfirmationItem:
    """A single item the user must review before code is finalised."""

    id: str  # unique key, e.g. "safety:measure_spectrum"
    type: ConfirmationType
    primitive_name: str  # which primitive this relates to
    question: str  # what to ask the user
    current_value: Any  # auto-inferred value
    options: list[str] | None = None  # valid choices (if applicable)
    confirmed: bool = False
    confirmed_value: Any = None

    @property
    def final_value(self) -> Any:
        """Return confirmed value if confirmed, else current auto-inferred value."""
        return self.confirmed_value if self.confirmed else self.current_value


# ---------------------------------------------------------------------------
# Data models — output
# ---------------------------------------------------------------------------


@dataclass
class GeneratedFile:
    """A single generated file or patch."""

    path: str  # relative to project root
    content: str  # full file content (for new files) or patch snippet
    is_patch: bool = False  # True = append/insert into existing file
    patch_marker: str = ""  # where to insert (for patches)
    description: str = ""  # human-readable explanation


@dataclass
class OnboardingResult:
    """Complete result of instrument onboarding code generation."""

    spec: InstrumentSpec
    files: list[GeneratedFile] = field(default_factory=list)
    pending_confirmations: list[ConfirmationItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manual_todo: list[str] = field(default_factory=list)

    @property
    def ready_to_write(self) -> bool:
        """True if all confirmations are resolved."""
        return all(c.confirmed for c in self.pending_confirmations)

    @property
    def confirmed_count(self) -> int:
        return sum(1 for c in self.pending_confirmations if c.confirmed)

    @property
    def total_confirmations(self) -> int:
        return len(self.pending_confirmations)


# ---------------------------------------------------------------------------
# Service implementation
# ---------------------------------------------------------------------------


class InstrumentOnboardingService:
    """Generates all integration code for a new instrument.

    The workflow is:
    1. ``generate(spec)`` — produces code + confirmation prompts
    2. ``confirm(result, confirmations)`` — user approves safety decisions
    3. ``write_files(result, project_root)`` — writes files to disk
    """

    def __init__(self, project_root: str | Path | None = None) -> None:
        if project_root is None:
            # Auto-detect: walk up from this file to find project root
            project_root = Path(__file__).resolve().parent.parent.parent
        self.project_root = Path(project_root)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(self, spec: InstrumentSpec) -> OnboardingResult:
        """Generate all integration code and confirmation prompts.

        Returns an ``OnboardingResult`` with generated files and any
        items requiring human confirmation.
        """
        if not spec.resource_id:
            spec.resource_id = spec.name

        result = OnboardingResult(spec=spec)

        # Step 1: Infer safety classifications and build confirmations
        self._infer_safety_classifications(spec, result)

        # Step 2: Build communication detail confirmations
        self._build_communication_confirmations(spec, result)

        # Step 3: Build KPI confirmations for data-generating primitives
        self._build_kpi_confirmations(spec, result)

        # Step 4: Generate all files (using current/inferred values)
        self._generate_all_files(spec, result)

        # Step 5: Populate manual TODO items
        result.manual_todo.append(
            f"Install SDK package: pip install {spec.sdk_package}"
            if spec.sdk_package
            else f"Ensure {spec.display_name} Python SDK is available"
        )
        result.manual_todo.append(
            f"Test with real {spec.display_name} hardware before production use"
        )
        result.manual_todo.append(
            "Review and adjust timeout values based on actual instrument response times"
        )

        return result

    def confirm(
        self,
        result: OnboardingResult,
        confirmations: dict[str, Any],
    ) -> OnboardingResult:
        """Apply user confirmations and regenerate affected files.

        Args:
            result: Previous OnboardingResult from ``generate()``.
            confirmations: Mapping of confirmation id → confirmed value.

        Returns:
            Updated OnboardingResult with confirmed items and regenerated files.
        """
        for item in result.pending_confirmations:
            if item.id in confirmations:
                item.confirmed = True
                item.confirmed_value = confirmations[item.id]

        # Apply confirmed safety classifications back to spec
        for item in result.pending_confirmations:
            if item.confirmed and item.type == ConfirmationType.SAFETY_CLASSIFICATION:
                prim_name = item.primitive_name
                for prim in result.spec.primitives:
                    if prim.name == prim_name:
                        if "safety_class" in item.id:
                            prim.safety_class = item.final_value
                        elif "error_class" in item.id:
                            prim.error_class = item.final_value

        # Regenerate files with confirmed values
        result.files.clear()
        self._generate_all_files(result.spec, result)

        return result

    def write_files(
        self,
        result: OnboardingResult,
        *,
        force: bool = False,
    ) -> list[str]:
        """Write generated files to disk.

        Args:
            result: OnboardingResult with generated files.
            force: If True, write even if confirmations are pending.

        Returns:
            List of file paths written.

        Raises:
            RuntimeError: If confirmations are pending and force=False.
        """
        if not result.ready_to_write and not force:
            pending = [c for c in result.pending_confirmations if not c.confirmed]
            raise RuntimeError(
                f"{len(pending)} confirmation(s) still pending. "
                f"Call confirm() first or use force=True."
            )

        written: list[str] = []
        for gf in result.files:
            full_path = self.project_root / gf.path
            if gf.is_patch:
                self._apply_patch(full_path, gf)
            else:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(gf.content, encoding="utf-8")
            written.append(str(full_path))

        # If a skill file was written, refresh the PrimitivesRegistry singleton
        # so that newly onboarded instruments are immediately available to the
        # conversation engine and NL parser.
        skill_written = any(
            gf.path.endswith(".md") and "skills/" in gf.path
            for gf in result.files
        )
        if skill_written:
            try:
                from app.services.primitives_registry import refresh_registry
                refresh_registry()
                logger.info(
                    "Refreshed PrimitivesRegistry after writing skill for %s",
                    result.spec.display_name,
                )
            except Exception:
                logger.warning(
                    "Failed to refresh PrimitivesRegistry after onboarding %s",
                    result.spec.display_name,
                    exc_info=True,
                )

        return written

    def format_confirmations_for_chat(self, result: OnboardingResult) -> str:
        """Format pending confirmations as a user-friendly chat message.

        This is what the agent sends to the user in the conversation.
        """
        pending = [c for c in result.pending_confirmations if not c.confirmed]
        if not pending:
            return ""

        parts: list[str] = []
        parts.append(
            f"## New Instrument: {result.spec.display_name}\n"
        )
        parts.append(
            "Before I generate the integration code, I need you to confirm "
            "the following safety and configuration decisions:\n"
        )

        by_type: dict[ConfirmationType, list[ConfirmationItem]] = {}
        for item in pending:
            by_type.setdefault(item.type, []).append(item)

        # Safety classifications first (most important)
        if ConfirmationType.SAFETY_CLASSIFICATION in by_type:
            parts.append("### Safety Classifications\n")
            parts.append(
                "These determine how the system handles errors during execution. "
                "**Getting this wrong can damage equipment or ruin experiments.**\n"
            )
            for item in by_type[ConfirmationType.SAFETY_CLASSIFICATION]:
                options_str = ""
                if item.options:
                    options_str = f" (options: {', '.join(item.options)})"
                parts.append(
                    f"- **{item.primitive_name}**: {item.question}\n"
                    f"  Current: `{item.current_value}`{options_str}\n"
                )

        # Communication details
        if ConfirmationType.COMMUNICATION_DETAILS in by_type:
            parts.append("\n### Communication Details\n")
            for item in by_type[ConfirmationType.COMMUNICATION_DETAILS]:
                parts.append(
                    f"- **{item.primitive_name}**: {item.question}\n"
                    f"  Current: `{item.current_value}`\n"
                )

        # KPI extraction
        if ConfirmationType.KPI_EXTRACTION in by_type:
            parts.append("\n### KPI Extraction\n")
            parts.append(
                "These primitives generate measurement data. "
                "Please confirm what KPI should be extracted:\n"
            )
            for item in by_type[ConfirmationType.KPI_EXTRACTION]:
                parts.append(
                    f"- **{item.primitive_name}**: {item.question}\n"
                    f"  Suggested: `{item.current_value}`\n"
                )

        parts.append(
            "\nPlease confirm or adjust these values. "
            "Reply with the corrections, or say 'confirm all' to accept the defaults."
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Confirmation builders
    # ------------------------------------------------------------------ #

    def _infer_safety_classifications(
        self, spec: InstrumentSpec, result: OnboardingResult
    ) -> None:
        """Auto-infer safety/error classes and create confirmation prompts."""
        for prim in spec.primitives:
            # Infer safety_class from hazardous flag
            if not prim.safety_class:
                if prim.hazardous:
                    prim.safety_class = "HAZARDOUS"
                elif prim.generates_data:
                    prim.safety_class = "CAREFUL"
                else:
                    prim.safety_class = "REVERSIBLE"

            # Infer error_class from safety_class
            if not prim.error_class:
                if prim.safety_class in ("HAZARDOUS", "CAREFUL"):
                    prim.error_class = "CRITICAL"
                else:
                    prim.error_class = "BYPASS"

            # Create confirmation for safety_class
            result.pending_confirmations.append(ConfirmationItem(
                id=f"safety_class:{prim.name}",
                type=ConfirmationType.SAFETY_CLASSIFICATION,
                primitive_name=prim.name,
                question=(
                    f"Safety class for `{spec.prefix}.{prim.name}`? "
                    f"{'This primitive is marked hazardous.' if prim.hazardous else ''}"
                ),
                current_value=prim.safety_class,
                options=["INFORMATIONAL", "REVERSIBLE", "CAREFUL", "HAZARDOUS"],
            ))

            # Create confirmation for error_class
            result.pending_confirmations.append(ConfirmationItem(
                id=f"error_class:{prim.name}",
                type=ConfirmationType.SAFETY_CLASSIFICATION,
                primitive_name=prim.name,
                question=(
                    f"Error class for `{spec.prefix}.{prim.name}`? "
                    f"CRITICAL = abort on failure, BYPASS = log and continue."
                ),
                current_value=prim.error_class,
                options=["CRITICAL", "BYPASS"],
            ))

    def _build_communication_confirmations(
        self, spec: InstrumentSpec, result: OnboardingResult
    ) -> None:
        """Build confirmations for communication parameters."""
        comm = spec.communication

        if comm == CommunicationType.SERIAL or comm == CommunicationType.USB:
            result.pending_confirmations.append(ConfirmationItem(
                id="comm:port",
                type=ConfirmationType.COMMUNICATION_DETAILS,
                primitive_name="(connection)",
                question="Serial/USB port for the instrument?",
                current_value=_default_port(comm),
            ))
            result.pending_confirmations.append(ConfirmationItem(
                id="comm:baudrate",
                type=ConfirmationType.COMMUNICATION_DETAILS,
                primitive_name="(connection)",
                question="Baud rate?",
                current_value=9600,
            ))
        elif comm == CommunicationType.TCP or comm == CommunicationType.MODBUS:
            result.pending_confirmations.append(ConfirmationItem(
                id="comm:host",
                type=ConfirmationType.COMMUNICATION_DETAILS,
                primitive_name="(connection)",
                question="IP address or hostname of the instrument?",
                current_value="192.168.1.100",
            ))
            result.pending_confirmations.append(ConfirmationItem(
                id="comm:port_number",
                type=ConfirmationType.COMMUNICATION_DETAILS,
                primitive_name="(connection)",
                question="TCP port number?",
                current_value=502 if comm == CommunicationType.MODBUS else 5000,
            ))

    def _build_kpi_confirmations(
        self, spec: InstrumentSpec, result: OnboardingResult
    ) -> None:
        """Build confirmations for KPI extraction from data-generating primitives."""
        for prim in spec.primitives:
            if prim.generates_data:
                result.pending_confirmations.append(ConfirmationItem(
                    id=f"kpi:{prim.name}",
                    type=ConfirmationType.KPI_EXTRACTION,
                    primitive_name=prim.name,
                    question=(
                        f"What KPI should be extracted from `{spec.prefix}.{prim.name}` results? "
                        f"(e.g. 'peak_absorbance', 'impedance_ohm', 'current_density_ma_cm2')"
                    ),
                    current_value=_guess_kpi_name(prim),
                ))

    # ------------------------------------------------------------------ #
    # File generators
    # ------------------------------------------------------------------ #

    def _generate_all_files(
        self, spec: InstrumentSpec, result: OnboardingResult
    ) -> None:
        """Generate all integration files."""
        result.files.append(self._gen_controller(spec))
        result.files.append(self._gen_skill_md(spec))
        result.files.append(self._gen_dispatcher_patch(spec))
        result.files.append(self._gen_simulated_patch(spec))
        result.files.append(self._gen_adapter_patch(spec))
        result.files.append(self._gen_dry_run_patch(spec))
        result.files.append(self._gen_tests(spec))

    # -- 1. Hardware controller --

    def _gen_controller(self, spec: InstrumentSpec) -> GeneratedFile:
        """Generate ``app/hardware/{name}_controller.py``."""
        cls = spec.class_name
        comm = spec.communication
        sdk = spec.sdk_package or f"{spec.name}_sdk"

        # Connection init code depends on communication type
        if comm in (CommunicationType.SERIAL, CommunicationType.USB):
            init_params = "self, port: str = \"/dev/ttyUSB0\", baudrate: int = 9600"
            init_body = textwrap.dedent(f"""\
                self.port = port
                self.baudrate = baudrate
                self._device = None
                LOGGER.info("Initializing {cls} on %s @ %d baud", port, baudrate)
                try:
                    self._device = External{cls.replace('Controller', '')}(port=port, baudrate=baudrate)
                    LOGGER.info("{cls} connected successfully")
                except Exception as exc:
                    LOGGER.error("Failed to connect to {spec.display_name}: %s", exc)
                    raise""")
            close_body = textwrap.dedent("""\
                if self._device:
                    try:
                        self._device.close()
                        LOGGER.info("{cls} connection closed")
                    except Exception as exc:
                        LOGGER.error("Error closing {cls}: %s", exc)""").format(cls=cls)
        elif comm in (CommunicationType.TCP, CommunicationType.MODBUS):
            init_params = "self, host: str = \"192.168.1.100\", port: int = 5000"
            init_body = textwrap.dedent(f"""\
                self.host = host
                self.port = port
                self._device = None
                LOGGER.info("Initializing {cls} at %s:%d", host, port)
                try:
                    self._device = External{cls.replace('Controller', '')}(host=host, port=port)
                    LOGGER.info("{cls} connected successfully")
                except Exception as exc:
                    LOGGER.error("Failed to connect to {spec.display_name}: %s", exc)
                    raise""")
            close_body = textwrap.dedent("""\
                if self._device:
                    try:
                        self._device.close()
                        LOGGER.info("{cls} connection closed")
                    except Exception as exc:
                        LOGGER.error("Error closing {cls}: %s", exc)""").format(cls=cls)
        else:
            init_params = "self"
            init_body = textwrap.dedent(f"""\
                self._device = None
                LOGGER.info("Initializing {cls} (simulated)")""")
            close_body = "pass"

        # Generate method stubs for each primitive
        methods: list[str] = []
        for prim in spec.primitives:
            method_name = prim.name
            params_sig = ", ".join(
                f"{pname}: {_python_type(pdef.type)}"
                + (f" = {_python_default(pdef.default)}" if pdef.default is not None else "")
                for pname, pdef in prim.params.items()
            )
            params_call = ", ".join(
                f"{pname}={pname}" for pname in prim.params
            )
            params_log = ", ".join(
                f"{pname}=%s" for pname in prim.params
            )
            params_log_args = ", ".join(prim.params.keys())

            return_type = "dict[str, Any]" if prim.generates_data else "bool"
            return_val = (
                '{\n                "ok": True,\n                # TODO: parse instrument response into structured data\n            }'
                if prim.generates_data
                else "True"
            )

            method = textwrap.dedent(f"""\
    def {method_name}(self, {params_sig}) -> {return_type}:
        \"\"\"{prim.description or f'Execute {prim.name} on {spec.display_name}'}\"\"\"
        if not self.is_available():
            LOGGER.warning("{spec.name} not available - skipping {method_name}")
            return {"{"}'ok': False{"}"} if {prim.generates_data} else False

        with self._lock:
            try:
                LOGGER.info("[{spec.prefix.upper()}] {method_name}({params_log})", {params_log_args})
                # TODO: Replace with actual SDK call
                # result = self._device.{method_name}({params_call})
                LOGGER.info("[{spec.prefix.upper()}] {method_name} completed")
                return {return_val}
            except Exception as exc:
                LOGGER.error("[{spec.prefix.upper()}] {method_name} failed: %s", exc)
                raise
""")
            methods.append(method)

        methods_str = "\n".join(methods)

        content = textwrap.dedent(f'''\
"""
{spec.display_name} Controller
{"Manufacturer: " + spec.manufacturer if spec.manufacturer else ""}
{"Model: " + spec.model if spec.model else ""}
Communication: {comm.value}

Auto-generated by InstrumentOnboardingService.
TODO: Replace SDK stubs with actual hardware calls.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

LOGGER = logging.getLogger(__name__)

# External SDK import — only available when the SDK is installed.
# In simulated / dry-run mode this module is never imported.
try:
    # TODO: Replace with actual SDK import
    # from {sdk} import {cls.replace("Controller", "")} as External{cls.replace("Controller", "")}
    External{cls.replace("Controller", "")} = None  # type: ignore[assignment]
except ImportError:
    External{cls.replace("Controller", "")} = None  # type: ignore[assignment,misc]


class {cls}:
    """Wrapper for {spec.display_name} instrument control.

    Provides a clean interface for the OTbot ActionDispatcher.
    Thread-safe via ``_lock``.
    """

    _lock = threading.Lock()

    def __init__({init_params}) -> None:
        """Initialize {spec.display_name} controller."""
{textwrap.indent(init_body, "        ")}

    def is_available(self) -> bool:
        """Check if controller is connected and ready."""
        return self._device is not None

{methods_str}
    def close(self) -> None:
        """Close instrument connection."""
{textwrap.indent(close_body, "        ")}
''')
        return GeneratedFile(
            path=f"app/hardware/{spec.name}_controller.py",
            content=content,
            description=f"Hardware controller for {spec.display_name}",
        )

    # -- 2. Skill markdown --

    def _gen_skill_md(self, spec: InstrumentSpec) -> GeneratedFile:
        """Generate ``agent/skills/{name}.md``."""
        primitives_yaml: list[str] = []
        for prim in spec.primitives:
            params_yaml = ""
            if prim.params:
                param_lines = []
                for pname, pdef in prim.params.items():
                    desc = pdef.description or pname.replace("_", " ")
                    param_lines.append(
                        f"      {pname}: {{type: {pdef.type}, description: \"{desc}\"}}"
                    )
                params_yaml = "\n    params:\n" + "\n".join(param_lines)

            preconds_yaml = ""
            if prim.preconditions:
                lines = "\n".join(f'        - "{p}"' for p in prim.preconditions)
                preconds_yaml = f"\n      preconditions:\n{lines}"
            else:
                preconds_yaml = "\n      preconditions: []"

            effects_yaml = ""
            if prim.effects:
                lines = "\n".join(f'        - "{e}"' for e in prim.effects)
                effects_yaml = f"\n      effects:\n{lines}"
            else:
                effects_yaml = "\n      effects: []"

            prim_yaml = textwrap.dedent(f"""\
  - name: {spec.prefix}.{prim.name}
    error_class: {prim.error_class or 'BYPASS'}
    safety_class: {prim.safety_class or 'REVERSIBLE'}{params_yaml}
    description: "{prim.description or prim.name}"
    contract:{preconds_yaml}{effects_yaml}
    timeout:
      seconds: {prim.timeout_seconds}
      retries: {prim.retries}""")
            primitives_yaml.append(prim_yaml)

        primitives_block = "\n".join(primitives_yaml)

        # Error behavior table
        error_rows: list[str] = []
        for prim in spec.primitives:
            ec = prim.error_class or "BYPASS"
            sc = prim.safety_class or "REVERSIBLE"
            on_fail = {
                "HAZARDOUS": "Abort immediately",
                "CAREFUL": f"Retry ({prim.retries}x), then abort",
                "REVERSIBLE": f"Retry ({prim.retries}x), log and continue",
                "INFORMATIONAL": "Log and continue",
            }.get(sc, "Log and continue")
            error_rows.append(
                f"| {spec.prefix}.{prim.name} | {ec} | {sc} | {on_fail} |"
            )
        error_table = "\n".join(error_rows)

        n_prims = len(spec.primitives)
        safety_summary = ", ".join(
            f"{count} {cls}"
            for cls, count in _count_safety_classes(spec.primitives).items()
        )

        content = textwrap.dedent(f"""\
---
name: {spec.name}
description: "{spec.description or spec.display_name + ' instrument control'}"
version: "1.0.0"
instrument: {spec.name}
resource_id: {spec.resource_id}
primitives:
{primitives_block}
---

# {spec.display_name} — Instrument Control

{spec.description or f"Controls the {spec.display_name} instrument."}
{"Manufacturer: " + spec.manufacturer + "  " if spec.manufacturer else ""}{"Model: " + spec.model if spec.model else ""}

## When to Use

Use `{spec.prefix}.*` primitives when the protocol requires interaction
with the {spec.display_name}.

## Resource Locking

All `{spec.prefix}.*` primitives require the `{spec.resource_id}` resource lock.

## Error Behavior

| Primitive | Error Class | Safety Class | On Failure |
|-----------|------------|-------------|------------|
{error_table}

*{n_prims} primitive(s). {safety_summary}.*
""")
        return GeneratedFile(
            path=f"agent/skills/{spec.name}.md",
            content=content,
            description=f"Skill definition for {spec.display_name}",
        )

    # -- 3. Dispatcher patch --

    def _gen_dispatcher_patch(self, spec: InstrumentSpec) -> GeneratedFile:
        """Generate patch snippet for ``app/hardware/dispatcher.py``."""
        handler_registrations: list[str] = []
        handler_methods: list[str] = []

        for prim in spec.primitives:
            full_name = f"{spec.prefix}.{prim.name}"
            handler_name = f"_handle_{spec.prefix}_{prim.name}"

            handler_registrations.append(
                f'            "{full_name}": self.{handler_name},'
            )

            # Build handler method
            param_sig = ", ".join(
                f"{pname}: {_python_type(pdef.type)}"
                + (" = None" if pdef.optional else "")
                for pname, pdef in prim.params.items()
            )
            params_log = ", ".join(f"{pname}=%s" for pname in prim.params)
            params_log_args = ", ".join(prim.params.keys())

            is_bypass = (prim.error_class or "BYPASS") == "BYPASS"
            error_tag = "BYPASS" if is_bypass else "CRITICAL"
            error_handling = (
                f'            logging.warning("[{error_tag}] Continuing workflow")\n'
                f'            return False'
                if is_bypass
                else "            raise"
            )

            method = textwrap.dedent(f"""\
    def {handler_name}(self, {param_sig}, **kwargs):
        \"\"\"{prim.description or prim.name}\"\"\"
        try:
            logging.info("[{spec.prefix.upper()}] {prim.name}({params_log})", {params_log_args})
            result = self.{spec.name}.{prim.name}({", ".join(f"{p}={p}" for p in prim.params)})
            logging.info("[{spec.prefix.upper()}] {prim.name} completed")
            return result
        except Exception as exc:
            logging.error("[{error_tag}] {spec.prefix}.{prim.name} failed: %s", exc)
{error_handling}
""")
            handler_methods.append(method)

        registrations_str = "\n".join(handler_registrations)
        methods_str = "\n".join(handler_methods)

        content = textwrap.dedent(f"""\
# ===========================================================================
# Auto-generated dispatcher patch for: {spec.display_name}
# Add to app/hardware/dispatcher.py
# ===========================================================================

# 1. Add to ActionDispatcher.__init__() parameter list:
#    {spec.name}=None,

# 2. Store reference in __init__() body:
#    self.{spec.name} = {spec.name}

# 3. Add these entries to self.action_handlers dict:
{registrations_str}

# 4. Add these handler methods to ActionDispatcher class:

{methods_str}
""")
        return GeneratedFile(
            path=f"_onboarding_patches/{spec.name}_dispatcher_patch.py",
            content=content,
            is_patch=True,
            patch_marker="app/hardware/dispatcher.py",
            description=f"Dispatcher handler patch for {spec.display_name}",
        )

    # -- 4. Simulated adapter patch --

    def _gen_simulated_patch(self, spec: InstrumentSpec) -> GeneratedFile:
        """Generate patch snippet for ``app/adapters/simulated_instrument.py``."""
        entries: list[str] = []
        for prim in spec.primitives:
            full_name = f"{spec.prefix}.{prim.name}"
            entries.append(f'    "{full_name}": _handle_generic_ok,')

        entries_str = "\n".join(entries)

        content = textwrap.dedent(f"""\
# ===========================================================================
# Auto-generated simulated adapter patch for: {spec.display_name}
# Add to app/adapters/simulated_instrument.py
# ===========================================================================

# Add these entries to the _PRIMITIVE_HANDLERS dict:
{entries_str}
""")
        return GeneratedFile(
            path=f"_onboarding_patches/{spec.name}_simulated_patch.py",
            content=content,
            is_patch=True,
            patch_marker="app/adapters/simulated_instrument.py",
            description=f"Simulated handler entries for {spec.display_name}",
        )

    # -- 5. Battery lab adapter patch --

    def _gen_adapter_patch(self, spec: InstrumentSpec) -> GeneratedFile:
        """Generate patch snippet for ``app/adapters/battery_lab.py``."""
        cls = spec.class_name
        comm = spec.communication

        if comm in (CommunicationType.SERIAL, CommunicationType.USB):
            init_line = f"self._{spec.name} = {cls}(port=settings.{spec.name}_port)"
        elif comm in (CommunicationType.TCP, CommunicationType.MODBUS):
            init_line = f"self._{spec.name} = {cls}(host=settings.{spec.name}_host, port=settings.{spec.name}_port)"
        else:
            init_line = f"self._{spec.name} = {cls}()"

        content = textwrap.dedent(f"""\
# ===========================================================================
# Auto-generated adapter patch for: {spec.display_name}
# Modify app/adapters/battery_lab.py
# ===========================================================================

# 1. Add to BatteryLabAdapter.__init__():
#    self._{spec.name} = None

# 2. Add to BatteryLabAdapter.connect() (inside 'else' / real-hardware branch):
#    from app.hardware.{spec.name}_controller import {cls}
#    {init_line}

# 3. Pass to ActionDispatcher:
#    self._dispatcher = ActionDispatcher(
#        ...existing args...,
#        {spec.name}=self._{spec.name},
#    )

# 4. Add to BatteryLabAdapter.disconnect():
#    if self._{spec.name}:
#        try:
#            self._{spec.name}.close()
#        except Exception:
#            logger.exception("Error closing {spec.name}")

# 5. Add to BatteryLabAdapter.health_check() return dict:
#    "{spec.name}": self._{spec.name} is not None if not self.dry_run else "stub",
""")
        return GeneratedFile(
            path=f"_onboarding_patches/{spec.name}_adapter_patch.py",
            content=content,
            is_patch=True,
            patch_marker="app/adapters/battery_lab.py",
            description=f"Adapter registration patch for {spec.display_name}",
        )

    # -- 6. Dry-run dispatcher patch --

    def _gen_dry_run_patch(self, spec: InstrumentSpec) -> GeneratedFile:
        """Generate patch for _DryRunDispatcher.KNOWN_ACTIONS."""
        actions = [f'"{spec.prefix}.{p.name}"' for p in spec.primitives]
        actions_str = ", ".join(actions)

        content = textwrap.dedent(f"""\
# ===========================================================================
# Auto-generated dry-run patch for: {spec.display_name}
# Modify app/adapters/battery_lab.py → _DryRunDispatcher.KNOWN_ACTIONS
# ===========================================================================

# Add to KNOWN_ACTIONS frozenset:
#    {actions_str},
""")
        return GeneratedFile(
            path=f"_onboarding_patches/{spec.name}_dryrun_patch.py",
            content=content,
            is_patch=True,
            patch_marker="app/adapters/battery_lab.py",
            description=f"Dry-run action entries for {spec.display_name}",
        )

    # -- 7. Tests --

    def _gen_tests(self, spec: InstrumentSpec) -> GeneratedFile:
        """Generate ``tests/test_{name}_onboarded.py``."""
        cls = spec.class_name

        # Test: controller can be imported
        import_test = textwrap.dedent(f"""\
class TestImport:
    def test_controller_importable(self):
        from app.hardware.{spec.name}_controller import {cls}
        assert {cls} is not None

    def test_controller_class_name(self):
        from app.hardware.{spec.name}_controller import {cls}
        assert {cls}.__name__ == "{cls}"
""")

        # Test: skill file is valid YAML
        skill_test = textwrap.dedent(f"""\

class TestSkillFile:
    def test_skill_file_exists(self):
        from pathlib import Path
        skill_path = Path(__file__).parent.parent / "agent" / "skills" / "{spec.name}.md"
        assert skill_path.exists(), f"Skill file not found: {{skill_path}}"

    def test_skill_frontmatter_parseable(self):
        from pathlib import Path
        import re
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        skill_path = Path(__file__).parent.parent / "agent" / "skills" / "{spec.name}.md"
        text = skill_path.read_text()
        match = re.match(r"^---\\n(.*?)\\n---", text, re.DOTALL)
        assert match, "No YAML frontmatter found"
        data = yaml.safe_load(match.group(1))
        assert data["name"] == "{spec.name}"
        assert len(data["primitives"]) == {len(spec.primitives)}

    def test_all_primitives_have_safety_class(self):
        from pathlib import Path
        import re
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        skill_path = Path(__file__).parent.parent / "agent" / "skills" / "{spec.name}.md"
        text = skill_path.read_text()
        match = re.match(r"^---\\n(.*?)\\n---", text, re.DOTALL)
        data = yaml.safe_load(match.group(1))
        for prim in data["primitives"]:
            assert "safety_class" in prim, f"Missing safety_class in {{prim['name']}}"
            assert prim["safety_class"] in ("INFORMATIONAL", "REVERSIBLE", "CAREFUL", "HAZARDOUS")
""")

        # Test: simulated adapter handles new primitives
        sim_tests: list[str] = []
        for prim in spec.primitives:
            full_name = f"{spec.prefix}.{prim.name}"
            sim_tests.append(textwrap.dedent(f"""\
    def test_simulated_{prim.name}(self):
        \"\"\"Simulated adapter should handle {full_name}.\"\"\"
        from app.adapters.simulated_instrument import _PRIMITIVE_HANDLERS
        assert "{full_name}" in _PRIMITIVE_HANDLERS, (
            "Primitive '{full_name}' not registered in simulated adapter. "
            "Apply the onboarding patch to simulated_instrument.py."
        )
"""))
        sim_tests_str = "\n".join(sim_tests)

        simulated_test = f"""
class TestSimulatedAdapter:
{sim_tests_str}"""

        content = textwrap.dedent(f"""\
\"\"\"Tests for onboarded instrument: {spec.display_name}.

Auto-generated by InstrumentOnboardingService.
\"\"\"
import pytest


{import_test}
{skill_test}
{simulated_test}
""")
        return GeneratedFile(
            path=f"tests/test_{spec.name}_onboarded.py",
            content=content,
            description=f"Tests for {spec.display_name} onboarding",
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _apply_patch(self, path: Path, gf: GeneratedFile) -> None:
        """Write a patch file alongside the target."""
        # For patches, we write them as standalone files for manual review.
        # The user (or a follow-up agent step) applies them.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(gf.content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _python_type(param_type: str) -> str:
    """Map YAML param types to Python type hints."""
    return {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool",
        "array": "list",
    }.get(param_type, "Any")


def _python_default(val: Any) -> str:
    """Format a default value for Python code."""
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, bool):
        return "True" if val else "False"
    return str(val)


def _default_port(comm: CommunicationType) -> str:
    """Default port string for a communication type."""
    if comm == CommunicationType.USB:
        return "/dev/ttyUSB0"
    if comm == CommunicationType.SERIAL:
        return "/dev/ttyS0"
    return ""


def _guess_kpi_name(prim: PrimitiveInput) -> str:
    """Guess a KPI name from the primitive name."""
    name = prim.name.lower()
    if "spectrum" in name or "absorbance" in name:
        return "peak_absorbance"
    if "impedance" in name or "eis" in name:
        return "impedance_ohm"
    if "current" in name or "cv" in name:
        return "peak_current_ma"
    if "voltage" in name or "potential" in name:
        return "overpotential_mv"
    if "temperature" in name or "temp" in name:
        return "temperature_c"
    if "weight" in name or "mass" in name:
        return "mass_mg"
    return f"{prim.name}_value"


def _count_safety_classes(
    primitives: list[PrimitiveInput],
) -> dict[str, int]:
    """Count primitives by safety class."""
    counts: dict[str, int] = {}
    for p in primitives:
        sc = p.safety_class or "REVERSIBLE"
        counts[sc] = counts.get(sc, 0) + 1
    return counts

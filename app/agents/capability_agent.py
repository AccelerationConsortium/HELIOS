"""Capability Agent: queries device capabilities and tool registry.

Provides a unified CapabilitySnapshot to the ExecutionAgent before any
compilation or execution attempt. Two data sources are merged:

1. **Opentrons HTTP API** (port 31950) — live robot health, instruments, deck state.
2. **PrimitivesRegistry** — locally registered skills / primitives.

If the robot is unreachable (simulated mode, network failure), the agent
returns a degraded snapshot with warnings rather than raising.

Layer: cross-cutting  (called by ExecutionAgent before compilation)
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.agents.base import AgentResult, BaseAgent, DecisionNode
from app.core.config import get_settings
from app.services.primitives_registry import PrimitivesRegistry, get_registry

logger = logging.getLogger(__name__)

_OPENTRONS_API_VERSION = "3"
_OPENTRONS_PORT = 31950


# ── Output models ─────────────────────────────────────────────────────────────

class PipetteInfo(BaseModel):
    """Attached pipette on one robot mount."""

    mount: str                    # "left" | "right"
    model: str                    # e.g. "p300_single_gen2"
    max_volume_ul: float
    tip_attached: bool = False


class SlotInfo(BaseModel):
    """One deck slot and its current occupant (if any)."""

    slot: str                     # "1"-"11" (OT-2) or "A1"-"D3" (Flex)
    occupied: bool
    labware_name: str | None = None
    labware_uri: str | None = None
    role: str | None = None       # "tips" | "reagent" | "waste" | ...


class CapabilitySnapshot(BaseModel):
    """Unified view of what the robot can do right now.

    Consumed by ExecutionAgent to route WorkflowGraph steps to the correct
    backend and reject primitives that cannot be satisfied.
    """

    robot_type: str               # "ot2" | "flex"
    robot_ip: str
    robot_reachable: bool
    firmware_version: str | None = None

    pipettes: list[PipetteInfo] = Field(default_factory=list)
    deck_slots: list[SlotInfo] = Field(default_factory=list)

    available_primitives: list[str] = Field(
        default_factory=list,
        description="Primitive names executable on the current hardware config",
    )
    constrained_primitives: list[str] = Field(
        default_factory=list,
        description="Primitives blocked by current deck / instrument state",
    )

    warnings: list[str] = Field(default_factory=list)
    raw_instrument_data: dict[str, Any] = Field(default_factory=dict)

    def can_execute(self, primitive_name: str) -> bool:
        return primitive_name in self.available_primitives


# ── Input model ───────────────────────────────────────────────────────────────

class CapabilityQueryInput(BaseModel):
    """Input to CapabilityAgent."""

    campaign_id: str
    requested_primitives: list[str] = Field(
        default_factory=list,
        description="Check only these primitives; empty list = check all registered",
    )
    include_deck_state: bool = True
    timeout_s: float = Field(default=5.0, ge=0.5, le=30.0)


# ── Agent ─────────────────────────────────────────────────────────────────────

class CapabilityAgent(BaseAgent[CapabilityQueryInput, CapabilitySnapshot]):
    """Query device capabilities, deck state, and executable primitives.

    Talks to the Opentrons HTTP API for live robot state.
    Falls back to a degraded (simulated) snapshot when the robot is unreachable.
    """

    name = "capability_agent"
    description = "Query device capabilities, deck state, and primitive registry"
    layer = "cross-cutting"

    def __init__(self, registry: PrimitivesRegistry | None = None) -> None:
        self._registry = registry or get_registry()

    # ── BaseAgent interface ───────────────────────────────────────────────

    def validate_input(self, input_data: CapabilityQueryInput) -> list[str]:
        errors: list[str] = []
        if not input_data.campaign_id.strip():
            errors.append("campaign_id must not be empty")
        return errors

    async def process(self, input_data: CapabilityQueryInput) -> CapabilitySnapshot:
        settings = get_settings()
        robot_ip: str = settings.robot_ip
        robot_type: str = settings.robot_type

        # 1. Probe robot health ------------------------------------------------
        reachable, health_data = await self._probe_health(robot_ip, input_data.timeout_s)

        reachability_node = DecisionNode(
            id="robot_reachability",
            label="Robot reachability check",
            options=["reachable", "unreachable"],
            selected="reachable" if reachable else "unreachable",
            reason=(
                f"HTTP GET http://{robot_ip}:{_OPENTRONS_PORT}/health "
                f"→ {'200 OK' if reachable else 'timeout / connection error'}"
            ),
            outcome=(
                "proceeding with live robot query"
                if reachable
                else "falling back to simulated capability state"
            ),
        )
        logger.info(
            "capability_agent: robot_reachable=%s ip=%s",
            reachable,
            robot_ip,
            extra={"campaign_id": input_data.campaign_id},
        )

        # 2. Query instruments & deck ------------------------------------------
        pipettes: list[PipetteInfo] = []
        deck_slots: list[SlotInfo] = []
        raw_instrument_data: dict[str, Any] = {}
        warnings: list[str] = []

        if reachable:
            pipettes, raw_instrument_data = await self._query_instruments(
                robot_ip, input_data.timeout_s
            )
            if input_data.include_deck_state:
                deck_slots, deck_warnings = await self._query_deck(
                    robot_ip, input_data.timeout_s
                )
                warnings.extend(deck_warnings)
        else:
            warnings.append(
                f"Robot at {robot_ip}:{_OPENTRONS_PORT} is unreachable; "
                "capability snapshot is simulated."
            )

        # 3. Evaluate primitives against current state -------------------------
        all_primitives = self._registry.list_primitives()
        target_names = (
            set(input_data.requested_primitives)
            if input_data.requested_primitives
            else None
        )

        available: list[str] = []
        constrained: list[str] = []

        for prim in all_primitives:
            if target_names and prim.name not in target_names:
                continue
            blocked, reason = self._is_constrained(prim.name, pipettes, deck_slots)
            if blocked:
                constrained.append(prim.name)
                if reason:
                    warnings.append(f"primitive '{prim.name}' constrained: {reason}")
            else:
                available.append(prim.name)

        primitive_node = DecisionNode(
            id="primitive_availability",
            label="Primitive availability check",
            options=["available", "constrained"],
            selected=f"{len(available)} available / {len(constrained)} constrained",
            reason=(
                f"Evaluated {len(available) + len(constrained)} primitives "
                "against current instrument and deck state"
            ),
            outcome=(
                f"constrained: {constrained[:3]}{'...' if len(constrained) > 3 else ''}"
            ),
        )
        logger.debug(
            "capability_agent: primitives available=%d constrained=%d",
            len(available),
            len(constrained),
            extra={"campaign_id": input_data.campaign_id},
        )

        # decision nodes captured; run() will store them in AgentResult
        _ = reachability_node
        _ = primitive_node

        return CapabilitySnapshot(
            robot_type=robot_type,
            robot_ip=robot_ip,
            robot_reachable=reachable,
            firmware_version=health_data.get("fw_version"),
            pipettes=pipettes,
            deck_slots=deck_slots,
            available_primitives=available,
            constrained_primitives=constrained,
            warnings=warnings,
            raw_instrument_data=raw_instrument_data,
        )

    # ── Private HTTP helpers ──────────────────────────────────────────────

    async def _probe_health(
        self, robot_ip: str, timeout_s: float
    ) -> tuple[bool, dict[str, Any]]:
        """GET /health — returns (reachable, health_payload)."""
        url = f"http://{robot_ip}:{_OPENTRONS_PORT}/health"
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.get(
                    url, headers={"Opentrons-Version": _OPENTRONS_API_VERSION}
                )
                resp.raise_for_status()
                return True, resp.json()
        except (httpx.HTTPError, httpx.TimeoutException, OSError) as exc:
            logger.warning("capability_agent: health probe failed: %s", exc)
            return False, {}

    async def _query_instruments(
        self, robot_ip: str, timeout_s: float
    ) -> tuple[list[PipetteInfo], dict[str, Any]]:
        """GET /instruments — returns (pipette list, raw response)."""
        url = f"http://{robot_ip}:{_OPENTRONS_PORT}/instruments"
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.get(
                    url, headers={"Opentrons-Version": _OPENTRONS_API_VERSION}
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
        except (httpx.HTTPError, httpx.TimeoutException, OSError) as exc:
            logger.warning("capability_agent: instrument query failed: %s", exc)
            return [], {}

        pipettes: list[PipetteInfo] = []
        for item in data.get("data", []):
            if item.get("instrumentType") != "pipette":
                continue
            details = item.get("data", {})
            pipettes.append(
                PipetteInfo(
                    mount=item.get("mount", "unknown"),
                    model=item.get("instrumentModel", "unknown"),
                    max_volume_ul=float(details.get("maxVolume", 0)),
                    tip_attached=details.get("tipLength") is not None,
                )
            )
        return pipettes, data

    async def _query_deck(
        self, robot_ip: str, timeout_s: float
    ) -> tuple[list[SlotInfo], list[str]]:
        """Infer deck state from the most recent run's labware manifest.

        The Opentrons HTTP API does not expose a dedicated deck-state endpoint;
        we use the latest run's labware list as a best-effort proxy.
        """
        url = f"http://{robot_ip}:{_OPENTRONS_PORT}/runs"
        warnings: list[str] = []
        slots: list[SlotInfo] = []

        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.get(
                    url, headers={"Opentrons-Version": _OPENTRONS_API_VERSION}
                )
                resp.raise_for_status()
                runs_data: dict[str, Any] = resp.json()
        except (httpx.HTTPError, httpx.TimeoutException, OSError) as exc:
            logger.warning("capability_agent: deck query failed: %s", exc)
            warnings.append(f"Could not query deck state: {exc}")
            return slots, warnings

        runs = runs_data.get("data", [])
        if not runs:
            warnings.append("No previous runs found; deck state is unknown.")
            return slots, warnings

        # Use the most recent run (index 0 = newest, per Opentrons API ordering)
        latest_run = runs[0]
        for lw in latest_run.get("labware", []):
            loc = lw.get("location", {})
            slot = str(loc.get("slotName", ""))
            if not slot:
                continue
            uri = lw.get("definitionUri", "")
            slots.append(
                SlotInfo(
                    slot=slot,
                    occupied=True,
                    labware_name=uri.split("/")[-1] if uri else None,
                    labware_uri=uri or None,
                )
            )

        return slots, warnings

    # ── Constraint rules ──────────────────────────────────────────────────

    _LIQUID_OPS: frozenset[str] = frozenset(
        {"liquid_transfer", "mix", "aspirate", "dispense", "pick_up_tip", "drop_tip"}
    )
    _TIP_REQUIRED: frozenset[str] = frozenset(
        {"pick_up_tip", "liquid_transfer", "mix"}
    )

    def _is_constrained(
        self,
        primitive_name: str,
        pipettes: list[PipetteInfo],
        deck_slots: list[SlotInfo],
    ) -> tuple[bool, str]:
        """Return (is_blocked, human_readable_reason).

        Rules are intentionally simple — the ValidationAgent does deep checks.
        Here we only block what is structurally impossible.
        """
        if primitive_name in self._LIQUID_OPS and not pipettes:
            return True, "no pipettes attached"

        # Only apply tip-rack check when we have actual deck data
        if primitive_name in self._TIP_REQUIRED and deck_slots:
            has_tip_rack = any(
                "tiprack" in (s.labware_name or "").lower() for s in deck_slots
            )
            if not has_tip_rack:
                return True, "no tip rack detected on deck"

        return False, ""

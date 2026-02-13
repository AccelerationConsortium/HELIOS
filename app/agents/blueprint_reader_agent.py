"""Blueprint Reader Agent — extract tool holder positions from PDF engineering drawings.

Wraps the ``pdf_layout_extractor`` service with the BaseAgent interface.
Supports multi-turn confirmation when extraction is incomplete.

Workflow:
1. Extract positions from PDF using regex + text analysis
2. If extraction is incomplete, present partial results and ask user to supplement
3. Build and save ToolHolderConfig

Layer: L1 (compilation helper)
"""
from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------


class BlueprintReaderInput(BaseModel):
    """Input for the BlueprintReader agent."""

    phase: str = Field(
        default="extract",
        description="Phase: extract | supplement | confirm",
    )

    # --- extract phase ---
    pdf_path: str = Field(default="", description="Path to the PDF file")
    pdf_base64: str = Field(
        default="",
        description="Base64-encoded PDF content (alternative to pdf_path)",
    )
    slot_number: int = Field(default=5, description="OT-2 deck slot for this holder")
    holder_name: str = Field(default="", description="Name for the holder config")
    scale_hint: float = Field(
        default=1.0,
        description="Scale factor if drawing is not in mm",
    )

    # --- supplement phase ---
    supplemented_positions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="User-provided positions to fill gaps",
    )

    # --- confirm phase ---
    confirmed: bool = False
    edits: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Position edits keyed by label → field overrides",
    )

    # --- carry-over state ---
    previous_result: dict[str, Any] | None = None


class BlueprintReaderOutput(BaseModel):
    """Output from the BlueprintReader agent."""

    status: str = Field(
        ...,
        description=(
            "Status: extracted | needs_supplement | awaiting_confirmation | "
            "finalized | error"
        ),
    )

    # Extraction results
    positions_found: int = 0
    high_confidence_count: int = 0

    # Resulting config (available after confirm)
    config_path: str = ""

    # Rendered page image (base64 PNG, for UI display)
    page_image_b64: str = ""

    # Chat message
    chat_message: str = ""

    # Warnings from extraction
    warnings: list[str] = Field(default_factory=list)

    # Serialised state
    serialised_result: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class BlueprintReaderAgent(BaseAgent[BlueprintReaderInput, BlueprintReaderOutput]):
    """Extract tool holder positions from PDF blueprints.

    Graceful degradation: if automatic extraction fails, presents partial
    results and asks the user to fill in the gaps.
    """

    name = "blueprint_reader"
    description = "PDF engineering drawing → ToolHolderConfig"
    layer = "L1"

    def validate_input(self, input_data: BlueprintReaderInput) -> list[str]:
        errors: list[str] = []
        if input_data.phase == "extract":
            if not input_data.pdf_path and not input_data.pdf_base64:
                errors.append("Either pdf_path or pdf_base64 is required")
            from app.services.deck_layout import valid_slot, RobotType
            rt = RobotType(getattr(input_data, 'robot_type', 'ot2') or 'ot2')
            if not valid_slot(input_data.slot_number, rt):
                errors.append(f"slot_number {input_data.slot_number} is invalid for {rt.value}")
        elif input_data.phase in ("supplement", "confirm"):
            if not input_data.previous_result:
                errors.append(f"previous_result is required for {input_data.phase} phase")
        else:
            errors.append(f"Unknown phase: {input_data.phase}")
        return errors

    async def process(self, input_data: BlueprintReaderInput) -> BlueprintReaderOutput:
        if input_data.phase == "extract":
            return self._handle_extract(input_data)
        elif input_data.phase == "supplement":
            return self._handle_supplement(input_data)
        elif input_data.phase == "confirm":
            return self._handle_confirm(input_data)
        else:
            raise ValueError(f"Unknown phase: {input_data.phase}")

    # ------------------------------------------------------------------ #
    # Phase handlers
    # ------------------------------------------------------------------ #

    def _handle_extract(self, input_data: BlueprintReaderInput) -> BlueprintReaderOutput:
        from app.services.pdf_layout_extractor import (
            extract_positions_from_pdf,
            render_pdf_page_to_image,
        )

        # Resolve PDF path
        pdf_path = input_data.pdf_path
        tmp_file = None

        if not pdf_path and input_data.pdf_base64:
            # Write base64 to temp file
            tmp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            tmp_file.write(base64.b64decode(input_data.pdf_base64))
            tmp_file.close()
            pdf_path = tmp_file.name

        try:
            result = extract_positions_from_pdf(
                pdf_path=pdf_path,
                scale=input_data.scale_hint,
            )
        except ImportError as exc:
            return BlueprintReaderOutput(
                status="error",
                chat_message=str(exc),
            )
        except FileNotFoundError:
            return BlueprintReaderOutput(
                status="error",
                chat_message=f"PDF file not found: {pdf_path}",
            )

        # Render first page for UI
        page_image_b64 = ""
        try:
            img_bytes = render_pdf_page_to_image(pdf_path, page_number=0, dpi=150)
            page_image_b64 = base64.b64encode(img_bytes).decode("ascii")
        except Exception:
            logger.debug("Could not render PDF page image", exc_info=True)

        # Classify results
        high_conf = [p for p in result.positions if p.confidence >= 0.5]
        low_conf = [p for p in result.positions if p.confidence < 0.5]

        state = {
            "pdf_path": pdf_path,
            "slot_number": input_data.slot_number,
            "holder_name": input_data.holder_name or f"holder_slot_{input_data.slot_number}",
            "scale": input_data.scale_hint,
            "positions": [
                {
                    "label": p.label,
                    "x_mm": p.x_mm,
                    "y_mm": p.y_mm,
                    "z_mm": p.z_mm,
                    "diameter_mm": p.diameter_mm,
                    "depth_mm": p.depth_mm,
                    "confidence": p.confidence,
                }
                for p in result.positions
            ],
            "holder_width_mm": result.holder_width_mm,
            "holder_height_mm": result.holder_height_mm,
            "holder_depth_mm": result.holder_depth_mm,
            "raw_texts": result.raw_texts,
            "warnings": result.warnings,
        }

        if not result.positions or low_conf:
            # Need user supplementation
            lines = ["## PDF Extraction Results\n"]
            if high_conf:
                lines.append(f"Found {len(high_conf)} high-confidence positions:")
                for p in high_conf:
                    lines.append(
                        f"  - **{p.label}**: ({p.x_mm:.1f}, {p.y_mm:.1f}) mm"
                    )
            if low_conf:
                lines.append(f"\n{len(low_conf)} positions need verification:")
                for p in low_conf:
                    lines.append(
                        f"  - **{p.label}**: coordinates uncertain "
                        f"(confidence {p.confidence:.0%})"
                    )
            if not result.positions:
                lines.append(
                    "No positions could be automatically extracted.\n"
                    "Please provide position data manually."
                )
            if result.warnings:
                lines.append("\n**Warnings:**")
                for w in result.warnings:
                    lines.append(f"  - {w}")

            lines.append(
                "\nPlease supplement with corrected/missing position data."
            )

            return BlueprintReaderOutput(
                status="needs_supplement",
                positions_found=len(result.positions),
                high_confidence_count=len(high_conf),
                page_image_b64=page_image_b64,
                chat_message="\n".join(lines),
                warnings=result.warnings,
                serialised_result=state,
            )

        # All positions extracted with good confidence
        return self._build_confirmation_output(state, page_image_b64)

    def _handle_supplement(self, input_data: BlueprintReaderInput) -> BlueprintReaderOutput:
        state = dict(input_data.previous_result or {})
        positions = list(state.get("positions", []))

        # Merge supplemented positions
        existing_labels = {p["label"] for p in positions}
        for sup in input_data.supplemented_positions:
            label = sup.get("label", "")
            if label in existing_labels:
                # Update existing
                for p in positions:
                    if p["label"] == label:
                        p.update(sup)
                        p["confidence"] = max(p.get("confidence", 0), 0.8)
                        break
            else:
                # Add new
                sup.setdefault("confidence", 0.9)
                positions.append(sup)

        state["positions"] = positions
        return self._build_confirmation_output(state)

    def _handle_confirm(self, input_data: BlueprintReaderInput) -> BlueprintReaderOutput:
        state = dict(input_data.previous_result or {})
        positions = list(state.get("positions", []))

        # Apply edits
        if input_data.edits:
            for label, overrides in input_data.edits.items():
                for p in positions:
                    if p["label"] == label:
                        p.update(overrides)
            state["positions"] = positions

        if not input_data.confirmed:
            return self._build_confirmation_output(state)

        # Build and save config
        from app.services.pdf_layout_extractor import (
            ExtractedPosition,
            build_tool_holder_from_positions,
        )
        from app.services.tool_holder_config import save_tool_holder_config

        extracted = [
            ExtractedPosition(
                label=p.get("label", ""),
                x_mm=p.get("x_mm", 0),
                y_mm=p.get("y_mm", 0),
                z_mm=p.get("z_mm", 0),
                diameter_mm=p.get("diameter_mm", 0),
                depth_mm=p.get("depth_mm", 0),
                confidence=p.get("confidence", 0),
            )
            for p in positions
        ]

        config = build_tool_holder_from_positions(
            positions=extracted,
            holder_name=state.get("holder_name", "holder"),
            slot_number=state.get("slot_number", 5),
            holder_width_mm=state.get("holder_width_mm", 127.76),
            holder_height_mm=state.get("holder_height_mm", 85.48),
            holder_depth_mm=state.get("holder_depth_mm", 60.0),
        )

        saved_path = save_tool_holder_config(config)

        return BlueprintReaderOutput(
            status="finalized",
            positions_found=len(positions),
            high_confidence_count=len(positions),
            config_path=saved_path,
            chat_message=(
                f"Tool holder config created from PDF and saved to `{saved_path}`.\n"
                f"{len(positions)} positions configured."
            ),
            warnings=state.get("warnings", []),
            serialised_result=state,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_confirmation_output(
        self,
        state: dict[str, Any],
        page_image_b64: str = "",
    ) -> BlueprintReaderOutput:
        positions = state.get("positions", [])
        lines = ["## Extracted Positions — Please Confirm\n"]
        for p in positions:
            conf_str = f"{p.get('confidence', 0):.0%}"
            lines.append(
                f"  - **{p.get('label', '?')}**: "
                f"({p.get('x_mm', 0):.1f}, {p.get('y_mm', 0):.1f}) mm, "
                f"confidence {conf_str}"
            )
        if state.get("holder_width_mm"):
            lines.append(
                f"\nHolder dimensions: "
                f"{state['holder_width_mm']:.1f} x {state.get('holder_height_mm', 0):.1f} mm"
            )
        lines.append("\nConfirm these positions or provide edits.")

        return BlueprintReaderOutput(
            status="awaiting_confirmation",
            positions_found=len(positions),
            high_confidence_count=sum(
                1 for p in positions if p.get("confidence", 0) >= 0.5
            ),
            page_image_b64=page_image_b64,
            chat_message="\n".join(lines),
            warnings=state.get("warnings", []),
            serialised_result=state,
        )

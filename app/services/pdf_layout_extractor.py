"""PDF Layout Extractor — extract tool holder positions from engineering drawings.

Parses PDF blueprints (dimensioned drawings of electrode/tool holders) to
extract physical position coordinates and build a
:class:`~app.services.tool_holder_config.ToolHolderConfig`.

Requires the optional ``pymupdf`` package (``pip install pymupdf``).
All public functions raise :class:`ImportError` with a helpful message
when ``pymupdf`` is not installed.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ExtractedPosition",
    "ExtractionResult",
    "extract_positions_from_pdf",
    "render_pdf_page_to_image",
    "parse_dimension_text",
    "build_tool_holder_from_positions",
]

# ---------------------------------------------------------------------------
# Optional dependency: PyMuPDF
# ---------------------------------------------------------------------------

try:
    import fitz  # PyMuPDF  # noqa: F401

    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False


def _require_fitz() -> None:
    if not _HAS_FITZ:
        raise ImportError(
            "PyMuPDF is required for PDF extraction. "
            "Install it with:  pip install pymupdf"
        )


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ExtractedPosition:
    """A single position extracted from a PDF drawing."""

    label: str = ""
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0
    diameter_mm: float = 0.0
    depth_mm: float = 0.0
    confidence: float = 0.0
    source_page: int = 0
    raw_text: str = ""


@dataclass
class ExtractionResult:
    """Aggregate result of PDF extraction."""

    positions: list[ExtractedPosition] = field(default_factory=list)
    page_count: int = 0
    holder_width_mm: float = 0.0
    holder_height_mm: float = 0.0
    holder_depth_mm: float = 0.0
    warnings: list[str] = field(default_factory=list)
    raw_texts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regex patterns for dimension parsing
# ---------------------------------------------------------------------------

# Match patterns like "25.4mm", "25.4 mm", "12.5MM", "Ø15mm"
_DIM_PATTERN = re.compile(
    r"(?P<prefix>[Øø∅]?)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|in|inch)",
    re.IGNORECASE,
)

# Match coordinate-like patterns: "(25.4, 30.0)" or "x=25.4 y=30.0"
_COORD_PATTERN = re.compile(
    r"(?:x\s*[=:]\s*(?P<x>\d+(?:\.\d+)?)).*?(?:y\s*[=:]\s*(?P<y>\d+(?:\.\d+)?))",
    re.IGNORECASE,
)

# Match position labels: "Pos 1", "Position A1", "P1", "Hole 1"
_LABEL_PATTERN = re.compile(
    r"(?:pos(?:ition)?|hole|slot|well|point)\s*[#]?\s*(?P<label>[A-Za-z]?\d+)",
    re.IGNORECASE,
)

# Match depth annotations: "depth 30mm", "D=25.4"
_DEPTH_PATTERN = re.compile(
    r"(?:depth|deep|D)\s*[=:]\s*(?P<value>\d+(?:\.\d+)?)\s*(?:mm)?",
    re.IGNORECASE,
)

# Match diameter annotations: "Ø15", "dia 20mm", "diameter=15"
_DIAMETER_PATTERN = re.compile(
    r"(?:[Øø∅]|dia(?:meter)?)\s*[=:]\s*(?P<value>\d+(?:\.\d+)?)\s*(?:mm)?",
    re.IGNORECASE,
)

# Match overall dimensions: "127.76 x 85.48 x 50mm"
_OVERALL_DIM_PATTERN = re.compile(
    r"(?P<w>\d+(?:\.\d+)?)\s*[xX×]\s*(?P<h>\d+(?:\.\d+)?)"
    r"(?:\s*[xX×]\s*(?P<d>\d+(?:\.\d+)?))?"
    r"\s*(?:mm)?",
)


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------


def parse_dimension_text(text: str) -> dict[str, Any]:
    """Parse dimension annotations from raw text.

    Returns a dict with keys like ``dimensions``, ``coordinates``,
    ``labels``, ``depths``, ``diameters``, ``overall``.
    """
    result: dict[str, Any] = {
        "dimensions": [],
        "coordinates": [],
        "labels": [],
        "depths": [],
        "diameters": [],
        "overall": None,
    }

    for m in _DIM_PATTERN.finditer(text):
        value = float(m.group("value"))
        unit = m.group("unit").lower()
        prefix = m.group("prefix") or ""
        if unit == "cm":
            value *= 10.0
        elif unit in ("in", "inch"):
            value *= 25.4
        result["dimensions"].append({
            "value_mm": value,
            "is_diameter": bool(prefix),
            "raw": m.group(0),
        })

    for m in _COORD_PATTERN.finditer(text):
        result["coordinates"].append({
            "x": float(m.group("x")),
            "y": float(m.group("y")),
        })

    for m in _LABEL_PATTERN.finditer(text):
        result["labels"].append(m.group("label"))

    for m in _DEPTH_PATTERN.finditer(text):
        result["depths"].append(float(m.group("value")))

    for m in _DIAMETER_PATTERN.finditer(text):
        result["diameters"].append(float(m.group("value")))

    m = _OVERALL_DIM_PATTERN.search(text)
    if m:
        overall: dict[str, float] = {
            "width": float(m.group("w")),
            "height": float(m.group("h")),
        }
        if m.group("d"):
            overall["depth"] = float(m.group("d"))
        result["overall"] = overall

    return result


def render_pdf_page_to_image(
    pdf_path: str,
    page_number: int = 0,
    dpi: int = 200,
) -> bytes:
    """Render a single PDF page to a PNG image.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    page_number:
        Zero-based page index.
    dpi:
        Resolution for rendering.

    Returns
    -------
    bytes
        PNG image data.
    """
    _require_fitz()
    import fitz  # noqa: F811

    doc = fitz.open(pdf_path)
    try:
        if page_number >= len(doc):
            raise IndexError(
                f"Page {page_number} out of range (PDF has {len(doc)} pages)"
            )
        page = doc[page_number]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
    finally:
        doc.close()


def extract_positions_from_pdf(
    pdf_path: str,
    scale: float = 1.0,
    pages: list[int] | None = None,
) -> ExtractionResult:
    """Extract tool holder positions from a PDF engineering drawing.

    Workflow:
    1. Extract all text from specified pages (or all pages).
    2. Parse dimension annotations, coordinate labels, diameters, depths.
    3. Attempt to identify individual position holes and their coordinates.
    4. If coordinate extraction fails, fall back to collecting raw
       dimension data and flag ``warnings`` for user supplementation.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    scale:
        Scale factor to convert drawing units to mm (default 1.0 = already mm).
    pages:
        Specific page indices to process (zero-based). ``None`` = all pages.

    Returns
    -------
    ExtractionResult
        Extracted positions and metadata. Check ``warnings`` for any issues.
    """
    _require_fitz()
    import fitz  # noqa: F811

    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    result = ExtractionResult(page_count=len(doc))

    try:
        page_indices = pages if pages is not None else list(range(len(doc)))

        all_text_parts: list[str] = []

        for page_idx in page_indices:
            if page_idx >= len(doc):
                result.warnings.append(f"Page {page_idx} out of range, skipped")
                continue

            page = doc[page_idx]
            text = page.get_text("text")
            all_text_parts.append(text)
            result.raw_texts.append(text)

            parsed = parse_dimension_text(text)

            # Extract overall dimensions if found
            if parsed["overall"] and result.holder_width_mm == 0.0:
                result.holder_width_mm = parsed["overall"]["width"] * scale
                result.holder_height_mm = parsed["overall"]["height"] * scale
                if "depth" in parsed["overall"]:
                    result.holder_depth_mm = parsed["overall"]["depth"] * scale

            # Try to match labels with coordinates
            labels = parsed["labels"]
            coords = parsed["coordinates"]
            diameters = parsed["diameters"]
            depths = parsed["depths"]

            if labels and coords and len(labels) == len(coords):
                # Best case: matched labels and coordinates
                for i, label in enumerate(labels):
                    pos = ExtractedPosition(
                        label=label,
                        x_mm=coords[i]["x"] * scale,
                        y_mm=coords[i]["y"] * scale,
                        confidence=0.8,
                        source_page=page_idx,
                    )
                    if i < len(diameters):
                        pos.diameter_mm = diameters[i] * scale
                    if i < len(depths):
                        pos.depth_mm = depths[i] * scale
                    result.positions.append(pos)

            elif coords:
                # Have coordinates but no matching labels — auto-label
                for i, coord in enumerate(coords):
                    pos = ExtractedPosition(
                        label=f"P{i + 1}",
                        x_mm=coord["x"] * scale,
                        y_mm=coord["y"] * scale,
                        confidence=0.5,
                        source_page=page_idx,
                    )
                    if i < len(diameters):
                        pos.diameter_mm = diameters[i] * scale
                    if i < len(depths):
                        pos.depth_mm = depths[i] * scale
                    result.positions.append(pos)
                result.warnings.append(
                    f"Page {page_idx}: coordinates found but labels missing; "
                    f"auto-assigned P1..P{len(coords)}"
                )

            elif labels:
                # Have labels but no coordinates — partial extraction
                for label in labels:
                    pos = ExtractedPosition(
                        label=label,
                        confidence=0.2,
                        source_page=page_idx,
                        raw_text=text[:200],
                    )
                    result.positions.append(pos)
                result.warnings.append(
                    f"Page {page_idx}: position labels found ({labels}) "
                    f"but coordinates could not be extracted. "
                    f"User input needed for x/y positions."
                )

            else:
                # No structured data found — collect raw dimensions
                dims = parsed["dimensions"]
                if dims:
                    result.warnings.append(
                        f"Page {page_idx}: found {len(dims)} dimension annotations "
                        f"but could not identify individual positions. "
                        f"User may need to specify positions manually."
                    )
                else:
                    result.warnings.append(
                        f"Page {page_idx}: no dimension annotations found in text. "
                        f"The PDF may contain only vector graphics. "
                        f"Consider providing positions via dialog instead."
                    )

        if not result.positions:
            result.warnings.append(
                "No positions could be automatically extracted. "
                "The blueprint reader agent will present raw text "
                "and ask the user to supplement missing information."
            )

    finally:
        doc.close()

    logger.info(
        "PDF extraction from '%s': %d positions, %d warnings",
        pdf_path,
        len(result.positions),
        len(result.warnings),
    )
    return result


# ---------------------------------------------------------------------------
# Bridge to ToolHolderConfig
# ---------------------------------------------------------------------------


def build_tool_holder_from_positions(
    positions: list[ExtractedPosition],
    holder_name: str,
    slot_number: int,
    labware_name: str = "custom_tool_holder_4pos",
    holder_width_mm: float = 127.76,
    holder_height_mm: float = 85.48,
    holder_depth_mm: float = 60.0,
) -> "ToolHolderConfig":
    """Convert extracted PDF positions into a ToolHolderConfig.

    Maps each :class:`ExtractedPosition` to a well reference and offset
    based on quadrant heuristics (positions in upper-left map to A1, etc.).

    Parameters
    ----------
    positions:
        Extracted positions from :func:`extract_positions_from_pdf`.
    holder_name:
        Name for the holder configuration.
    slot_number:
        OT-2 deck slot number.
    labware_name:
        Labware load name for the holder.
    holder_width_mm, holder_height_mm, holder_depth_mm:
        Physical dimensions of the holder.

    Returns
    -------
    ToolHolderConfig
        Ready-to-use configuration.
    """
    from app.services.tool_holder_config import ToolHolderConfig, ToolPosition

    # Quadrant mapping: divide holder into 2x2 grid
    mid_x = holder_width_mm / 2
    mid_y = holder_height_mm / 2

    well_map = {
        "upper-left": "A1",
        "upper-right": "A2",
        "lower-left": "B1",
        "lower-right": "B2",
    }

    tool_positions: list[ToolPosition] = []
    for pos in positions:
        # Determine quadrant
        if pos.x_mm <= mid_x:
            quadrant = "upper-left" if pos.y_mm >= mid_y else "lower-left"
        else:
            quadrant = "upper-right" if pos.y_mm >= mid_y else "lower-right"

        well = well_map[quadrant]

        # Compute offset from well center (approximate well centers)
        well_centers = {
            "A1": (32.0, 60.0),
            "A2": (96.0, 60.0),
            "B1": (32.0, 25.0),
            "B2": (96.0, 25.0),
        }
        cx, cy = well_centers[well]

        tool_positions.append(ToolPosition(
            name=pos.label or f"pos_{len(tool_positions) + 1}",
            well_name=well,
            offset_x=round(pos.x_mm - cx, 2),
            offset_y=round(pos.y_mm - cy, 2),
            offset_z=0.0,
            tool_type="",
            quadrant=quadrant,
            description=f"Extracted from PDF (confidence={pos.confidence:.1%})",
        ))

    return ToolHolderConfig(
        holder_name=holder_name,
        slot_number=slot_number,
        labware_name=labware_name,
        positions=tool_positions,
        holder_dimensions={
            "x_total": holder_width_mm,
            "y_total": holder_height_mm,
            "z_height": holder_depth_mm,
        },
        created_by="pdf_reader",
    )

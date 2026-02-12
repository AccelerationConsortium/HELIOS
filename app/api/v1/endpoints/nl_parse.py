"""Natural-language intent parsing — free-text → OrchestratorInput fields.

Extracts experiment parameters from a single paragraph of text
(supports mixed Chinese + English) and converts them into structured
fields suitable for launching an orchestrator campaign.
"""
from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/nl", tags=["nl"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class NLParseRequest(BaseModel):
    """A single paragraph describing the experiment."""
    text: str


class ExtractedParam(BaseModel):
    """One extracted parameter with its source span."""
    key: str
    value: Any
    raw_span: str = ""
    confidence: float = 1.0


class NLParseResponse(BaseModel):
    """Structured extraction result from free-text."""
    # Core campaign fields (may be None if not detected)
    objective_kpi: str | None = None
    direction: str | None = None
    max_rounds: int | None = None
    target_value: float | None = None
    batch_size: int | None = None
    strategy: str | None = None
    protocol_pattern_id: str | None = None

    # Extracted dimensions (parameter space)
    dimensions: list[dict[str, Any]] = Field(default_factory=list)

    # Slot / labware mentions
    slot_assignments: dict[str, str] = Field(default_factory=dict)

    # Instrument detection
    detected_instruments: list[str] = Field(
        default_factory=list,
        description="Instrument short names recognised from the text",
    )
    unknown_instruments: list[str] = Field(
        default_factory=list,
        description="Instrument-like mentions that are not in the registry",
    )
    onboarding_suggested: bool = Field(
        default=False,
        description="True when unknown instruments are detected — client should offer onboarding",
    )

    # All extracted parameters with provenance
    extracted: list[ExtractedParam] = Field(default_factory=list)

    # Anything we couldn't parse
    unparsed_fragments: list[str] = Field(default_factory=list)

    # Original text echoed back
    original_text: str = ""


# ---------------------------------------------------------------------------
# Extraction helpers (regex-based, works for CN + EN mixed text)
# ---------------------------------------------------------------------------

_DIRECTION_PATTERNS = [
    # Chinese
    (r"最小化|最小|越小越好|小于|降低|减少|minimize", "minimize"),
    (r"最大化|最大|越大越好|大于|提高|增加|maximize", "maximize"),
    # English
    (r"\bminimize\b|\blower\b|\breduce\b|\bdecrease\b", "minimize"),
    (r"\bmaximize\b|\bhigher\b|\bincrease\b|\bimprove\b", "maximize"),
]

_KPI_PATTERNS = [
    # Common lab KPIs
    (r"(CV|cv|变异系数|coefficient\s+of\s+variation)", "cv"),
    (r"(accuracy|准确[度率])", "accuracy"),
    (r"(precision|精密[度率])", "precision"),
    (r"(yield|产率|产量)", "yield"),
    (r"(absorbance|吸光度|OD值?)", "absorbance"),
    (r"(fluorescence|荧光[强度值]?)", "fluorescence"),
    (r"(concentration|浓度)", "concentration"),
    (r"(viscosity|粘度)", "viscosity"),
]

_ROUND_PATTERNS = [
    (r"(\d+)\s*(?:轮|rounds?|次|批|batches)", int),
    (r"(?:max|最多|不超过)\s*(\d+)\s*(?:轮|rounds?|次)", int),
]

_TARGET_PATTERNS = [
    (r"(?:小于|<|less\s+than|低于|below)\s*([\d.]+)", float),
    (r"(?:大于|>|greater\s+than|高于|above)\s*([\d.]+)", float),
    (r"(?:target|目标)[=:：]?\s*([\d.]+)", float),
]

_BATCH_PATTERNS = [
    (r"(?:batch[_ ]?size|每批|每轮)\s*[=:：]?\s*(\d+)", int),
    (r"(\d+)\s*(?:个样品?|samples?|candidates?|per\s+round)", int),
]

_SLOT_PATTERN = re.compile(
    r"(?:slot|槽位?)\s*(\d{1,2})\s*[=:：放上]?\s*([^\s,;，；、]+)",
    re.IGNORECASE,
)

_DIMENSION_PATTERN = re.compile(
    r"(\w+)\s*[=:：]?\s*\[?\s*([\d.]+)\s*[-–~到]\s*([\d.]+)\s*\]?\s*(?:([µuμ]?[lLmMgG]+|%|rpm|°?[cC]?))?",
)

_STRATEGY_PATTERNS = [
    (r"(bayesian|贝叶斯)", "bayesian"),
    (r"(lhs|拉丁超立方)", "lhs"),
    (r"(random|随机)", "random"),
    (r"(grid|网格)", "grid"),
]

_PROTOCOL_PATTERNS = [
    (r"(serial[_ ]?dilution|连续稀释|梯度稀释)", "serial_dilution"),
    (r"(normalization|归一化|标准化)", "normalization"),
    (r"(mixing|混合|搅拌)", "mixing"),
    (r"(transfer|转移|移液)", "transfer"),
    (r"(dispensing|分液|分装)", "dispensing"),
]

# Instrument detection patterns (known instruments, CN + EN)
_INSTRUMENT_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)(OT[\s-]?2|opentrons|移液工作站|液体处理)", "ot2"),
    (r"(?i)\b(PLC|可编程逻辑控制器)\b", "plc"),
    (r"(?i)\b(relay|继电器)\b", "relay"),
    (r"(?i)(squidstat|电化学工作站|恒电位仪|电位仪)", "squidstat"),
    (r"(?i)(furnace|马弗炉|管式炉|加热炉|退火炉)", "furnace"),
    (r"(?i)(spin[\s_-]?coat(?:er|ing)?|旋涂(?:仪|机)?)", "spin_coater"),
]

# Generic instrument-like patterns — used to detect mentions of unknown
# instruments that might need onboarding.
_UNKNOWN_INSTRUMENT_RE = re.compile(
    r"(?i)"
    r"(?:(?:用|使用|连接|接入|配置)\s*)?"          # optional Chinese verb prefix
    r"([A-Z][A-Za-z0-9]{2,}[\s_-]?(?:仪|机|器|计|台)?)"  # Capitalised name + optional CN suffix
    r"|"
    r"([\u4e00-\u9fff]{2,}(?:仪|机|器|计|台|炉))"   # Chinese instrument name ending in 仪/机/器 etc.
)


def _find_first(text: str, patterns: list[tuple]) -> tuple[Any, str]:
    """Return (extracted_value, matched_span) for the first matching pattern."""
    for pattern, value in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return value, m.group(0)
    return None, ""


def _find_number(text: str, patterns: list[tuple]) -> tuple[Any, str]:
    """Extract a numeric value using the first matching pattern."""
    for pattern, converter in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return converter(m.group(1)), m.group(0)
    return None, ""


def parse_nl_text(text: str) -> NLParseResponse:
    """Parse a free-text experiment description into structured fields."""
    extracted: list[ExtractedParam] = []
    resp = NLParseResponse(original_text=text)

    # Direction
    direction, span = _find_first(text, _DIRECTION_PATTERNS)
    if direction:
        resp.direction = direction
        extracted.append(ExtractedParam(key="direction", value=direction, raw_span=span))

    # KPI
    kpi, span = _find_first(text, _KPI_PATTERNS)
    if kpi:
        resp.objective_kpi = kpi
        extracted.append(ExtractedParam(key="objective_kpi", value=kpi, raw_span=span))

    # Max rounds
    max_rounds, span = _find_number(text, _ROUND_PATTERNS)
    if max_rounds:
        resp.max_rounds = max_rounds
        extracted.append(ExtractedParam(key="max_rounds", value=max_rounds, raw_span=span))

    # Target value
    target, span = _find_number(text, _TARGET_PATTERNS)
    if target:
        resp.target_value = target
        extracted.append(ExtractedParam(key="target_value", value=target, raw_span=span))

    # Batch size
    batch, span = _find_number(text, _BATCH_PATTERNS)
    if batch:
        resp.batch_size = batch
        extracted.append(ExtractedParam(key="batch_size", value=batch, raw_span=span))

    # Strategy
    strategy, span = _find_first(text, _STRATEGY_PATTERNS)
    if strategy:
        resp.strategy = strategy
        extracted.append(ExtractedParam(key="strategy", value=strategy, raw_span=span))

    # Protocol pattern
    pattern_id, span = _find_first(text, _PROTOCOL_PATTERNS)
    if pattern_id:
        resp.protocol_pattern_id = pattern_id
        extracted.append(ExtractedParam(key="protocol_pattern_id", value=pattern_id, raw_span=span))

    # Slot assignments
    for m in _SLOT_PATTERN.finditer(text):
        slot_num = m.group(1)
        labware = m.group(2)
        resp.slot_assignments[f"slot_{slot_num}"] = labware
        extracted.append(ExtractedParam(
            key=f"slot_{slot_num}", value=labware, raw_span=m.group(0)
        ))

    # Dimensions (parameter ranges)
    for m in _DIMENSION_PATTERN.finditer(text):
        name = m.group(1)
        low = float(m.group(2))
        high = float(m.group(3))
        unit = m.group(4) or ""
        dim = {"name": name, "low": low, "high": high}
        if unit:
            dim["unit"] = unit
        resp.dimensions.append(dim)
        extracted.append(ExtractedParam(
            key=f"dim_{name}", value=dim, raw_span=m.group(0)
        ))

    # Instrument detection
    detected_instruments: set[str] = set()
    for pattern, instr_name in _INSTRUMENT_PATTERNS:
        m = re.search(pattern, text)
        if m:
            detected_instruments.add(instr_name)
            extracted.append(ExtractedParam(
                key="instrument", value=instr_name, raw_span=m.group(0),
            ))
    resp.detected_instruments = sorted(detected_instruments)

    # Unknown instrument detection — cross-check against registry
    known_short_names = _get_known_instrument_names()
    unknown: list[str] = []
    for m in _UNKNOWN_INSTRUMENT_RE.finditer(text):
        mention = (m.group(1) or m.group(2) or "").strip()
        if not mention or len(mention) < 2:
            continue
        # Normalise: lowercase for matching
        normalised = mention.lower().replace(" ", "_").replace("-", "_")
        # Skip if it's already a known instrument or matched by known patterns
        if normalised in known_short_names or normalised in detected_instruments:
            continue
        # Skip common false positives (generic lab terms, units, etc.)
        if normalised in _INSTRUMENT_FALSE_POSITIVES:
            continue
        unknown.append(mention)

    if unknown:
        resp.unknown_instruments = unknown
        resp.onboarding_suggested = True
        for unk in unknown:
            extracted.append(ExtractedParam(
                key="unknown_instrument", value=unk, raw_span=unk, confidence=0.6,
            ))

    resp.extracted = extracted
    return resp


# ---------------------------------------------------------------------------
# Helpers for instrument detection
# ---------------------------------------------------------------------------

# Common false positives — words that look like instruments but aren't
_INSTRUMENT_FALSE_POSITIVES = frozenset({
    "experiment", "optimization", "protocol", "parameter", "sample",
    "reagent", "solution", "buffer", "analysis", "result", "data",
    "target", "batch", "round", "strategy", "objective",
    "实验", "优化", "协议", "参数", "样品", "试剂", "溶液",
    "缓冲液", "分析", "结果", "数据", "目标",
})


def _get_known_instrument_names() -> set[str]:
    """Return all known instrument short names (from registry + fallback)."""
    try:
        from app.services.primitives_registry import get_registry
        names = get_registry().list_instrument_short_names()
        if names:
            return set(names)
    except Exception:
        pass
    # Fallback: hardcoded known names
    return {"ot2", "plc", "relay", "squidstat", "furnace", "spin_coater"}


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

@router.post("/parse", response_model=NLParseResponse)
async def parse_intent(payload: NLParseRequest) -> NLParseResponse:
    """Parse free-text experiment description into structured parameters."""
    return parse_nl_text(payload.text)

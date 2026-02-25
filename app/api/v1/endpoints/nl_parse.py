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

    # Fields that were expected but not found
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Required fields that could not be extracted from the text",
    )

    # Original text echoed back
    original_text: str = ""


# ---------------------------------------------------------------------------
# Extraction helpers (regex-based, works for CN + EN mixed text)
# ---------------------------------------------------------------------------

_DIRECTION_PATTERNS = [
    # Chinese
    (r"最小化|最小|越小越好|小于|降低|减少|减小", "minimize"),
    (r"最大化|最大|越大越好|大于|提高|增加|增大|提升", "maximize"),
    # English — keywords
    (r"\bminimize\b|\bminimum\b|\bminimal\b|\blower\b|\breduce\b|\bdecrease\b", "minimize"),
    (r"\bmaximize\b|\bmaximum\b|\bmaximal\b|\bhigher\b|\bincrease\b|\bimprove\b|\benhance\b|\bboost\b", "maximize"),
    # Natural-language phrases
    (r"(?:want|aim|try|goal|need)\s+to\s+(?:reduce|lower|decrease|minimize|cut|shrink|suppress)", "minimize"),
    (r"(?:want|aim|try|goal|need)\s+to\s+(?:increase|raise|boost|maximize|improve|enhance|push)", "maximize"),
    (r"as\s+(?:low|small|little|few)\s+as\s+possible", "minimize"),
    (r"as\s+(?:high|large|great|big|much)\s+as\s+possible", "maximize"),
    (r"keep\s+(?:it\s+)?(?:low|minimal|minimum)", "minimize"),
    (r"keep\s+(?:it\s+)?(?:high|maximal|maximum)", "maximize"),
]

_KPI_PATTERNS = [
    # ── Pipetting / liquid handling ───────────────────────────────────────────
    (r"\b(CV|cv|变异系数|coefficient\s+of\s+variation)\b", "cv"),
    (r"\b(accuracy|准确[度率])\b", "accuracy"),
    (r"\b(precision|精密[度率])\b", "precision"),
    (r"\b(yield|产率|产量)\b", "yield"),
    (r"\b(absorbance|吸光度|OD值?)\b", "absorbance"),
    (r"\b(fluorescence|荧光[强度值]?)\b", "fluorescence"),
    (r"\b(concentration|浓度)\b", "concentration"),
    (r"\b(viscosity|粘度)\b", "viscosity"),
    # ── Electrochemistry ─────────────────────────────────────────────────────
    (r"\b(overpotential|过电位|η10|eta[\s_]?10|eta)\b", "overpotential"),
    (r"\b(current[\s_]?density|电流密度|j[\s_]?\d+|j\b)", "current_density"),
    (r"\b(power[\s_]?density|功率密度)\b", "power_density"),
    (r"\b(faradaic[\s_]?efficiency|法拉第效率|FE)\b", "faradaic_efficiency"),
    (r"\b(onset[\s_]?potential|起始电位)\b", "onset_potential"),
    (r"\b(tafel[\s_]?slope|塔菲尔斜率)\b", "tafel_slope"),
    (r"\b(impedance|EIS|电化学阻抗)\b", "eis"),
    (r"\b(charge[\s_]?transfer[\s_]?resistance|Rct|电荷转移阻抗)\b", "rct"),
    (r"\b(OER|oxygen[\s_]?evolution)\b", "oer_activity"),
    (r"\b(HER|hydrogen[\s_]?evolution)\b", "her_activity"),
    # ── Biology / biochemistry ────────────────────────────────────────────────
    (r"\b(OD\s?600|OD600|optical\s+density)\b", "od600"),
    (r"\b(cell[\s_]?viability|细胞活力|细胞存活率)\b", "viability"),
    (r"\b(protein[\s_]?(?:yield|content|expression)|蛋白(?:产量|表达))\b", "protein_yield"),
    (r"\b(growth[\s_]?rate|生长速率)\b", "growth_rate"),
    # ── Materials science ─────────────────────────────────────────────────────
    (r"\b(conductivity|电导率|电导)\b", "conductivity"),
    (r"\b(pH|酸碱度)\b", "ph"),
    (r"\b(turbidity|浊度|浑浊度)\b", "turbidity"),
    (r"\b(purity|纯度|纯净度)\b", "purity"),
    (r"\b(efficiency|效率|能效)\b", "efficiency"),
    (r"\b(selectivity|选择性|选择率)\b", "selectivity"),
    (r"\b(conversion|转化率|转化度)\b", "conversion"),
    (r"\b(throughput|吞吐量|通量)\b", "throughput"),
    (r"\b(FOM|figure[\s_]?of[\s_]?merit|品质因数)\b", "fom"),
    (r"\b(hardness|硬度)\b", "hardness"),
    (r"\b(roughness|粗糙度)\b", "roughness"),
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

# Multiple dimension patterns to catch natural language variations
_DIMENSION_PATTERNS: list[re.Pattern[str]] = [
    # "name = [low-high] unit" or "name: low-high"
    re.compile(
        r"(\w+)\s*[=:：]?\s*\[?\s*([\d.]+)\s*[-–~到]\s*([\d.]+)\s*\]?"
        r"\s*(?:([µuμ]?[lLmMgGsS]+|%|rpm|°?[cC]?))?",
    ),
    # "name between A and B [unit]" or "name ranging from A to B"
    re.compile(
        r"(\w+)\s+(?:between|ranging\s+from|from)\s+([\d.]+)\s+(?:to|and|–|-)\s+([\d.]+)"
        r"\s*([µuμ]?[lLmMgGsS]+|%|rpm|°?[cC]?)?",
        re.IGNORECASE,
    ),
    # "name of A to B [unit]"
    re.compile(
        r"(\w+)\s+of\s+([\d.]+)\s*(?:to|–|-)\s*([\d.]+)"
        r"\s*([µuμ]?[lLmMgGsS]+|%|rpm|°?[cC]?)?",
        re.IGNORECASE,
    ),
    # "A to B [unit] (for|of) name"
    re.compile(
        r"([\d.]+)\s*(?:to|–|-)\s*([\d.]+)\s*([µuμ]?[lLmMgGsS]+|%|rpm|°?[cC]?)\s+"
        r"(?:for|of|in)\s+(\w+)",
        re.IGNORECASE,
    ),
]

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
# NOTE: NO (?i) flag — we require genuine uppercase first letter (proper nouns only).
_UNKNOWN_INSTRUMENT_RE = re.compile(
    r"(?:(?:用|使用|连接|接入|配置)\s*)?"          # optional Chinese verb prefix
    r"([A-Z][A-Za-z0-9]{2,}[\s_-]?(?:仪|机|器|计|台)?)"  # Capitalised proper noun + optional CN suffix
    r"|"
    r"([\u4e00-\u9fff]{2,}(?:仪|机|器|计|台|炉))"   # Chinese instrument name ending in 仪/机/器 etc.
)


# Separators used to split remaining text into clause fragments
_SEP_RE = re.compile(r"[,;，；、。\n\r]+")

# English stop-words filtered out when deciding if a fragment is meaningful
_STOP_WORDS = frozenset({
    "and", "or", "the", "a", "an", "is", "are", "was", "were",
    "to", "in", "at", "of", "with", "for", "on", "from", "by",
    "that", "this", "be", "have", "has", "do", "it", "we", "i",
})

# Required campaign fields checked for completeness
_REQUIRED_FIELDS: list[str] = [
    "objective_kpi", "direction", "dimensions", "max_rounds",
]


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

    # Dimensions (parameter ranges) — try all patterns in order
    seen_dim_names: set[str] = set()
    for pat_idx, dim_pat in enumerate(_DIMENSION_PATTERNS):
        for m in dim_pat.finditer(text):
            groups = m.groups()
            # Pattern 3 (A-to-B for name) has reversed group order
            if pat_idx == 3:
                low_raw, high_raw, unit_raw, name = groups[0], groups[1], groups[2] or "", groups[3]
            else:
                name, low_raw, high_raw, unit_raw = groups[0], groups[1], groups[2], groups[3] or ""
            # Skip common stop-words / single-char matches
            if not name or len(name) < 2 or name.lower() in {"in", "at", "to", "of", "a", "an"}:
                continue
            if name in seen_dim_names:
                continue
            seen_dim_names.add(name)
            low = float(low_raw)
            high = float(high_raw)
            dim: dict[str, Any] = {"name": name, "low": low, "high": high}
            if unit_raw:
                dim["unit"] = unit_raw
            resp.dimensions.append(dim)
            extracted.append(ExtractedParam(
                key=f"dim_{name}", value=dim, raw_span=m.group(0),
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
        # Cap to 5 to avoid flooding the UI with false positives from long input
        resp.unknown_instruments = unknown[:5]
        resp.onboarding_suggested = True
        for unk in unknown[:5]:
            extracted.append(ExtractedParam(
                key="unknown_instrument", value=unk, raw_span=unk, confidence=0.6,
            ))

    # ── Missing fields ────────────────────────────────────────────────────────
    missing: list[str] = []
    if not resp.objective_kpi:
        missing.append("objective_kpi")
    if not resp.direction:
        missing.append("direction")
    if not resp.dimensions:
        missing.append("dimensions")
    if not resp.max_rounds:
        missing.append("max_rounds")
    resp.missing_fields = missing

    # ── Unparsed fragments ────────────────────────────────────────────────────
    # Blank every matched raw_span from a working copy of the text, then
    # collect clause-level chunks that still have meaningful content.
    remaining = text
    for ep in extracted:
        if ep.raw_span:
            remaining = remaining.replace(ep.raw_span, " ")

    unparsed: list[str] = []
    for chunk in _SEP_RE.split(remaining):
        chunk = chunk.strip()
        if not chunk:
            continue
        words = chunk.split()
        meaningful = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 1]
        if meaningful:
            unparsed.append(chunk)
    resp.unparsed_fragments = unparsed

    resp.extracted = extracted
    return resp


# ---------------------------------------------------------------------------
# Helpers for instrument detection
# ---------------------------------------------------------------------------

# Common false positives — words that look like instruments but aren't.
# All entries MUST be lowercase because comparison is against normalised
# (mention.lower()). Mixed-case entries would never match.
_INSTRUMENT_FALSE_POSITIVES = frozenset({
    # Lab / experiment generic terms
    "experiment", "optimization", "protocol", "parameter", "sample",
    "reagent", "solution", "buffer", "analysis", "result", "data",
    "target", "batch", "round", "strategy", "objective", "agent",
    "model", "system", "platform", "workflow", "pipeline", "process",
    "method", "methods", "procedure", "step", "task", "job", "run", "campaign",
    "baseline", "control", "standard", "reference", "configuration",
    # Common English prompt / doc words (lowercase — normalised comparison)
    "role", "goal", "you", "your", "the", "this", "that", "these",
    "please", "note", "use", "when", "how", "what", "where", "which",
    "important", "description", "instructions", "overview", "summary",
    "input", "output", "context", "config", "settings", "options",
    "version", "type", "name", "value", "unit", "range", "format",
    "list", "dict", "json", "api", "url", "http", "true", "false",
    "none", "null", "nan", "inf", "max", "min", "mean", "std",
    "lab", "user", "admin", "tool", "class", "mode", "state",
    "error", "warning", "info", "debug", "log", "event", "message",
    "phase", "stage", "status", "code", "key", "field", "column",
    "row", "table", "index", "id", "uuid", "hash",
    "instruction", "example", "section", "note", "tip", "rule",
    # Chinese generic terms
    "实验", "优化", "协议", "参数", "样品", "试剂", "溶液",
    "缓冲液", "分析", "结果", "数据", "目标", "代理", "系统",
    "平台", "工作流", "流程", "方法", "步骤", "任务", "批次",
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

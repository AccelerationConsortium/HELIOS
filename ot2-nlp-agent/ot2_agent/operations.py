"""
Operation definitions and mapping for OT-2 robot commands.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class OperationType(Enum):
    """Types of OT-2 operations."""
    # Tip operations
    PICK_UP_TIP = "pick_up_tip"
    DROP_TIP = "drop_tip"

    # Liquid handling
    ASPIRATE = "aspirate"
    DISPENSE = "dispense"
    BLOWOUT = "blowout"
    MIX = "mix"
    TRANSFER = "transfer"

    # Movement
    MOVE_TO = "move_to"
    HOME = "home"

    # Labware
    LOAD_LABWARE = "load_labware"
    LOAD_PIPETTE = "load_pipette"

    # Utilities
    PAUSE = "pause"
    COMMENT = "comment"
    SET_TEMPERATURE = "set_temperature"
    WAIT = "wait"


@dataclass
class Operation:
    """A single OT-2 operation."""
    type: OperationType
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    line_number: Optional[int] = None

    def to_python(self) -> str:
        """Convert operation to Python code."""
        return OPERATION_TEMPLATES.get(self.type, lambda p: f"# {self.type}")(self.params)

    def validate(self) -> List[str]:
        """Validate operation parameters. Returns list of errors."""
        errors = []
        required = REQUIRED_PARAMS.get(self.type, [])
        for param in required:
            if param not in self.params:
                errors.append(f"Missing required parameter: {param}")
        return errors


# Required parameters for each operation type
REQUIRED_PARAMS = {
    OperationType.ASPIRATE: ["volume", "location"],
    OperationType.DISPENSE: ["volume", "location"],
    OperationType.TRANSFER: ["volume", "source", "destination"],
    OperationType.PICK_UP_TIP: [],
    OperationType.DROP_TIP: [],
    OperationType.MOVE_TO: ["location"],
    OperationType.MIX: ["repetitions", "volume", "location"],
    OperationType.BLOWOUT: [],
    OperationType.LOAD_LABWARE: ["labware_type", "slot"],
    OperationType.LOAD_PIPETTE: ["pipette_type", "mount"],
    OperationType.PAUSE: [],
    OperationType.WAIT: ["seconds"],
    OperationType.SET_TEMPERATURE: ["temperature"],
}


# Python code templates for each operation
# Use .get() with defaults to handle missing parameters gracefully
def _format_location(p):
    """Format location parameter for code generation."""
    loc = p.get('location', '')
    if loc:
        return f"plate['{loc}']"  # Default to 'plate' variable
    return ''

OPERATION_TEMPLATES = {
    OperationType.PICK_UP_TIP: lambda p: f"pipette.pick_up_tip({p.get('location', '')})",

    OperationType.DROP_TIP: lambda p: f"pipette.drop_tip({p.get('location', '')})",

    OperationType.ASPIRATE: lambda p: f"pipette.aspirate({p.get('volume', 'VOLUME')}, plate['{p.get('location', 'A1')}'])",

    OperationType.DISPENSE: lambda p: f"pipette.dispense({p.get('volume', 'VOLUME')}, plate['{p.get('location', 'A1')}'])",

    OperationType.TRANSFER: lambda p: f"pipette.transfer({p.get('volume', 'VOLUME')}, plate['{p.get('source', 'A1')}'], plate['{p.get('destination', 'A2')}'], new_tip='{p.get('new_tip', 'always')}')",

    OperationType.MIX: lambda p: f"pipette.mix({p.get('repetitions', 3)}, {p.get('volume', 100)}, plate['{p.get('location', 'A1')}'])",

    OperationType.BLOWOUT: lambda p: f"pipette.blow_out({p.get('location', '')})",

    OperationType.MOVE_TO: lambda p: f"pipette.move_to(plate['{p.get('location', 'A1')}'])",

    OperationType.HOME: lambda p: "protocol.home()",

    OperationType.PAUSE: lambda p: f"protocol.pause('{p.get('message', 'Paused')}')",

    OperationType.WAIT: lambda p: f"protocol.delay(seconds={p.get('seconds', 1)})",

    OperationType.COMMENT: lambda p: f"protocol.comment('{p.get('message', '')}')",

    OperationType.LOAD_LABWARE: lambda p: f"{p.get('name', 'labware')} = protocol.load_labware('{p.get('labware_type', 'LABWARE')}', {p.get('slot', 1)})",

    OperationType.LOAD_PIPETTE: lambda p: f"pipette = protocol.load_instrument('{p.get('pipette_type', 'PIPETTE')}', '{p.get('mount', 'left')}')",

    OperationType.SET_TEMPERATURE: lambda p: f"temp_module.set_temperature({p.get('temperature', 25)})",
}


class OperationMapper:
    """Maps parsed intents to OT-2 operations."""

    # Keyword mappings for operation detection (English and Chinese)
    KEYWORDS = {
        OperationType.ASPIRATE: {
            "en": ["aspirate", "suck", "draw", "pull", "uptake", "take up"],
            "zh": ["吸取", "吸", "抽取", "抽", "吸入"],
        },
        OperationType.DISPENSE: {
            "en": ["dispense", "release", "deposit", "put", "add", "deliver"],
            "zh": ["分配", "释放", "注入", "加入", "放入", "滴入"],
        },
        OperationType.TRANSFER: {
            "en": ["transfer", "move liquid", "pipette from", "take from"],
            "zh": ["转移", "移液", "从...到", "移动液体"],
        },
        OperationType.PICK_UP_TIP: {
            "en": ["pick up tip", "get tip", "grab tip", "take tip"],
            "zh": ["取枪头", "拿枪头", "抓取枪头", "获取枪头"],
        },
        OperationType.DROP_TIP: {
            "en": ["drop tip", "eject tip", "discard tip", "throw tip"],
            "zh": ["丢弃枪头", "弃枪头", "扔掉枪头", "退枪头"],
        },
        OperationType.MIX: {
            "en": ["mix", "stir", "blend", "homogenize"],
            "zh": ["混合", "混匀", "搅拌", "吹打混匀"],
        },
        OperationType.MOVE_TO: {
            "en": ["move to", "go to", "position at", "navigate to"],
            "zh": ["移动到", "前往", "定位到", "移至"],
        },
        OperationType.PAUSE: {
            "en": ["pause", "stop", "wait for user", "hold"],
            "zh": ["暂停", "停止", "等待用户", "中断"],
        },
        OperationType.WAIT: {
            "en": ["wait", "delay", "sleep", "hold for"],
            "zh": ["等待", "延迟", "等", "停留"],
        },
        OperationType.BLOWOUT: {
            "en": ["blowout", "blow out", "expel", "push out"],
            "zh": ["吹出", "排空", "吹干"],
        },
        OperationType.HOME: {
            "en": ["home", "reset position", "return home"],
            "zh": ["归位", "复位", "回原点"],
        },
    }

    # Common labware mappings
    LABWARE_ALIASES = {
        # 96-well plates
        "96孔板": "corning_96_wellplate_360ul_flat",
        "96-well plate": "corning_96_wellplate_360ul_flat",
        "96 well": "corning_96_wellplate_360ul_flat",

        # 24-well plates
        "24孔板": "corning_24_wellplate_3.4ml_flat",
        "24-well plate": "corning_24_wellplate_3.4ml_flat",

        # Tip racks
        "枪头盒": "opentrons_96_tiprack_300ul",
        "tip rack": "opentrons_96_tiprack_300ul",
        "300ul tips": "opentrons_96_tiprack_300ul",
        "1000ul tips": "opentrons_96_tiprack_1000ul",
        "20ul tips": "opentrons_96_tiprack_20ul",

        # Reservoirs
        "试剂槽": "nest_12_reservoir_15ml",
        "reservoir": "nest_12_reservoir_15ml",

        # Tube racks
        "离心管架": "opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap",
        "tube rack": "opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap",
    }

    # Pipette mappings
    PIPETTE_ALIASES = {
        "单道移液器": "p300_single_gen2",
        "8道移液器": "p300_multi_gen2",
        "p300": "p300_single_gen2",
        "p1000": "p1000_single_gen2",
        "p20": "p20_single_gen2",
        "single channel": "p300_single_gen2",
        "multi channel": "p300_multi_gen2",
    }

    def __init__(self):
        self._build_keyword_index()

    def _build_keyword_index(self):
        """Build inverted index for fast keyword lookup."""
        self._keyword_index = {}
        for op_type, lang_keywords in self.KEYWORDS.items():
            for lang, keywords in lang_keywords.items():
                for keyword in keywords:
                    self._keyword_index[keyword.lower()] = op_type

    def detect_operation_type(self, text: str) -> Optional[OperationType]:
        """Detect operation type from text."""
        text_lower = text.lower()
        for keyword, op_type in self._keyword_index.items():
            if keyword in text_lower:
                return op_type
        return None

    def resolve_labware(self, alias: str) -> str:
        """Resolve labware alias to Opentrons labware name."""
        alias_lower = alias.lower().strip()
        return self.LABWARE_ALIASES.get(alias_lower, alias)

    def resolve_pipette(self, alias: str) -> str:
        """Resolve pipette alias to Opentrons pipette name."""
        alias_lower = alias.lower().strip()
        return self.PIPETTE_ALIASES.get(alias_lower, alias)

    def create_operation(
        self,
        op_type: OperationType,
        params: Dict[str, Any],
        description: str = ""
    ) -> Operation:
        """Create an Operation object."""
        return Operation(
            type=op_type,
            params=params,
            description=description
        )

"""
Liquid Handler Operations

Defines all operations supported by liquid handling robots.
"""

from enum import Enum
from typing import Dict, List
from ...core.plugin_base import OperationDef


class LiquidOperation(Enum):
    """Liquid handling operation types."""
    ASPIRATE = "aspirate"
    DISPENSE = "dispense"
    TRANSFER = "transfer"
    DISTRIBUTE = "distribute"
    CONSOLIDATE = "consolidate"
    MIX = "mix"
    PICK_UP_TIP = "pick_up_tip"
    DROP_TIP = "drop_tip"
    TOUCH_TIP = "touch_tip"
    BLOW_OUT = "blow_out"
    AIR_GAP = "air_gap"
    MOVE_TO = "move_to"
    HOME = "home"
    PAUSE = "pause"
    WAIT = "wait"


# Operation definitions with multilingual keywords
LIQUID_OPERATIONS: Dict[LiquidOperation, OperationDef] = {
    LiquidOperation.ASPIRATE: OperationDef(
        name="aspirate",
        action="liquid_handler.aspirate",
        keywords={
            "en": ["aspirate", "suck", "draw", "uptake", "pull", "take up"],
            "zh": ["吸取", "吸", "抽取", "吸液", "抽"],
        },
        params_schema={
            "volume": {"type": "float", "unit": "ul", "required": True},
            "location": {"type": "well", "required": False},
            "rate": {"type": "float", "unit": "ul/s", "required": False},
        },
        description="Aspirate liquid from a well",
    ),

    LiquidOperation.DISPENSE: OperationDef(
        name="dispense",
        action="liquid_handler.dispense",
        keywords={
            "en": ["dispense", "release", "add", "eject", "put", "deposit"],
            "zh": ["分配", "释放", "注入", "加", "添加", "放"],
        },
        params_schema={
            "volume": {"type": "float", "unit": "ul", "required": True},
            "location": {"type": "well", "required": False},
            "rate": {"type": "float", "unit": "ul/s", "required": False},
        },
        description="Dispense liquid to a well",
    ),

    LiquidOperation.TRANSFER: OperationDef(
        name="transfer",
        action="liquid_handler.transfer",
        keywords={
            "en": ["transfer", "move liquid", "move", "pipette"],
            "zh": ["转移", "移液", "转", "移动"],
        },
        params_schema={
            "volume": {"type": "float", "unit": "ul", "required": True},
            "source": {"type": "well", "required": True},
            "destination": {"type": "well", "required": True},
            "mix_before": {"type": "int", "required": False},
            "mix_after": {"type": "int", "required": False},
        },
        description="Transfer liquid from source to destination",
    ),

    LiquidOperation.DISTRIBUTE: OperationDef(
        name="distribute",
        action="liquid_handler.distribute",
        keywords={
            "en": ["distribute", "spread", "disperse"],
            "zh": ["分发", "分布", "散布"],
        },
        params_schema={
            "volume": {"type": "float", "unit": "ul", "required": True},
            "source": {"type": "well", "required": True},
            "destinations": {"type": "well_list", "required": True},
        },
        description="Distribute from one source to multiple destinations",
    ),

    LiquidOperation.CONSOLIDATE: OperationDef(
        name="consolidate",
        action="liquid_handler.consolidate",
        keywords={
            "en": ["consolidate", "collect", "gather", "pool"],
            "zh": ["汇集", "收集", "聚集", "合并"],
        },
        params_schema={
            "volume": {"type": "float", "unit": "ul", "required": True},
            "sources": {"type": "well_list", "required": True},
            "destination": {"type": "well", "required": True},
        },
        description="Consolidate from multiple sources to one destination",
    ),

    LiquidOperation.MIX: OperationDef(
        name="mix",
        action="liquid_handler.mix",
        keywords={
            "en": ["mix", "stir", "blend", "homogenize", "vortex"],
            "zh": ["混合", "混匀", "搅拌", "混", "涡旋"],
        },
        params_schema={
            "repetitions": {"type": "int", "required": False, "default": 3},
            "volume": {"type": "float", "unit": "ul", "required": False},
            "location": {"type": "well", "required": False},
            "rate": {"type": "float", "unit": "ul/s", "required": False},
        },
        description="Mix liquid by aspirating and dispensing repeatedly",
    ),

    LiquidOperation.PICK_UP_TIP: OperationDef(
        name="pick_up_tip",
        action="liquid_handler.pick_up_tip",
        keywords={
            "en": ["pick up tip", "get tip", "grab tip", "take tip", "pick tip"],
            "zh": ["取枪头", "拿枪头", "取吸头", "取尖"],
        },
        params_schema={
            "location": {"type": "well", "required": False},
            "tip_rack": {"type": "string", "required": False},
        },
        description="Pick up a pipette tip",
    ),

    LiquidOperation.DROP_TIP: OperationDef(
        name="drop_tip",
        action="liquid_handler.drop_tip",
        keywords={
            "en": ["drop tip", "eject tip", "discard tip", "remove tip", "trash tip"],
            "zh": ["丢弃枪头", "弃枪头", "扔枪头", "丢枪头", "弃尖"],
        },
        params_schema={
            "location": {"type": "well", "required": False},
        },
        description="Drop/discard the current tip",
    ),

    LiquidOperation.TOUCH_TIP: OperationDef(
        name="touch_tip",
        action="liquid_handler.touch_tip",
        keywords={
            "en": ["touch tip", "touch wall", "remove droplet"],
            "zh": ["触壁", "点触", "去除液滴"],
        },
        params_schema={
            "location": {"type": "well", "required": False},
            "radius": {"type": "float", "required": False},
            "speed": {"type": "float", "required": False},
        },
        description="Touch the tip to the well wall to remove droplets",
    ),

    LiquidOperation.BLOW_OUT: OperationDef(
        name="blow_out",
        action="liquid_handler.blow_out",
        keywords={
            "en": ["blow out", "blow", "expel", "push out"],
            "zh": ["吹出", "吹", "排出"],
        },
        params_schema={
            "location": {"type": "well", "required": False},
        },
        description="Blow out any remaining liquid",
    ),

    LiquidOperation.AIR_GAP: OperationDef(
        name="air_gap",
        action="liquid_handler.air_gap",
        keywords={
            "en": ["air gap", "air cushion"],
            "zh": ["气隙", "空气垫"],
        },
        params_schema={
            "volume": {"type": "float", "unit": "ul", "required": True},
        },
        description="Add an air gap to prevent dripping",
    ),

    LiquidOperation.MOVE_TO: OperationDef(
        name="move_to",
        action="liquid_handler.move_to",
        keywords={
            "en": ["move to", "go to", "position at"],
            "zh": ["移动到", "移至", "定位"],
        },
        params_schema={
            "location": {"type": "well", "required": True},
            "offset_x": {"type": "float", "required": False},
            "offset_y": {"type": "float", "required": False},
            "offset_z": {"type": "float", "required": False},
            "speed": {"type": "float", "required": False},
        },
        description="Move pipette to a specific location",
    ),

    LiquidOperation.HOME: OperationDef(
        name="home",
        action="liquid_handler.home",
        keywords={
            "en": ["home", "reset", "return home"],
            "zh": ["归位", "复位", "回原点"],
        },
        params_schema={},
        description="Home the robot to its starting position",
    ),

    LiquidOperation.PAUSE: OperationDef(
        name="pause",
        action="liquid_handler.pause",
        keywords={
            "en": ["pause", "stop", "halt"],
            "zh": ["暂停", "停止", "中断"],
        },
        params_schema={
            "message": {"type": "string", "required": False},
        },
        description="Pause protocol execution",
    ),

    LiquidOperation.WAIT: OperationDef(
        name="wait",
        action="wait",
        keywords={
            "en": ["wait", "delay", "sleep", "hold"],
            "zh": ["等待", "延迟", "延时", "等"],
        },
        params_schema={
            "duration_seconds": {"type": "float", "required": True},
        },
        description="Wait for a specified duration",
    ),
}


def get_operation(op_type: LiquidOperation) -> OperationDef:
    """Get operation definition by type."""
    return LIQUID_OPERATIONS[op_type]


def get_all_operations() -> Dict[LiquidOperation, OperationDef]:
    """Get all operation definitions."""
    return LIQUID_OPERATIONS.copy()

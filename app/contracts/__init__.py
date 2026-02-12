"""Contract layer: typed Pydantic models defining data flow between layers.

L3 -> L2: TaskContract      (Intake/Clarifier output)
L2 -> L1: CampaignPlan      (Campaign Planner output)
L1 -> L0: RunBundle          (Protocol Compiler + Deck/Layout output)
L0 out:   ResultPacket       (Data/Feature agent output)
"""
from __future__ import annotations

from app.contracts.task_contract import (
    DimensionDef,
    ExplorationSpace,
    HumanGatePolicy,
    ObjectiveSpec,
    SafetyEnvelope,
    StopCondition,
    TaskContract,
    new_task_contract_id,
)
from app.contracts.campaign_plan import (
    CampaignPlan,
    ResourceRequirements,
    RoundSpec,
    new_campaign_plan_id,
)
from app.contracts.run_bundle import (
    DeckLayout,
    RunBundle,
    SlotAssignment,
    new_run_bundle_id,
)
from app.contracts.query_contract import (
    ColumnSpec,
    QueryConstraints,
    QueryPlan,
    QueryRequest,
    QueryResult,
    new_query_plan_id,
)
from app.contracts.result_packet import (
    QualityLabel,
    ResultPacket,
    new_result_packet_id,
)

__all__ = [
    # L3 -> L2
    "DimensionDef",
    "ExplorationSpace",
    "HumanGatePolicy",
    "ObjectiveSpec",
    "SafetyEnvelope",
    "StopCondition",
    "TaskContract",
    "new_task_contract_id",
    # L2 -> L1
    "CampaignPlan",
    "ResourceRequirements",
    "RoundSpec",
    "new_campaign_plan_id",
    # L1 -> L0
    "DeckLayout",
    "RunBundle",
    "SlotAssignment",
    "new_run_bundle_id",
    # L0 output
    "QualityLabel",
    "ResultPacket",
    "new_result_packet_id",
    # Query agent
    "ColumnSpec",
    "QueryConstraints",
    "QueryPlan",
    "QueryRequest",
    "QueryResult",
    "new_query_plan_id",
]

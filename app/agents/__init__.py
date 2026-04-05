"""OTbot Agent system -- typed agents with unified interface.

Individual agents follow the BaseAgent protocol. For paper-aligned
grouping, use the four specialist swarms via SwarmFactory.
"""
from app.agents.base import AgentResult, BaseAgent
from app.agents.blueprint_reader_agent import BlueprintReaderAgent, BlueprintReaderInput, BlueprintReaderOutput
from app.agents.cleaning_agent import CleaningAgent, CleaningInput, CleaningOutput
from app.agents.code_writer_agent import CodeWriterAgent, CodeWriterInput, CodeWriterOutput
from app.agents.compiler_agent import CompilerAgent, CompileInput, CompileOutput
from app.agents.deck_layout_agent import DeckLayoutAgent, DeckLayoutInput, DeckLayoutOutput
from app.agents.design_agent import DesignAgent, DesignInput, DesignOutput
from app.agents.nlp_code_agent import NLPCodeAgent, NLPCodeInput, NLPCodeOutput
from app.agents.onboarding_agent import OnboardingAgent, OnboardingInput, OnboardingOutput
from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput, OrchestratorOutput
from app.agents.planner_agent import PlannerAgent, PlannerInput, PlannerOutput, PlannedRound
from app.agents.recovery_agent import RecoveryAgent, RecoveryInput, RecoveryOutput
from app.agents.safety_agent import SafetyAgent, SafetyCheckInput, SafetyCheckOutput
from app.agents.sensing_agent import SensingAgent, SensingInput, SensingOutput, QCCheck, QCResult
from app.agents.simulation_agent import SimulationAgent, SimulationInput, SimulationOutput
from app.agents.monitor_agent import MonitorAgent, MonitorInput, MonitorOutput
from app.agents.analyzer_agent import AnalyzerAgent, AnalyzerInput, AnalyzerOutput
from app.agents.query_agent import QueryAgent, QueryRequest, QueryResult
from app.agents.stop_agent import StopAgent, StopInput, StopOutput
from app.agents.swarm import (
    AnalystSwarm,
    BaseSwarm,
    EngineerSwarm,
    ScientistSwarm,
    SwarmContext,
    SwarmFactory,
    SwarmResult,
    ValidatorSwarm,
    list_swarms,
)
from app.agents.tool_holder_dialog_agent import ToolHolderDialogAgent, ToolHolderDialogInput, ToolHolderDialogOutput
from app.agents.capability_agent import (
    CapabilityAgent,
    CapabilityQueryInput,
    CapabilitySnapshot,
    PipetteInfo,
    SlotInfo,
)
from app.agents.execution_agent import (
    ExecutionAgent,
    ExecutionInput,
    ExecutionOutput,
)
from app.agents.validation_agent import (
    ValidationAgent,
    ValidationInput,
    ValidationOutput,
)
from app.agents.observation_agent import (
    ObservationAgent,
    ObservationInput,
    ObservationPacket,
)
from app.agents.optimization_agent import (
    OptimizationAgent,
    OptimizationInput,
    OptimizationOutput,
    CandidatePoint,
)

__all__ = [
    # Base
    "AgentResult",
    "BaseAgent",
    # Individual agents
    "BlueprintReaderAgent", "BlueprintReaderInput", "BlueprintReaderOutput",
    "CleaningAgent", "CleaningInput", "CleaningOutput",
    "CodeWriterAgent", "CodeWriterInput", "CodeWriterOutput",
    "CompilerAgent", "CompileInput", "CompileOutput",
    "DeckLayoutAgent", "DeckLayoutInput", "DeckLayoutOutput",
    "DesignAgent", "DesignInput", "DesignOutput",
    "NLPCodeAgent", "NLPCodeInput", "NLPCodeOutput",
    "OnboardingAgent", "OnboardingInput", "OnboardingOutput",
    "OrchestratorAgent", "OrchestratorInput", "OrchestratorOutput",
    "PlannerAgent", "PlannerInput", "PlannerOutput", "PlannedRound",
    "RecoveryAgent", "RecoveryInput", "RecoveryOutput",
    "SafetyAgent", "SafetyCheckInput", "SafetyCheckOutput",
    "SensingAgent", "SensingInput", "SensingOutput", "QCCheck", "QCResult",
    "SimulationAgent", "SimulationInput", "SimulationOutput",
    "MonitorAgent", "MonitorInput", "MonitorOutput",
    "AnalyzerAgent", "AnalyzerInput", "AnalyzerOutput",
    "QueryAgent", "QueryRequest", "QueryResult",
    "StopAgent", "StopInput", "StopOutput",
    "ToolHolderDialogAgent", "ToolHolderDialogInput", "ToolHolderDialogOutput",
    # Capability agent (new agentic architecture)
    "CapabilityAgent", "CapabilityQueryInput", "CapabilitySnapshot",
    "PipetteInfo", "SlotInfo",
    # Phase 2-5 agents
    "ExecutionAgent", "ExecutionInput", "ExecutionOutput",
    "ValidationAgent", "ValidationInput", "ValidationOutput",
    "ObservationAgent", "ObservationInput", "ObservationPacket",
    "OptimizationAgent", "OptimizationInput", "OptimizationOutput", "CandidatePoint",
    # Swarm system (paper-aligned 4 specialist groups)
    "BaseSwarm",
    "SwarmContext",
    "SwarmFactory",
    "SwarmResult",
    "ScientistSwarm",
    "EngineerSwarm",
    "AnalystSwarm",
    "ValidatorSwarm",
    "list_swarms",
]

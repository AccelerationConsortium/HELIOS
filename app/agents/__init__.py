"""OTbot Agent system -- typed agents with unified interface.

Individual agents follow the BaseAgent protocol. For paper-aligned
grouping, use the four specialist swarms via SwarmFactory.
"""
from app.agents.base import AgentResult, BaseAgent
from app.agents.code_writer_agent import CodeWriterAgent, CodeWriterInput, CodeWriterOutput
from app.agents.compiler_agent import CompilerAgent, CompileInput, CompileOutput
from app.agents.design_agent import DesignAgent, DesignInput, DesignOutput
from app.agents.onboarding_agent import OnboardingAgent, OnboardingInput, OnboardingOutput
from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput, OrchestratorOutput
from app.agents.planner_agent import PlannerAgent, PlannerInput, PlannerOutput, PlannedRound
from app.agents.recovery_agent import RecoveryAgent, RecoveryInput, RecoveryOutput
from app.agents.safety_agent import SafetyAgent, SafetyCheckInput, SafetyCheckOutput
from app.agents.sensing_agent import SensingAgent, SensingInput, SensingOutput, QCCheck, QCResult
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

__all__ = [
    # Base
    "AgentResult",
    "BaseAgent",
    # Individual agents
    "CodeWriterAgent", "CodeWriterInput", "CodeWriterOutput",
    "CompilerAgent", "CompileInput", "CompileOutput",
    "DesignAgent", "DesignInput", "DesignOutput",
    "OnboardingAgent", "OnboardingInput", "OnboardingOutput",
    "OrchestratorAgent", "OrchestratorInput", "OrchestratorOutput",
    "PlannerAgent", "PlannerInput", "PlannerOutput", "PlannedRound",
    "RecoveryAgent", "RecoveryInput", "RecoveryOutput",
    "SafetyAgent", "SafetyCheckInput", "SafetyCheckOutput",
    "SensingAgent", "SensingInput", "SensingOutput", "QCCheck", "QCResult",
    "QueryAgent", "QueryRequest", "QueryResult",
    "StopAgent", "StopInput", "StopOutput",
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

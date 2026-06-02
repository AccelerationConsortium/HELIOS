"""MCP-style manifest for HELIOS capabilities.

This is not a network MCP server. It is a stable manifest that maps HELIOS
primitives and agent capabilities into MCP-shaped tools/resources so a future
MCP transport or agent SDK can consume the same contract without scraping
Python classes.
"""
from __future__ import annotations

from typing import Any

from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.compiler_agent import CompilerAgent
from app.agents.design_agent import DesignAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.sensing_agent import SensingAgent
from app.agents.validation_agent import ValidationAgent
from app.services.primitives_registry import PrimitiveSpec, get_registry


def build_mcp_manifest() -> dict[str, Any]:
    registry = get_registry()
    primitive_tools = [_primitive_to_tool(p) for p in registry.list_primitives()]
    agent_resources = [_agent_to_resource(agent) for agent in _agent_instances()]
    return {
        "protocol": "mcp-style-manifest",
        "version": "0.1",
        "tools": primitive_tools,
        "resources": agent_resources,
        "prompts": [
            {
                "name": "campaign_context",
                "description": (
                    "Injects campaign memory, peer knowledge, blackboard entries, "
                    "and safety state into an agent call."
                ),
            }
        ],
    }


def _primitive_to_tool(primitive: PrimitiveSpec) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in primitive.params:
        properties[param.name] = {
            "type": _json_schema_type(param.type),
            "description": param.description,
        }
        if param.default is not None:
            properties[param.name]["default"] = param.default
        if not param.optional:
            required.append(param.name)

    return {
        "name": primitive.name,
        "description": primitive.description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
        "annotations": {
            "instrument": primitive.instrument,
            "resource_id": primitive.resource_id,
            "safety_class": primitive.safety_class.name,
            "error_class": primitive.error_class,
        },
    }


def _agent_to_resource(agent: Any) -> dict[str, Any]:
    return {
        "uri": f"helios://agents/{agent.name}",
        "name": agent.name,
        "description": agent.description,
        "mimeType": "application/json",
        "metadata": {
            "layer": agent.layer,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "min_confidence": getattr(cap, "min_confidence", 0.0),
                }
                for cap in getattr(agent, "capabilities", [])
            ],
        },
    }


def _agent_instances() -> list[Any]:
    return [
        PlannerAgent(),
        DesignAgent(),
        CompilerAgent(),
        SensingAgent(),
        AnalyzerAgent(),
        ValidationAgent(),
    ]


def _json_schema_type(raw_type: str) -> str:
    normalized = raw_type.lower()
    if normalized in {"str", "string"}:
        return "string"
    if normalized in {"int", "integer"}:
        return "integer"
    if normalized in {"float", "number", "double"}:
        return "number"
    if normalized in {"bool", "boolean"}:
        return "boolean"
    if normalized in {"list", "array"}:
        return "array"
    if normalized in {"dict", "object"}:
        return "object"
    return "string"

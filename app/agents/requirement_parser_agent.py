"""Requirement Parser Agent -- structured NL experiment parsing.

This agent is the regularization layer before planning/compilation. It turns
free-form experiment text into a typed Pydantic contract with explicit facts,
assumptions, missing fields, and evidence spans.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent

_OT2_AGENT_DIR = str(Path(__file__).resolve().parents[2] / "ot2-nlp-agent")


ParseStatus = Literal["parsed", "needs_clarification", "unsupported", "unsafe"]
EvidenceSource = Literal["explicit", "inferred", "context"]


class ParsedField(BaseModel):
    """One extracted field with provenance."""

    name: str
    value: Any
    source: EvidenceSource = "explicit"
    evidence: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ParsedStep(BaseModel):
    """One parsed operation candidate."""

    operation: str
    params: dict[str, Any] = Field(default_factory=dict)
    evidence: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: EvidenceSource = "explicit"
    missing_fields: list[str] = Field(default_factory=list)


class RequirementParseInput(BaseModel):
    """Input for requirement parsing."""

    intent: str = Field(..., description="Raw natural-language experiment request")
    context: dict[str, Any] = Field(default_factory=dict)


class RequirementParseOutput(BaseModel):
    """Typed parse result consumed by downstream agents."""

    status: ParseStatus
    goal: str = ""
    domain: str = "general"
    language: str = "en"
    fields: list[ParsedField] = Field(default_factory=list)
    steps: list[ParsedStep] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    unsupported_terms: list[str] = Field(default_factory=list)
    prompt_contract_xml: str = ""
    original_text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RequirementParserAgent(
    BaseAgent[RequirementParseInput, RequirementParseOutput]
):
    """Parse NL experiment requirements into a closed, typed contract."""

    name = "requirement_parser"
    description = "NL experiment requirement -> Pydantic parse contract"
    layer = "L0"

    _UNSAFE_PATTERNS = (
        ("explosive", "Potential explosive hazard mentioned."),
        ("爆炸", "Potential explosive hazard mentioned."),
        ("toxic gas", "Potential toxic gas hazard mentioned."),
        ("有毒气体", "Potential toxic gas hazard mentioned."),
    )

    _FIELD_QUESTIONS = {
        "operation": (
            "Which operation should the robot perform?",
            "需要机器人执行哪种操作？",
        ),
        "volume": (
            "What volume should be used?",
            "需要使用多少体积？",
        ),
        "location": (
            "Which well or location should be used?",
            "需要使用哪个孔位或位置？",
        ),
        "source": (
            "Which source well or location should be used?",
            "源孔位或源位置是哪一个？",
        ),
        "destination": (
            "Which destination well or location should be used?",
            "目标孔位或目标位置是哪一个？",
        ),
        "repetitions": (
            "How many repetitions should be used?",
            "需要重复多少次？",
        ),
        "seconds": (
            "How long should the wait be?",
            "需要等待多久？",
        ),
        "temperature": (
            "What target temperature should be used?",
            "目标温度是多少？",
        ),
    }

    def validate_input(self, input_data: RequirementParseInput) -> list[str]:
        if not input_data.intent.strip():
            return ["intent must be a non-empty string"]
        return []

    async def process(
        self,
        input_data: RequirementParseInput,
    ) -> RequirementParseOutput:
        if _OT2_AGENT_DIR not in sys.path:
            sys.path.insert(0, _OT2_AGENT_DIR)

        from ot2_agent.operations import REQUIRED_PARAMS
        from ot2_agent.parser import NLParser
        from ot2_agent.planner import IntentParser

        text = input_data.intent.strip()
        context = input_data.context or {}
        prompt_contract_xml = self._build_prompt_contract_xml(context)

        intent_parser = IntentParser()
        high_level_intent = intent_parser.parse(text)

        parser = NLParser()
        parsed_intents = parser.parse_multi_step(text)

        fields = self._fields_from_intent(high_level_intent)
        steps: list[ParsedStep] = []
        missing_fields: list[str] = []

        for parsed in parsed_intents:
            if parsed.operation_type is None:
                if (
                    not parsed.params
                    and high_level_intent.domain != "general"
                    and high_level_intent.confidence >= 0.5
                ):
                    continue
                missing_fields.append("operation")
                steps.append(
                    ParsedStep(
                        operation="unknown",
                        params=parsed.params,
                        evidence=parsed.original_text,
                        confidence=parsed.confidence,
                        missing_fields=["operation"],
                    )
                )
                continue

            operation = parsed.operation_type.value
            required = REQUIRED_PARAMS.get(parsed.operation_type, [])
            step_missing = [
                param for param in required if param not in parsed.params
            ]
            missing_fields.extend(step_missing)

            steps.append(
                ParsedStep(
                    operation=operation,
                    params=parsed.params,
                    evidence=parsed.original_text,
                    confidence=parsed.confidence,
                    missing_fields=step_missing,
                )
            )

        safety_flags = [
            message
            for pattern, message in self._UNSAFE_PATTERNS
            if re.search(pattern, text, re.IGNORECASE)
        ]
        unsupported_terms = self._unsupported_terms(context, steps)
        missing_fields = sorted(set(missing_fields))

        status: ParseStatus = "parsed"
        if safety_flags:
            status = "unsafe"
        elif unsupported_terms:
            status = "unsupported"
        elif missing_fields or (
            not steps and high_level_intent.domain == "general"
        ):
            status = "needs_clarification"

        questions = [
            self._question_for(field, high_level_intent.language)
            for field in missing_fields
        ]

        assumptions = self._assumptions(context, steps)

        return RequirementParseOutput(
            status=status,
            goal=high_level_intent.goal,
            domain=high_level_intent.domain,
            language=high_level_intent.language,
            fields=fields,
            steps=steps,
            missing_fields=missing_fields,
            clarifying_questions=questions,
            assumptions=assumptions,
            safety_flags=safety_flags,
            unsupported_terms=unsupported_terms,
            prompt_contract_xml=prompt_contract_xml,
            original_text=text,
            confidence=high_level_intent.confidence,
        )

    @staticmethod
    def _fields_from_intent(intent: Any) -> list[ParsedField]:
        fields: list[ParsedField] = [
            ParsedField(
                name="goal",
                value=intent.goal,
                source="inferred",
                evidence=intent.original_text,
                confidence=intent.confidence,
            ),
            ParsedField(
                name="domain",
                value=intent.domain,
                source="inferred",
                evidence=intent.original_text,
                confidence=intent.confidence,
            ),
        ]
        for metric in intent.target_metrics:
            fields.append(
                ParsedField(
                    name="target_metric",
                    value=metric,
                    evidence=metric,
                    confidence=0.8,
                )
            )
        for name, value in intent.known_conditions.items():
            fields.append(
                ParsedField(
                    name=name,
                    value=value,
                    evidence=str(value),
                    confidence=0.75,
                )
            )
        return fields

    @classmethod
    def _question_for(cls, field: str, language: str) -> str:
        en, zh = cls._FIELD_QUESTIONS.get(
            field,
            (f"Please provide {field}.", f"请提供 {field}。"),
        )
        return zh if language == "zh" else en

    @staticmethod
    def _unsupported_terms(
        context: dict[str, Any],
        steps: list[ParsedStep],
    ) -> list[str]:
        allowed_operations = set(context.get("allowed_operations") or [])
        if not allowed_operations:
            return []
        return sorted(
            {
                step.operation
                for step in steps
                if step.operation != "unknown"
                and step.operation not in allowed_operations
            }
        )

    @staticmethod
    def _assumptions(
        context: dict[str, Any],
        steps: list[ParsedStep],
    ) -> list[str]:
        assumptions: list[str] = []
        if steps and not context.get("devices"):
            assumptions.append("No device context provided; downstream planner must choose compatible devices.")
        return assumptions

    @staticmethod
    def _build_prompt_contract_xml(context: dict[str, Any]) -> str:
        allowed_operations = ", ".join(context.get("allowed_operations") or [])
        devices = ", ".join(context.get("devices") or [])
        return (
            "<requirement_parser_contract>"
            "<role>Parse experiment requirements into typed fields only.</role>"
            "<rules>"
            "<rule>Do not invent instruments, materials, units, wells, or durations.</rule>"
            "<rule>Return needs_clarification when required fields are absent.</rule>"
            "<rule>Separate explicit evidence from assumptions.</rule>"
            "</rules>"
            f"<available_devices>{devices}</available_devices>"
            f"<allowed_operations>{allowed_operations}</allowed_operations>"
            "<output>Pydantic RequirementParseOutput</output>"
            "</requirement_parser_contract>"
        )

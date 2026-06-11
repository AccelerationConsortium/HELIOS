from __future__ import annotations


async def test_requirement_parser_returns_pydantic_contract_for_complete_step():
    from app.agents.requirement_parser_agent import (
        RequirementParseInput,
        RequirementParserAgent,
    )

    result = await RequirementParserAgent().run(
        RequirementParseInput(intent="从A1吸取100微升")
    )

    assert result.success
    assert result.output is not None
    assert result.output.status == "parsed"
    assert result.output.steps[0].operation == "aspirate"
    assert result.output.steps[0].params["volume"] == 100
    assert result.output.steps[0].params["location"] == "A1"
    assert result.output.steps[0].evidence == "从A1吸取100微升"
    assert "Pydantic RequirementParseOutput" in result.output.prompt_contract_xml


async def test_requirement_parser_asks_for_missing_required_fields():
    from app.agents.requirement_parser_agent import (
        RequirementParseInput,
        RequirementParserAgent,
    )

    result = await RequirementParserAgent().run(
        RequirementParseInput(intent="aspirate 50ul")
    )

    assert result.success
    assert result.output is not None
    assert result.output.status == "needs_clarification"
    assert result.output.missing_fields == ["location"]
    assert result.output.clarifying_questions == [
        "Which well or location should be used?"
    ]


async def test_requirement_parser_allows_high_level_experiment_goals():
    from app.agents.requirement_parser_agent import (
        RequirementParseInput,
        RequirementParserAgent,
    )

    result = await RequirementParserAgent().run(
        RequirementParseInput(intent="我想做OER测量，用NiFe催化剂测试过电位")
    )

    assert result.success
    assert result.output is not None
    assert result.output.status == "parsed"
    assert result.output.domain == "electrochemistry"
    assert any(
        field.name == "target_metric" and field.value == "overpotential"
        for field in result.output.fields
    )


async def test_nlp_code_agent_blocks_generation_when_parser_needs_clarification():
    from app.agents.nlp_code_agent import NLPCodeAgent, NLPCodeInput

    result = await NLPCodeAgent().run(
        NLPCodeInput(intent="aspirate 50ul", auto_approve=True)
    )

    assert result.success
    assert result.output is not None
    assert result.output.status == "needs_clarification"
    assert result.output.parsed_requirements["status"] == "needs_clarification"
    assert result.output.clarifying_questions == [
        "Which well or location should be used?"
    ]

from __future__ import annotations


def test_inverse_design_nexus_advisor_is_opt_in(monkeypatch, request):
    from app.agents.inverse_design_agent import (
        CandidateSystem,
        InverseDesignAgent,
        InverseDesignInput,
        InverseDesignOutput,
    )
    from app.core.config import get_settings

    monkeypatch.delenv("NEXUS_ADVISOR_ENABLED", raising=False)
    get_settings.cache_clear()
    request.addfinalizer(get_settings.cache_clear)

    result = InverseDesignOutput(
        candidate_systems=[
            CandidateSystem(
                system_name="NiFeOOH",
                elements=["Ni", "Fe"],
                rationale="test",
                literature_refs=[],
                predicted_performance={"eta10_mv": 100.0},
                confidence=0.5,
                recommended_precursors=[],
            )
        ],
        recommended_stock_solutions=[],
        suggested_dimensions=[],
        suggested_protocol_template={},
        search_summary="base",
    )
    input_data = InverseDesignInput(
        objective="OER catalyst",
        target_metrics={"eta10_mv": {"direction": "minimize", "target": 80}},
    )

    out = InverseDesignAgent()._enhance_with_nexus(result, input_data)

    assert out is result

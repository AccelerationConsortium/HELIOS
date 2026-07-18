from __future__ import annotations

import math
from pathlib import Path


def test_electrochem_surrogate_is_deterministic_and_structured():
    from app.services.electrochem_surrogate import (
        ElectrochemSurrogateConfig,
        evaluate_electrochem_candidate,
    )

    near_optimal = {
        "catalyst_conc_mm": 50.0,
        "deposition_time_s": 90.0,
        "electrolyte_ph": 13.0,
        "scan_rate_mv_s": 20.0,
        "anneal_temp_c": 60.0,
    }
    poor = {
        "catalyst_conc_mm": 90.0,
        "deposition_time_s": 150.0,
        "electrolyte_ph": 11.0,
        "scan_rate_mv_s": 80.0,
        "anneal_temp_c": 95.0,
    }
    cfg = ElectrochemSurrogateConfig(seed=123, noise_std_mv=0.0)

    first = evaluate_electrochem_candidate(near_optimal, config=cfg)
    second = evaluate_electrochem_candidate(near_optimal, config=cfg)
    bad = evaluate_electrochem_candidate(poor, config=cfg)

    assert first.overpotential_mv == second.overpotential_mv
    assert first.overpotential_mv < bad.overpotential_mv
    assert len(first.lsv_curve["potential_v"]) == len(first.lsv_curve["current_ma"])
    assert first.eis_summary["r_charge_transfer_ohm"] > 0


def test_trace_completeness_detects_missing_required_events():
    from benchmarks.e2e_study import assess_trace_completeness

    trace = assess_trace_completeness([])

    assert not trace.complete
    assert trace.score == 0.0
    assert "agent_stage_graph" in trace.missing
    assert "agent_result:planner" in trace.missing


def test_study_spec_rejects_unsupported_baselines():
    import pytest

    from benchmarks.e2e_study import StudySpec

    with pytest.raises(ValueError, match="unsupported baseline"):
        StudySpec(baselines=("no_safety",))  # type: ignore[arg-type]


async def test_e2e_study_runs_real_orchestrator_path(monkeypatch, tmp_path):
    from app.core.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("SCIENTIFIC_LEDGER_ROOT", str(tmp_path / "scientific-ledger"))
    monkeypatch.setenv("SCIENTIFIC_LEDGER_ENABLED", "true")
    monkeypatch.setenv("SCIENTIFIC_LEDGER_GIT_ENABLED", "false")
    get_settings.cache_clear()

    from benchmarks.e2e_study import StudySpec, run_study

    aggregate = await run_study(
        StudySpec(
            seeds=(0,),
            baselines=("helios_full", "random"),
            max_rounds=1,
            batch_size=1,
            noise_std_mv=0.0,
        )
    )
    summary = aggregate.by_baseline()
    helios = next(result for result in aggregate.results if result.baseline == "helios_full")
    random = next(result for result in aggregate.results if result.baseline == "random")

    assert helios.trace.complete
    assert helios.trace.score == 1.0
    assert helios.metadata["n_evaluations"] == 1
    assert helios.safety_violations == 0
    assert not math.isnan(helios.final_best)
    assert random.trace.score == 0.0
    assert summary["helios_full"]["trace_completeness_rate"] == 1.0
    assert summary["random"]["n"] == 1

    # The real orchestrator path must close its live Pending Decision Card,
    # persist typed accounting, and make the trajectory exportable for RLVR.
    from app.services.scientific_ledger import get_scientific_ledger

    campaign_id = helios.metadata["campaign_id"]
    ledger = get_scientific_ledger()
    campaign_dir = Path(ledger.campaign_directory(campaign_id))
    card = campaign_dir / "rounds/001/decision_001.md"
    assert card.is_file()
    card_text = card.read_text(encoding="utf-8")
    assert "status: completed" in card_text
    assert "selected_backend: adaptive" in card_text
    assert "## Reward and Verification" in card_text
    assert len(ledger.export_rlvr_records(campaign_id)) == 1

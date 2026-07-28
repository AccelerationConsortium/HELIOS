from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tests.fixtures.scientific_ledger import decision_accounting


@pytest.fixture
def ledger_env(monkeypatch, request, tmp_path):
    from app.core.config import get_settings
    from app.core.db import init_db
    from app.services.decision_trajectory import persist_campaign_trajectory
    from app.services.scientific_ledger import get_scientific_ledger

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("SCIENTIFIC_LEDGER_ROOT", str(tmp_path / "ledger"))
    get_settings.cache_clear()
    request.addfinalizer(get_settings.cache_clear)
    init_db()
    accounting = decision_accounting()
    persist_campaign_trajectory(accounting)
    ledger = get_scientific_ledger()
    ledger.record_completed(
        accounting,
        failures=[{"root_cause": "pipette offset", "result": "aspiration failed"}],
    )
    return ledger


async def test_scientific_search_endpoint_returns_markdown_hit(ledger_env):
    from app.api.v1.endpoints.memory import scientific_memory_search

    response = await scientific_memory_search("pipette offset", None, 10)
    assert response["count"] >= 1
    assert any(hit["path"].endswith("failure.md") for hit in response["hits"])


def test_scientific_routes_are_mounted_on_v1_api(ledger_env):
    from app.main import app

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/memory/scientific/search",
            params={"q": "pipette offset", "campaign_id": "campaign-32"},
        )

    assert response.status_code == 200
    assert response.json()["count"] >= 1


async def test_scientific_artifact_endpoint_reads_markdown(ledger_env):
    from app.api.v1.endpoints.memory import scientific_memory_artifact

    response = await scientific_memory_artifact("campaign-32", "rounds/003/decision_003.md")
    assert response.media_type == "text/markdown"
    assert b"# Decision Card 003" in response.body


async def test_scientific_artifact_endpoint_blocks_traversal(ledger_env, tmp_path):
    from app.api.v1.endpoints.memory import scientific_memory_artifact

    (tmp_path / "secret.md").write_text("secret")
    with pytest.raises(HTTPException) as exc_info:
        await scientific_memory_artifact("campaign-32", "../../secret.md")
    assert exc_info.value.status_code == 400


async def test_scientific_rlvr_endpoint_returns_jsonl(ledger_env):
    from app.api.v1.endpoints.memory import scientific_memory_rlvr

    response = await scientific_memory_rlvr("campaign-32")
    assert response.media_type == "application/x-ndjson"
    assert b'"schema_version": "helios.rlvr/v1"' in response.body


def test_fixture_wrote_only_expected_root(ledger_env):
    assert Path(ledger_env.root).name == "ledger"

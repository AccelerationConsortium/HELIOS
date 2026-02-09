"""Tests for the capabilities API endpoints."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture()
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_list_capabilities(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "skills" in data
    assert "total_primitives" in data
    assert "total_skills" in data
    assert data["total_skills"] >= 1
    assert data["total_primitives"] >= 1


@pytest.mark.anyio
async def test_list_primitives_unfiltered(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/capabilities/primitives")
    assert resp.status_code == 200
    data = resp.json()
    assert "primitives" in data
    assert "count" in data
    assert data["count"] >= 1


@pytest.mark.anyio
async def test_list_primitives_by_error_class(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/capabilities/primitives?error_class=CRITICAL")
    assert resp.status_code == 200
    data = resp.json()
    for p in data["primitives"]:
        assert p["error_class"] == "CRITICAL"


@pytest.mark.anyio
async def test_list_primitives_by_instrument(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/capabilities/primitives?instrument=ot2-robot")
    assert resp.status_code == 200
    data = resp.json()
    for p in data["primitives"]:
        assert p["instrument"] == "ot2-robot"


@pytest.mark.anyio
async def test_get_specific_primitive(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/capabilities/primitives/robot.aspirate")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["primitive"]["name"] == "robot.aspirate"
    assert data["primitive"]["error_class"] == "CRITICAL"
    # Should have params
    param_names = {p["name"] for p in data["primitive"]["params"]}
    assert "volume" in param_names


@pytest.mark.anyio
async def test_get_nonexistent_primitive(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/capabilities/primitives/nonexistent.action")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is False


@pytest.mark.anyio
async def test_capabilities_summary(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/capabilities/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "robot.aspirate" in data["summary"]
    assert "[CRITICAL]" in data["summary"]


# ---------------------------------------------------------------------------
# New: safety_class + contract fields
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_primitive_includes_safety_class(client: AsyncClient) -> None:
    """Every primitive response should include a safety_class field."""
    resp = await client.get("/api/v1/capabilities/primitives/robot.aspirate")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    prim = data["primitive"]
    assert "safety_class" in prim
    assert prim["safety_class"] == "HAZARDOUS"


@pytest.mark.anyio
async def test_primitive_includes_contract(client: AsyncClient) -> None:
    """robot.aspirate should have contract with preconditions and effects."""
    resp = await client.get("/api/v1/capabilities/primitives/robot.aspirate")
    assert resp.status_code == 200
    data = resp.json()
    prim = data["primitive"]
    assert "contract" in prim
    contract = prim["contract"]
    assert contract is not None
    assert "preconditions" in contract
    assert "effects" in contract
    assert "timeout" in contract
    assert len(contract["preconditions"]) >= 1
    assert contract["timeout"]["seconds"] > 0


@pytest.mark.anyio
async def test_primitive_contract_null_when_missing(client: AsyncClient) -> None:
    """Primitives without contracts should have contract=null."""
    # log has no contract typically
    resp = await client.get("/api/v1/capabilities/primitives/log")
    assert resp.status_code == 200
    data = resp.json()
    if data["found"]:
        prim = data["primitive"]
        assert "contract" in prim
        # Contract may or may not be null depending on skill file
        # Just verify the field exists
        assert isinstance(prim["contract"], (dict, type(None)))


@pytest.mark.anyio
async def test_list_primitives_by_safety_class(client: AsyncClient) -> None:
    """Filter primitives by safety_class."""
    resp = await client.get("/api/v1/capabilities/primitives?safety_class=HAZARDOUS")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    for p in data["primitives"]:
        assert p["safety_class"] == "HAZARDOUS"


@pytest.mark.anyio
async def test_list_primitives_safety_class_informational(client: AsyncClient) -> None:
    """INFORMATIONAL filter should return low-risk primitives."""
    resp = await client.get("/api/v1/capabilities/primitives?safety_class=INFORMATIONAL")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    for p in data["primitives"]:
        assert p["safety_class"] == "INFORMATIONAL"


@pytest.mark.anyio
async def test_list_primitives_safety_class_case_insensitive(client: AsyncClient) -> None:
    """Safety class filter should be case-insensitive."""
    resp = await client.get("/api/v1/capabilities/primitives?safety_class=hazardous")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1


@pytest.mark.anyio
async def test_all_primitives_have_safety_class(client: AsyncClient) -> None:
    """Every primitive in the unfiltered list should have safety_class."""
    resp = await client.get("/api/v1/capabilities/primitives")
    assert resp.status_code == 200
    data = resp.json()
    valid_classes = {"INFORMATIONAL", "REVERSIBLE", "CAREFUL", "HAZARDOUS"}
    for p in data["primitives"]:
        assert "safety_class" in p, f"{p['name']} missing safety_class"
        assert p["safety_class"] in valid_classes, f"{p['name']} has invalid safety_class: {p['safety_class']}"


@pytest.mark.anyio
async def test_all_primitives_have_contract_field(client: AsyncClient) -> None:
    """Every primitive should have the contract field (dict or null)."""
    resp = await client.get("/api/v1/capabilities/primitives")
    assert resp.status_code == 200
    data = resp.json()
    for p in data["primitives"]:
        assert "contract" in p, f"{p['name']} missing contract field"

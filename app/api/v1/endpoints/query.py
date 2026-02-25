"""Query endpoint — structured lab data retrieval via DSL or natural language.

POST /api/v1/query
  Body: QueryRequest  (either `prompt` for NL path or `dsl_query` for DSL path)
  Returns: QueryResult

POST /api/v1/query/dsl
  Body: ExperimentQuery  (shorthand — wraps query in a QueryRequest)
  Returns: QueryResult

GET /api/v1/query/entities
  Returns available entity types and their whitelisted columns.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.query_agent import QueryAgent
from app.contracts.query_contract import QueryConstraints, QueryRequest, QueryResult
from app.contracts.query_dsl import ENTITY_SOURCE, ExperimentQuery, _VALID_COLUMNS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

# Module-level agent instance (shared, stateless)
_query_agent = QueryAgent()


# ---------------------------------------------------------------------------
# POST /query  — unified entry point
# ---------------------------------------------------------------------------


@router.post("", response_model=QueryResult)
async def run_query(payload: QueryRequest) -> QueryResult:
    """Execute a lab data query via NL prompt or typed DSL.

    Set ``dsl_query`` for deterministic structured queries (no LLM).
    Set ``prompt`` for free-form natural-language queries (uses LLM).
    """
    try:
        return await _query_agent.run(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Query execution error")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /query/dsl  — DSL shorthand (no need to wrap in QueryRequest)
# ---------------------------------------------------------------------------


class DslQueryRequest(BaseModel):
    """Shorthand request body for DSL-only queries."""

    query: ExperimentQuery
    constraints: QueryConstraints = QueryConstraints()
    campaign_id: str | None = None
    run_id: str | None = None
    snapshot_mode: bool = False
    snapshot_name: str | None = None


@router.post("/dsl", response_model=QueryResult)
async def run_dsl_query(payload: DslQueryRequest) -> QueryResult:
    """Execute a typed DSL query (deterministic, no LLM)."""
    request = QueryRequest(
        dsl_query=payload.query,
        constraints=payload.constraints,
        campaign_id=payload.campaign_id,
        run_id=payload.run_id,
        snapshot_mode=payload.snapshot_mode,
        snapshot_name=payload.snapshot_name,
    )
    try:
        return await _query_agent.run(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("DSL query execution error")
        raise HTTPException(status_code=500, detail=f"DSL query failed: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /query/entities  — schema introspection for frontend Builder
# ---------------------------------------------------------------------------


class EntityInfo(BaseModel):
    entity: str
    source: str
    columns: list[str]


class EntitiesResponse(BaseModel):
    entities: list[EntityInfo]


@router.get("/entities", response_model=EntitiesResponse)
async def list_entities() -> EntitiesResponse:
    """Return available entity types and their whitelisted columns.

    Used by the frontend Query Builder to populate dropdowns.
    """
    entities = [
        EntityInfo(
            entity=entity,
            source=source,
            columns=sorted(_VALID_COLUMNS.get(entity, frozenset())),
        )
        for entity, source in ENTITY_SOURCE.items()
    ]
    return EntitiesResponse(entities=entities)

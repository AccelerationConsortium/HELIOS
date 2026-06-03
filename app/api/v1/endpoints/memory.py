"""Memory snapshot endpoint — exposes the three-layer memory store to the UI.

The lab UI's "Memory" tab consumes this endpoint to show what HELIOS has
learned from past campaigns:
  - **Episodic**: most recent per-step outcomes (run X used primitive Y → outcome)
  - **Semantic**: aggregated parameter statistics (mean, stddev, success rate)
  - **Procedural**: repair recipes (when primitive X fails with pattern Y → steps)

Read-only. Writes happen via the `MemoryListener` on the event bus.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Query

from app.core.db import parse_json, row_to_dict, run_txn

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/snapshot")
async def memory_snapshot(
    episodes_limit: int = Query(20, ge=1, le=200),
) -> Dict[str, Any]:
    """Return a single snapshot of the three memory layers.

    Designed for the Lab UI's Memory tab — one round trip, ready to render.
    """

    def _load(conn):
        episodes_raw = conn.execute(
            "SELECT id, run_id, step_key, primitive, params_json, outcome, error, created_at "
            "FROM memory_episodes ORDER BY created_at DESC LIMIT ?",
            (episodes_limit,),
        ).fetchall()
        priors_raw = conn.execute(
            "SELECT primitive, param_name, mean, stddev, sample_count, "
            "success_rate, success_count, total_count, updated_at "
            "FROM memory_semantic ORDER BY sample_count DESC LIMIT 100"
        ).fetchall()
        recipes_raw = conn.execute(
            "SELECT id, trigger_primitive, trigger_error_pattern, recipe_json, "
            "source, hit_count, updated_at "
            "FROM memory_procedures ORDER BY hit_count DESC, updated_at DESC LIMIT 50"
        ).fetchall()
        return episodes_raw, priors_raw, recipes_raw

    try:
        episodes_raw, priors_raw, recipes_raw = run_txn(_load)
    except Exception:
        episodes_raw, priors_raw, recipes_raw = [], [], []

    episodes: List[Dict[str, Any]] = []
    for r in episodes_raw:
        episodes.append({
            "id": r["id"],
            "run_id": r["run_id"],
            "step_key": r["step_key"],
            "primitive": r["primitive"],
            "params": parse_json(r["params_json"], {}),
            "outcome": r["outcome"],
            "error": r["error"],
            "created_at": r["created_at"],
        })

    priors: List[Dict[str, Any]] = []
    for r in priors_raw:
        priors.append({
            "primitive": r["primitive"],
            "param_name": r["param_name"],
            "mean": r["mean"],
            "stddev": r["stddev"],
            "sample_count": r["sample_count"],
            "success_rate": r["success_rate"],
            "success_count": r["success_count"],
            "total_count": r["total_count"],
            "updated_at": r["updated_at"],
        })

    recipes: List[Dict[str, Any]] = []
    for r in recipes_raw:
        recipes.append({
            "id": r["id"],
            "trigger_primitive": r["trigger_primitive"],
            "trigger_error_pattern": r["trigger_error_pattern"],
            "recipe": parse_json(r["recipe_json"], []),
            "source": r["source"],
            "hit_count": r["hit_count"],
            "updated_at": r["updated_at"],
        })

    return {
        "episodes": episodes,
        "priors": priors,
        "recipes": recipes,
        "episodes_count": len(episodes),
        "priors_count": len(priors),
        "recipes_count": len(recipes),
    }


@router.get("/recall")
async def memory_recall(
    primitives: str = Query("", description="Comma-separated list of primitives to recall priors for"),
    limit: int = Query(5, ge=1, le=50),
) -> Dict[str, Any]:
    """Targeted memory recall for a planned experiment.

    Returns the priors + top recipes relevant to the given primitives.
    Used by the UI's "Memory Recall" banner shown when a new campaign starts.
    """
    primitive_list = [p.strip() for p in primitives.split(",") if p.strip()]

    priors: List[Dict[str, Any]] = []
    recipes: List[Dict[str, Any]] = []

    if primitive_list:
        placeholders = ",".join("?" for _ in primitive_list)

        def _load_priors(conn):
            return conn.execute(
                f"SELECT primitive, param_name, mean, stddev, sample_count, "
                f"success_rate, success_count, total_count, updated_at "
                f"FROM memory_semantic WHERE primitive IN ({placeholders}) "
                f"ORDER BY sample_count DESC LIMIT ?",
                (*primitive_list, limit),
            ).fetchall()

        def _load_recipes(conn):
            return conn.execute(
                f"SELECT id, trigger_primitive, trigger_error_pattern, recipe_json, "
                f"source, hit_count, updated_at "
                f"FROM memory_procedures WHERE trigger_primitive IN ({placeholders}) "
                f"ORDER BY hit_count DESC LIMIT ?",
                (*primitive_list, limit),
            ).fetchall()

        try:
            for r in run_txn(_load_priors):
                priors.append({
                    "primitive": r["primitive"],
                    "param_name": r["param_name"],
                    "mean": r["mean"],
                    "stddev": r["stddev"],
                    "sample_count": r["sample_count"],
                    "success_rate": r["success_rate"],
                    "success_count": r["success_count"],
                    "total_count": r["total_count"],
                })
        except Exception:
            pass

        try:
            for r in run_txn(_load_recipes):
                recipes.append({
                    "id": r["id"],
                    "trigger_primitive": r["trigger_primitive"],
                    "trigger_error_pattern": r["trigger_error_pattern"],
                    "recipe": parse_json(r["recipe_json"], []),
                    "source": r["source"],
                    "hit_count": r["hit_count"],
                })
        except Exception:
            pass

    summary_parts: List[str] = []
    if priors:
        top = priors[0]
        summary_parts.append(
            f"{top['primitive']}.{top['param_name']} prior: "
            f"mean={top['mean']:.3f}, sigma={top['stddev']:.3f}, n={top['sample_count']}"
        )
    if recipes:
        summary_parts.append(
            f"{len(recipes)} known repair recipe(s) for the requested primitives"
        )
    if not summary_parts:
        summary_parts.append("No prior experience for the requested primitives - first run")

    return {
        "primitives": primitive_list,
        "priors": priors,
        "recipes": recipes,
        "summary": " · ".join(summary_parts),
    }

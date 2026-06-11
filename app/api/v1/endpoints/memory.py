"""Memory snapshot endpoint — exposes the three-layer memory store to the UI.

The lab UI's "Memory" tab consumes this endpoint to show what HELIOS has
learned from past campaigns:
  - **Episodic**: most recent per-step outcomes (run X used primitive Y → outcome)
  - **Semantic**: aggregated parameter statistics (mean, stddev, success rate)
  - **Procedural**: repair recipes (when primitive X fails with pattern Y → steps)

Read-only. Writes happen via the `MemoryListener` on the event bus.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.core.db import parse_json, run_txn

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/snapshot")
async def memory_snapshot(
    episodes_limit: int = Query(20, ge=1, le=200),
) -> dict[str, Any]:
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

    episodes: list[dict[str, Any]] = []
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

    priors: list[dict[str, Any]] = []
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

    recipes: list[dict[str, Any]] = []
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
) -> dict[str, Any]:
    """Targeted memory recall for a planned experiment.

    Returns the priors + top recipes relevant to the given primitives.
    Used by the UI's "Memory Recall" banner shown when a new campaign starts.
    """
    primitive_list = [p.strip() for p in primitives.split(",") if p.strip()]

    priors: list[dict[str, Any]] = []
    recipes: list[dict[str, Any]] = []

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

    summary_parts: list[str] = []
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


@router.get("/graph")
async def memory_graph() -> dict[str, Any]:
    """Knowledge graph of past campaigns, primitives, failures, and recipes.

    Returns a node/edge list suitable for a D3 force-directed graph. The
    graph shows what HELIOS has learned: which campaigns used which
    primitives, which primitives fail with which error patterns, and
    which recovery recipes exist for those patterns.

    Node types:
      - ``campaign``     — one per run id (or "demo_seed" synthetic for the seed rows)
      - ``primitive``    — a hardware primitive (e.g. robot.dispense)
      - ``error_pattern``— a substring seen in a failure error (e.g. "tip")
      - ``recipe``       — a repair recipe in memory_procedures

    Edge labels:
      - "used"          — campaign → primitive
      - "fails_with"    — primitive → error_pattern
      - "recovered_by"  — error_pattern → recipe
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def node(node_id: str, ntype: str, label: str, **extra) -> None:
        # First-write-wins; merges extra fields into existing record
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "type": ntype, "label": label}
        for k, v in extra.items():
            nodes[node_id].setdefault(k, v)

    def edge(src: str, tgt: str, rel: str, weight: int = 1) -> None:
        # Coalesce duplicate edges; bump weight on repeat
        for e in edges:
            if e["source"] == src and e["target"] == tgt and e["relation"] == rel:
                e["weight"] += 1
                return
        edges.append({"source": src, "target": tgt, "relation": rel, "weight": weight})

    # ---- Episodes → campaign/primitive nodes + "used" edges ----
    def _load_episodes(conn):
        return conn.execute(
            "SELECT run_id, primitive, outcome, error FROM memory_episodes "
            "ORDER BY created_at DESC LIMIT 500"
        ).fetchall()

    try:
        episode_rows = run_txn(_load_episodes)
    except Exception:
        episode_rows = []

    # Track per-(primitive, error_pattern) failure counts so we can emit
    # primitive → error_pattern edges with sensible weights.
    fail_counts: dict[tuple[str, str], int] = {}

    for r in episode_rows:
        run_id    = r["run_id"] or "unknown"
        primitive = r["primitive"] or "unknown"
        outcome   = r["outcome"] or "succeeded"
        error     = r["error"] or ""

        node(f"campaign:{run_id}", "campaign", run_id[-8:])
        node(f"primitive:{primitive}", "primitive", primitive)
        edge(f"campaign:{run_id}", f"primitive:{primitive}", "used", 1)

        if outcome == "failed" and error:
            # Bucket the error into a coarse pattern (lowercase, stripped
            # of digits and punctuation so "Tip not attached" and
            # "tip 2" collapse to the same "tip" pattern).
            import re as _re
            pat = _re.sub(r"[^a-z ]+", "", error.lower()).strip()[:40] or "unknown"
            pat = pat or "unknown"
            node(f"err:{primitive}:{pat}", "error_pattern", f"{pat}", parent_primitive=primitive)
            edge(f"primitive:{primitive}", f"err:{primitive}:{pat}", "fails_with", 1)
            fail_counts[(primitive, pat)] = fail_counts.get((primitive, pat), 0) + 1

    # ---- Recipes → error_pattern/recipe nodes + "recovered_by" edges ----
    def _load_recipes(conn):
        return conn.execute(
            "SELECT id, trigger_primitive, trigger_error_pattern, source, hit_count "
            "FROM memory_procedures ORDER BY hit_count DESC"
        ).fetchall()

    try:
        recipe_rows = run_txn(_load_recipes)
    except Exception:
        recipe_rows = []

    for r in recipe_rows:
        prim = r["trigger_primitive"] or "unknown"
        pat  = r["trigger_error_pattern"] or "unknown"
        rid  = r["id"]
        node(f"recipe:{rid}", "recipe", f"{prim} ↳ {pat}", source=r["source"] or "—")
        # Connect to its triggering error pattern if we have one
        err_id = f"err:{prim}:{pat}"
        if err_id in nodes:
            edge(err_id, f"recipe:{rid}", "recovered_by", max(1, r["hit_count"] or 1))
        else:
            # Recipe exists but no failure was observed yet; still surface
            # the recipe, attached to the primitive.
            node(err_id, "error_pattern", pat, parent_primitive=prim)
            edge(f"primitive:{prim}", err_id, "fails_with", 1)
            edge(err_id, f"recipe:{rid}", "recovered_by", max(1, r["hit_count"] or 1))

    # ---- Annotate primitives with run counts from episodes ----
    prim_runs: dict[str, int] = {}
    prim_success: dict[str, int] = {}
    for r in episode_rows:
        p = r["primitive"] or "unknown"
        prim_runs[p] = prim_runs.get(p, 0) + 1
        if r["outcome"] == "succeeded":
            prim_success[p] = prim_success.get(p, 0) + 1
    for prim, total in prim_runs.items():
        ok = prim_success.get(prim, 0)
        rate = ok / total if total else 0.0
        nodes[f"primitive:{prim}"]["run_count"] = total
        nodes[f"primitive:{prim}"]["success_count"] = ok
        nodes[f"primitive:{prim}"]["success_rate"] = rate

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }

"""QueryPlan cache: deterministic caching for NL → SQL compilation products.

Cache key = SHA-256(normalised_prompt + schema_version + agent_version).
Same prompt + same schema → same QueryPlan → skip LLM recompilation.

Storage: query_plan_cache table in SQLite.
Eviction: LRU by last_used, max entries configurable.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata

from app.contracts.query_contract import QueryPlan
from app.core.db import connection, json_dumps, parse_json, run_txn, utcnow_iso


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CACHE_ENTRIES = 500


# ---------------------------------------------------------------------------
# Prompt normalisation
# ---------------------------------------------------------------------------

def _normalize_prompt(prompt: str) -> str:
    """Normalise prompt for deterministic hashing.

    - Lowercase
    - Unicode NFKC normalisation
    - Collapse whitespace
    - Strip leading/trailing whitespace
    """
    text = unicodedata.normalize("NFKC", prompt)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Cache key computation
# ---------------------------------------------------------------------------

def compute_cache_key(
    prompt: str,
    schema_version: str,
    agent_version: str = "1.0",
) -> str:
    """Compute deterministic cache key from prompt + schema + version.

    Returns a 64-char hex SHA-256 digest.
    """
    normalised = _normalize_prompt(prompt)
    raw = f"{normalised}|{schema_version}|{agent_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Cache CRUD
# ---------------------------------------------------------------------------

def get_cached_plan(cache_key: str) -> QueryPlan | None:
    """Retrieve a cached QueryPlan, or None if not found.

    Also bumps hit_count and updates last_used.
    """
    now = utcnow_iso()

    def _txn(conn):
        row = conn.execute(
            "SELECT plan_json FROM query_plan_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        # Bump hit_count
        conn.execute(
            "UPDATE query_plan_cache SET hit_count = hit_count + 1, "
            "last_used = ? WHERE cache_key = ?",
            (now, cache_key),
        )
        return parse_json(row["plan_json"], None)

    data = run_txn(_txn)
    if data is None:
        return None
    return QueryPlan(**data)


def store_plan(cache_key: str, plan: QueryPlan) -> None:
    """Store a QueryPlan in cache. Performs LRU eviction if over limit."""
    now = utcnow_iso()
    plan_json = json_dumps(plan.model_dump())

    def _txn(conn):
        conn.execute(
            "INSERT OR REPLACE INTO query_plan_cache "
            "(cache_key, plan_json, hit_count, created_at, last_used) "
            "VALUES (?, ?, 1, ?, ?)",
            (cache_key, plan_json, now, now),
        )
        # LRU eviction: keep only MAX_CACHE_ENTRIES most recent
        conn.execute(
            "DELETE FROM query_plan_cache WHERE cache_key NOT IN ("
            "  SELECT cache_key FROM query_plan_cache "
            "  ORDER BY last_used DESC LIMIT ?"
            ")",
            (MAX_CACHE_ENTRIES,),
        )

    run_txn(_txn)


def invalidate_by_schema(schema_version: str) -> int:
    """Delete cache entries that don't match the current schema version.

    Returns the number of deleted entries.
    """
    def _txn(conn):
        # Find entries whose plan_json contains a different schema_version
        rows = conn.execute(
            "SELECT cache_key, plan_json FROM query_plan_cache"
        ).fetchall()
        stale_keys: list[str] = []
        for row in rows:
            plan_data = parse_json(row["plan_json"], {})
            if plan_data.get("schema_version") != schema_version:
                stale_keys.append(row["cache_key"])
        if stale_keys:
            placeholders = ",".join("?" * len(stale_keys))
            conn.execute(
                f"DELETE FROM query_plan_cache WHERE cache_key IN ({placeholders})",
                stale_keys,
            )
        return len(stale_keys)

    return run_txn(_txn)


def clear_cache() -> int:
    """Delete all cache entries. Returns count of deleted entries."""
    def _txn(conn):
        cursor = conn.execute("DELETE FROM query_plan_cache")
        return cursor.rowcount
    return run_txn(_txn)


def cache_stats() -> dict:
    """Return cache statistics."""
    with connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as total, "
            "COALESCE(SUM(hit_count), 0) as total_hits "
            "FROM query_plan_cache"
        ).fetchone()
    return {
        "total_entries": row["total"],
        "total_hits": row["total_hits"],
        "max_entries": MAX_CACHE_ENTRIES,
    }

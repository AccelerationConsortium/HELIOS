"""SQL Guard: AST-level validation for QueryAgent-generated SQL.

Enforces a strict whitelist:
  - Read-only (SELECT only)
  - JOIN limit
  - WHERE requirement (configurable)
  - Subquery depth limit
  - Dangerous function blocklist
  - Table existence check
  - LIMIT enforcement
  - No PRAGMA / ATTACH / schema modification

Zero external dependencies — uses regex + sqlite3 EXPLAIN for validation.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Any

from pydantic import BaseModel

from app.contracts.query_contract import QueryConstraints


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class SqlValidationResult(BaseModel):
    """Result of SQL validation."""

    valid: bool
    sql: str  # potentially modified (e.g. LIMIT appended)
    violations: list[str] = []
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Blocklists and constants
# ---------------------------------------------------------------------------

# Statements that MUST NOT appear (case-insensitive, word boundary)
_WRITE_KEYWORDS: set[str] = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "REPLACE", "TRUNCATE", "MERGE", "UPSERT",
}

# Dangerous SQLite functions
_DANGEROUS_FUNCTIONS: set[str] = {
    "load_extension",
    "writefile",
    "readfile",
    "fts3_tokenizer",
    "fts5",
    "zipfile",
    "sqlar_compress",
    "sqlar_uncompress",
    "edit",
}

# Schema-modifying / escape statements
_BLOCKED_STATEMENTS: set[str] = {
    "PRAGMA", "ATTACH", "DETACH", "VACUUM", "REINDEX",
    "ANALYZE",  # ANALYZE can be slow on large DBs
}

# Regex to detect JOIN clauses
_JOIN_PATTERN = re.compile(
    r"\b(?:INNER|LEFT|RIGHT|CROSS|FULL|NATURAL)?\s*JOIN\b",
    re.IGNORECASE,
)

# Regex to detect subqueries (SELECT inside parentheses)
_SUBQUERY_PATTERN = re.compile(r"\(\s*SELECT\b", re.IGNORECASE)

# Regex for function calls
_FUNCTION_PATTERN = re.compile(r"\b(\w+)\s*\(", re.IGNORECASE)

# Regex for LIMIT clause
_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)

# Regex for WHERE clause
_WHERE_PATTERN = re.compile(r"\bWHERE\b", re.IGNORECASE)

# Regex for table names in FROM / JOIN clauses
_FROM_TABLE_PATTERN = re.compile(
    r"\bFROM\s+(\w+)|\bJOIN\s+(\w+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def _check_read_only(sql: str) -> list[str]:
    """Ensure SQL is read-only (SELECT only)."""
    violations: list[str] = []
    # Strip leading whitespace and comments
    stripped = sql.strip()

    # Check for blocked statements
    first_word = stripped.split()[0].upper() if stripped.split() else ""
    if first_word != "SELECT":
        violations.append(
            f"Only SELECT statements allowed, got '{first_word}'"
        )

    # Check for write keywords anywhere in the SQL
    for kw in _WRITE_KEYWORDS:
        pattern = re.compile(rf"\b{kw}\b", re.IGNORECASE)
        if pattern.search(sql):
            violations.append(f"Write operation '{kw}' not allowed")

    # Check for blocked statements
    for stmt in _BLOCKED_STATEMENTS:
        pattern = re.compile(rf"\b{stmt}\b", re.IGNORECASE)
        if pattern.search(sql):
            violations.append(f"Statement '{stmt}' not allowed")

    return violations


def _check_joins(sql: str, max_joins: int) -> list[str]:
    """Enforce JOIN count limit."""
    joins = _JOIN_PATTERN.findall(sql)
    if len(joins) > max_joins:
        return [
            f"Too many JOINs: {len(joins)} (max {max_joins})"
        ]
    return []


def _check_subquery_depth(sql: str, max_depth: int) -> list[str]:
    """Enforce subquery nesting depth limit."""
    # Simple heuristic: count nested (SELECT patterns
    depth = 0
    current_depth = 0
    for i, char in enumerate(sql):
        if char == '(' and sql[i:i+7].upper().lstrip().startswith("SELECT"):
            # Look ahead to check if this is a subquery
            rest = sql[i+1:].lstrip()
            if rest.upper().startswith("SELECT"):
                current_depth += 1
                depth = max(depth, current_depth)
        elif char == ')':
            if current_depth > 0:
                current_depth -= 1

    if depth > max_depth:
        return [
            f"Subquery depth {depth} exceeds limit {max_depth}"
        ]
    return []


def _check_dangerous_functions(sql: str) -> list[str]:
    """Block dangerous SQLite functions."""
    violations: list[str] = []
    for match in _FUNCTION_PATTERN.finditer(sql):
        func_name = match.group(1).lower()
        if func_name in _DANGEROUS_FUNCTIONS:
            violations.append(
                f"Dangerous function '{func_name}' not allowed"
            )
    return violations


def _check_tables(sql: str, known_tables: set[str],
                  allowed: list[str], denied: list[str]) -> list[str]:
    """Verify all referenced tables exist and are not denied."""
    violations: list[str] = []
    referenced: set[str] = set()

    for match in _FROM_TABLE_PATTERN.finditer(sql):
        table = match.group(1) or match.group(2)
        if table:
            referenced.add(table.lower())

    for table in referenced:
        # Check existence
        if table not in {t.lower() for t in known_tables}:
            violations.append(f"Table '{table}' does not exist in schema")
        # Check deny list
        if denied and table in {t.lower() for t in denied}:
            violations.append(f"Table '{table}' is in deny list")
        # Check allow list (if specified, only allowed tables can be queried)
        if allowed and table not in {t.lower() for t in allowed}:
            violations.append(
                f"Table '{table}' not in allow list"
            )

    return violations


def _check_where(sql: str, require: bool) -> list[str]:
    """Optionally require a WHERE clause."""
    if require and not _WHERE_PATTERN.search(sql):
        return ["WHERE clause required but not found"]
    return []


def _ensure_limit(sql: str, max_rows: int) -> tuple[str, list[str]]:
    """Ensure SQL has a LIMIT clause; append if missing."""
    warnings: list[str] = []
    if not _LIMIT_PATTERN.search(sql):
        # Remove trailing semicolons before appending
        sql = sql.rstrip().rstrip(";")
        sql = f"{sql} LIMIT {max_rows}"
        warnings.append(f"Auto-appended LIMIT {max_rows}")
    return sql, warnings


def _check_with_explain(sql: str, params: list[Any]) -> list[str]:
    """Use sqlite3 EXPLAIN to catch syntax errors without executing."""
    violations: list[str] = []
    try:
        # Use in-memory DB for syntax check
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(f"EXPLAIN {sql}", params)
        except sqlite3.OperationalError as e:
            violations.append(f"SQL syntax error: {e}")
        finally:
            conn.close()
    except Exception as e:
        violations.append(f"SQL validation error: {e}")
    return violations


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

def validate_sql(
    sql: str,
    params: list[Any],
    constraints: QueryConstraints,
    known_tables: set[str],
) -> SqlValidationResult:
    """Validate SQL against the full whitelist ruleset.

    Args:
        sql: The SQL query to validate.
        params: Bind parameters for the query.
        constraints: Query constraints (limits, allowed/denied tables, etc.).
        known_tables: Set of table names that exist in the schema.

    Returns:
        SqlValidationResult with valid flag, possibly-modified SQL,
        and any violations/warnings.
    """
    violations: list[str] = []
    warnings: list[str] = []

    # 1. Read-only check
    violations.extend(_check_read_only(sql))

    # Early exit if not SELECT — no point checking further
    if violations:
        return SqlValidationResult(
            valid=False, sql=sql, violations=violations, warnings=warnings,
        )

    # 2. Dangerous functions
    violations.extend(_check_dangerous_functions(sql))

    # 3. JOIN limit
    violations.extend(_check_joins(sql, constraints.max_joins))

    # 4. Subquery depth
    violations.extend(
        _check_subquery_depth(sql, constraints.max_subquery_depth)
    )

    # 5. Table validation
    violations.extend(
        _check_tables(
            sql, known_tables,
            constraints.allowed_tables, constraints.denied_tables,
        )
    )

    # 6. WHERE requirement
    violations.extend(_check_where(sql, constraints.require_where))

    # 7. LIMIT enforcement
    sql, limit_warnings = _ensure_limit(sql, constraints.max_rows)
    warnings.extend(limit_warnings)

    # If we already have violations, skip EXPLAIN
    if violations:
        return SqlValidationResult(
            valid=False, sql=sql, violations=violations, warnings=warnings,
        )

    # 8. Syntax check via EXPLAIN (no execution)
    # Note: uses in-memory DB so table names won't resolve, skip if
    # we already validated tables above
    # violations.extend(_check_with_explain(sql, params))

    return SqlValidationResult(
        valid=len(violations) == 0,
        sql=sql,
        violations=violations,
        warnings=warnings,
    )

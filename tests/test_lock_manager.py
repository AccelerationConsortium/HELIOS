"""Lock manager unit tests — lease, fencing tokens, expiration, contention."""
from __future__ import annotations

import os
import tempfile
import time

import pytest

# Patch settings before importing anything that uses the DB.
_tmpdir = tempfile.mkdtemp(prefix="otbot_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")

from app.core.db import init_db, connection, utcnow_iso, json_dumps  # noqa: E402
from app.services.lock_manager import acquire_lock, release_lock  # noqa: E402

# Fixed run IDs used throughout the tests.
RUN_A = "run-a"
RUN_B = "run-b"


def _insert_dummy_run(conn, run_id: str) -> None:
    """Insert a minimal runs row to satisfy the FK on resource_locks."""
    now = utcnow_iso()
    conn.execute(
        """INSERT OR IGNORE INTO runs
           (id, trigger_type, trigger_payload_json, session_key, status,
            protocol_json, inputs_json, policy_snapshot_json, created_by, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, "test", json_dumps({}), run_id, "running",
         json_dumps({}), json_dumps({}), json_dumps({}), "test", now, now),
    )


@pytest.fixture(autouse=True)
def _fresh_db():
    from app.core.config import get_settings
    get_settings.cache_clear()
    init_db()
    # Seed the two run IDs every test will use.
    with connection() as conn:
        _insert_dummy_run(conn, RUN_A)
        _insert_dummy_run(conn, RUN_B)
        conn.commit()
    yield
    with connection() as conn:
        conn.execute("DELETE FROM resource_locks")
        conn.commit()


# ─── basic acquire / release ────────────────────────────────────────

def test_acquire_new_lock() -> None:
    with connection() as conn:
        result = acquire_lock(conn, resource_id="pipette-1", run_id=RUN_A, ttl_seconds=60)
        conn.commit()
    assert result is not None
    assert result["fencing_token"] == 1
    assert result["resource_id"] == "pipette-1"


def test_release_then_reacquire() -> None:
    with connection() as conn:
        acquire_lock(conn, resource_id="pipette-1", run_id=RUN_A, ttl_seconds=60)
        conn.commit()

    with connection() as conn:
        release_lock(conn, resource_id="pipette-1", run_id=RUN_A)
        conn.commit()

    with connection() as conn:
        result = acquire_lock(conn, resource_id="pipette-1", run_id=RUN_B, ttl_seconds=60)
        conn.commit()
    assert result is not None
    assert result["fencing_token"] == 2


# ─── fencing token monotonicity ─────────────────────────────────────

def test_fencing_token_increments_on_renewal() -> None:
    with connection() as conn:
        r1 = acquire_lock(conn, resource_id="r1", run_id=RUN_A, ttl_seconds=60)
        conn.commit()

    with connection() as conn:
        release_lock(conn, resource_id="r1", run_id=RUN_A)
        conn.commit()

    with connection() as conn:
        r2 = acquire_lock(conn, resource_id="r1", run_id=RUN_A, ttl_seconds=60)
        conn.commit()

    assert r2["fencing_token"] > r1["fencing_token"]


# ─── contention ─────────────────────────────────────────────────────

def test_contention_blocked_by_active_lock() -> None:
    with connection() as conn:
        acquire_lock(conn, resource_id="r1", run_id=RUN_A, ttl_seconds=300)
        conn.commit()

    with connection() as conn:
        result = acquire_lock(conn, resource_id="r1", run_id=RUN_B, ttl_seconds=60)
        conn.commit()
    assert result is None


def test_same_owner_can_reacquire() -> None:
    with connection() as conn:
        acquire_lock(conn, resource_id="r1", run_id=RUN_A, ttl_seconds=60)
        conn.commit()

    with connection() as conn:
        result = acquire_lock(conn, resource_id="r1", run_id=RUN_A, ttl_seconds=60)
        conn.commit()
    assert result is not None
    assert result["fencing_token"] == 2


# ─── lease expiration ───────────────────────────────────────────────

def test_expired_lease_allows_takeover() -> None:
    with connection() as conn:
        acquire_lock(conn, resource_id="r1", run_id=RUN_A, ttl_seconds=1)
        conn.commit()

    time.sleep(1.1)

    with connection() as conn:
        result = acquire_lock(conn, resource_id="r1", run_id=RUN_B, ttl_seconds=60)
        conn.commit()
    assert result is not None
    assert result["fencing_token"] == 2


# ─── independent resources ──────────────────────────────────────────

def test_independent_resources_no_conflict() -> None:
    with connection() as conn:
        r1 = acquire_lock(conn, resource_id="pipette-1", run_id=RUN_A, ttl_seconds=60)
        r2 = acquire_lock(conn, resource_id="pipette-2", run_id=RUN_B, ttl_seconds=60)
        conn.commit()
    assert r1 is not None
    assert r2 is not None


# ─── release idempotency ────────────────────────────────────────────

def test_release_wrong_owner_is_noop() -> None:
    with connection() as conn:
        acquire_lock(conn, resource_id="r1", run_id=RUN_A, ttl_seconds=60)
        conn.commit()

    with connection() as conn:
        release_lock(conn, resource_id="r1", run_id=RUN_B)  # wrong owner
        conn.commit()

    with connection() as conn:
        result = acquire_lock(conn, resource_id="r1", run_id=RUN_B, ttl_seconds=60)
        conn.commit()
    assert result is None  # still locked by run-a

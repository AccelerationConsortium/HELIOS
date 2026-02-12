"""Tests for cross-run failure pattern learning layer."""
from __future__ import annotations

import os
import tempfile
import uuid

_tmpdir = tempfile.mkdtemp(prefix="otbot_failure_learning_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "failure_learning_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db  # noqa: E402
from app.services.failure_signatures import (  # noqa: E402
    FailureChain,
    FailurePattern,
    FailureSignature,
    RecommendedPatch,
    _ensure_failure_learning_tables,
    classify_failure,
    get_failure_chains,
    get_failure_learning_summary,
    get_frequent_failures,
    learn_from_run,
    predict_failures,
    record_failure,
    record_failure_chains,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize a fresh DB for each test."""
    get_settings.cache_clear()
    init_db()
    with connection() as conn:
        # FK-safe cleanup order
        conn.execute("DELETE FROM evolved_priors")
        conn.execute("DELETE FROM evolution_proposals")
        conn.execute("DELETE FROM protocol_templates")
        conn.execute("DELETE FROM batch_candidates")
        conn.execute("DELETE FROM batch_requests")
        conn.execute("DELETE FROM run_reviews")
        conn.execute("DELETE FROM run_kpis")
        conn.execute("DELETE FROM artifacts")
        conn.execute("DELETE FROM provenance_events")
        conn.execute("DELETE FROM run_steps")
        conn.execute("DELETE FROM snapshot_runs")
        conn.execute("DELETE FROM dataset_snapshots")
        conn.execute("DELETE FROM qc_flags")
        conn.execute("DELETE FROM run_failure_signatures")
        conn.execute("DELETE FROM experiment_index")
        conn.execute("DELETE FROM param_schema")
        conn.execute("DELETE FROM runs")
        conn.execute("DELETE FROM campaigns")
        # Also clean failure learning tables (they may not exist yet)
        conn.executescript("""
            DROP TABLE IF EXISTS failure_frequency;
            DROP TABLE IF EXISTS failure_chains;
        """)
        conn.commit()
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_run(run_id: str | None = None) -> str:
    """Insert a minimal test run and return its ID."""
    rid = run_id or str(uuid.uuid4())
    with connection() as conn:
        conn.execute(
            "INSERT INTO runs (id, campaign_id, trigger_type, trigger_payload_json, "
            "session_key, status, protocol_json, inputs_json, policy_snapshot_json, "
            "created_by, created_at, updated_at) "
            "VALUES (?, NULL, 'manual', '{}', 'test', 'completed', '{}', '{}', '{}', "
            "'test', datetime('now'), datetime('now'))",
            (rid,),
        )
        conn.commit()
    return rid


def _add_failed_step(run_id: str, step_key: str, primitive: str, error: str) -> None:
    """Insert a failed step into run_steps."""
    step_id = str(uuid.uuid4())
    with connection() as conn:
        conn.execute(
            "INSERT INTO run_steps (id, run_id, step_key, primitive, params_json, "
            "depends_on_json, resources_json, status, attempt, idempotency_key, error) "
            "VALUES (?, ?, ?, ?, '{}', '[]', '[]', 'failed', 1, ?, ?)",
            (step_id, run_id, step_key, primitive, str(uuid.uuid4()), error),
        )
        conn.commit()


def _add_succeeded_step(run_id: str, step_key: str, primitive: str) -> None:
    """Insert a succeeded step into run_steps."""
    step_id = str(uuid.uuid4())
    with connection() as conn:
        conn.execute(
            "INSERT INTO run_steps (id, run_id, step_key, primitive, params_json, "
            "depends_on_json, resources_json, status, attempt, idempotency_key, error) "
            "VALUES (?, ?, ?, ?, '{}', '[]', '[]', 'succeeded', 1, ?, NULL)",
            (step_id, run_id, step_key, primitive, str(uuid.uuid4())),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# FailurePattern dataclass
# ---------------------------------------------------------------------------


class TestFailurePatternDataclass:
    def test_failure_pattern_frozen(self):
        """FailurePattern is immutable — assigning to a field raises."""
        pattern = FailurePattern(
            failure_type="tip_shortage",
            likely_cause="tip_missing",
            primitive="robot.pick_up_tip",
            occurrence_count=3,
            first_seen_at="2025-01-01T00:00:00Z",
            last_seen_at="2025-01-02T00:00:00Z",
            run_ids=("run-1", "run-2"),
        )
        with pytest.raises(AttributeError):
            pattern.failure_type = "other"

    def test_failure_pattern_to_dict(self):
        """to_dict() returns correct dict with run_ids as list."""
        pattern = FailurePattern(
            failure_type="tip_shortage",
            likely_cause="tip_missing",
            primitive="robot.pick_up_tip",
            occurrence_count=2,
            first_seen_at="2025-01-01T00:00:00Z",
            last_seen_at="2025-01-02T00:00:00Z",
            run_ids=("run-a", "run-b"),
        )
        d = pattern.to_dict()
        assert d["failure_type"] == "tip_shortage"
        assert d["likely_cause"] == "tip_missing"
        assert d["primitive"] == "robot.pick_up_tip"
        assert d["occurrence_count"] == 2
        assert d["first_seen_at"] == "2025-01-01T00:00:00Z"
        assert d["last_seen_at"] == "2025-01-02T00:00:00Z"
        # run_ids serialized as a list, not a tuple
        assert d["run_ids"] == ["run-a", "run-b"]
        assert isinstance(d["run_ids"], list)


# ---------------------------------------------------------------------------
# FailureChain dataclass
# ---------------------------------------------------------------------------


class TestFailureChainDataclass:
    def test_failure_chain_frozen(self):
        """FailureChain is immutable — assigning to a field raises."""
        chain = FailureChain(
            predecessor_type="tip_shortage",
            predecessor_cause="tip_missing",
            successor_type="liquid_insufficient",
            successor_cause="insufficient_liquid",
            co_occurrence_count=5,
            confidence=0.8,
        )
        with pytest.raises(AttributeError):
            chain.predecessor_type = "other"

    def test_failure_chain_to_dict(self):
        """to_dict() returns correct dict."""
        chain = FailureChain(
            predecessor_type="tip_shortage",
            predecessor_cause="tip_missing",
            successor_type="instrument_timeout",
            successor_cause="connection_lost",
            co_occurrence_count=3,
            confidence=0.6,
        )
        d = chain.to_dict()
        assert d["predecessor_type"] == "tip_shortage"
        assert d["predecessor_cause"] == "tip_missing"
        assert d["successor_type"] == "instrument_timeout"
        assert d["successor_cause"] == "connection_lost"
        assert d["co_occurrence_count"] == 3
        assert d["confidence"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# record_failure
# ---------------------------------------------------------------------------


class TestRecordFailure:
    def test_record_new_failure(self):
        """Recording a failure creates one row with count=1."""
        sig = classify_failure("step_1", "robot.pick_up_tip", "no tips available")
        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            record_failure(conn, sig, "run-001")
            conn.commit()

        with connection() as conn:
            row = conn.execute(
                "SELECT occurrence_count, run_ids_json FROM failure_frequency "
                "WHERE failure_type = ? AND likely_cause = ? AND primitive = ?",
                (sig.failure_type, sig.likely_cause, sig.primitive),
            ).fetchone()
            assert row is not None
            assert row["occurrence_count"] == 1
            assert "run-001" in row["run_ids_json"]

    def test_record_duplicate_increments_count(self):
        """Recording the same failure type twice increments count to 2."""
        sig = classify_failure("step_1", "robot.pick_up_tip", "no tips available")
        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            record_failure(conn, sig, "run-001")
            record_failure(conn, sig, "run-002")
            conn.commit()

        with connection() as conn:
            row = conn.execute(
                "SELECT occurrence_count, run_ids_json FROM failure_frequency "
                "WHERE failure_type = ? AND likely_cause = ? AND primitive = ?",
                (sig.failure_type, sig.likely_cause, sig.primitive),
            ).fetchone()
            assert row is not None
            assert row["occurrence_count"] == 2
            assert "run-001" in row["run_ids_json"]
            assert "run-002" in row["run_ids_json"]

    def test_record_different_failures_separate_rows(self):
        """Two different failure types produce two separate rows."""
        sig_tip = classify_failure("step_1", "robot.pick_up_tip", "no tips available")
        sig_timeout = classify_failure("step_2", "robot.home", "step timeout after 30s")
        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            record_failure(conn, sig_tip, "run-001")
            record_failure(conn, sig_timeout, "run-001")
            conn.commit()

        with connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM failure_frequency",
            ).fetchone()["cnt"]
            assert count == 2


# ---------------------------------------------------------------------------
# record_failure_chains
# ---------------------------------------------------------------------------


class TestRecordChains:
    def test_chain_from_two_signatures(self):
        """Two consecutive failures create one chain row with count=1."""
        sig_a = classify_failure("step_1", "robot.pick_up_tip", "no tips available")
        sig_b = classify_failure("step_2", "robot.home", "step timeout after 30s")
        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            # record_failure for both so predecessor has a frequency row
            record_failure(conn, sig_a, "run-001")
            record_failure(conn, sig_b, "run-001")
            record_failure_chains(conn, [sig_a, sig_b], "run-001")
            conn.commit()

        with connection() as conn:
            row = conn.execute(
                "SELECT co_occurrence_count, confidence FROM failure_chains "
                "WHERE predecessor_type = ? AND predecessor_cause = ? "
                "  AND successor_type = ? AND successor_cause = ?",
                (
                    sig_a.failure_type,
                    sig_a.likely_cause,
                    sig_b.failure_type,
                    sig_b.likely_cause,
                ),
            ).fetchone()
            assert row is not None
            assert row["co_occurrence_count"] == 1

    def test_chain_confidence_grows(self):
        """Recording the same chain pair twice increases confidence."""
        sig_a = classify_failure("step_1", "robot.pick_up_tip", "no tips available")
        sig_b = classify_failure("step_2", "robot.home", "step timeout after 30s")

        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            # First occurrence
            record_failure(conn, sig_a, "run-001")
            record_failure(conn, sig_b, "run-001")
            record_failure_chains(conn, [sig_a, sig_b], "run-001")
            conn.commit()

        with connection() as conn:
            row_before = conn.execute(
                "SELECT confidence FROM failure_chains "
                "WHERE predecessor_type = ? AND predecessor_cause = ?",
                (sig_a.failure_type, sig_a.likely_cause),
            ).fetchone()
            confidence_1 = row_before["confidence"]

        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            # Second occurrence — bump the predecessor frequency too
            record_failure(conn, sig_a, "run-002")
            record_failure(conn, sig_b, "run-002")
            record_failure_chains(conn, [sig_a, sig_b], "run-002")
            conn.commit()

        with connection() as conn:
            row_after = conn.execute(
                "SELECT co_occurrence_count, confidence FROM failure_chains "
                "WHERE predecessor_type = ? AND predecessor_cause = ?",
                (sig_a.failure_type, sig_a.likely_cause),
            ).fetchone()
            assert row_after["co_occurrence_count"] == 2
            # With 2 co-occurrences and 2 predecessor occurrences,
            # confidence = 2/2 = 1.0 (or at least >= first confidence)
            assert row_after["confidence"] >= confidence_1


# ---------------------------------------------------------------------------
# learn_from_run
# ---------------------------------------------------------------------------


class TestLearnFromRun:
    def test_learn_from_run_with_failures(self):
        """learn_from_run records patterns + chains for a run with failures."""
        rid = _create_test_run()
        _add_failed_step(rid, "step_1", "robot.pick_up_tip", "no tips available in rack")
        _add_failed_step(rid, "step_2", "robot.home", "step timeout after 30s")

        learn_from_run(rid)

        # Both failure patterns should be recorded
        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            freq_count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM failure_frequency",
            ).fetchone()["cnt"]
            assert freq_count == 2

            # One chain between the two consecutive failures
            chain_count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM failure_chains",
            ).fetchone()["cnt"]
            assert chain_count == 1

    def test_learn_from_run_no_failures(self):
        """learn_from_run with only succeeded steps records nothing."""
        rid = _create_test_run()
        _add_succeeded_step(rid, "step_1", "robot.home")
        _add_succeeded_step(rid, "step_2", "robot.aspirate")

        learn_from_run(rid)

        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            freq_count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM failure_frequency",
            ).fetchone()["cnt"]
            assert freq_count == 0

            chain_count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM failure_chains",
            ).fetchone()["cnt"]
            assert chain_count == 0


# ---------------------------------------------------------------------------
# get_frequent_failures
# ---------------------------------------------------------------------------


class TestGetFrequentFailures:
    def test_frequent_failures_threshold(self):
        """Only patterns meeting min_count are returned."""
        sig = classify_failure("step_1", "robot.pick_up_tip", "no tips available")
        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            for i in range(5):
                record_failure(conn, sig, f"run-{i:03d}")
            conn.commit()

        results = get_frequent_failures(min_count=3)
        assert len(results) == 1
        assert results[0].failure_type == "tip_shortage"
        assert results[0].occurrence_count == 5

        results_high = get_frequent_failures(min_count=10)
        assert len(results_high) == 0

    def test_frequent_failures_ordering(self):
        """Results are ordered by occurrence_count descending."""
        sig_a = classify_failure("step_1", "robot.pick_up_tip", "no tips available")
        sig_b = classify_failure("step_2", "robot.home", "step timeout after 30s")

        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            # Record A five times
            for i in range(5):
                record_failure(conn, sig_a, f"run-a-{i}")
            # Record B three times
            for i in range(3):
                record_failure(conn, sig_b, f"run-b-{i}")
            conn.commit()

        results = get_frequent_failures(min_count=1)
        assert len(results) == 2
        assert results[0].failure_type == sig_a.failure_type  # 5 occurrences
        assert results[1].failure_type == sig_b.failure_type  # 3 occurrences
        assert results[0].occurrence_count > results[1].occurrence_count


# ---------------------------------------------------------------------------
# predict_failures
# ---------------------------------------------------------------------------


class TestPredictFailures:
    def test_predict_from_chain(self):
        """Predict returns successor when chain has sufficient confidence."""
        sig_tip = classify_failure("step_1", "robot.pick_up_tip", "no tips available")
        sig_volume = classify_failure(
            "step_2", "robot.aspirate", "volume error mismatch",
        )

        # Build up chain data: tip_shortage -> volume_delivery_failure
        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            # Record predecessor frequency (needed for confidence computation)
            record_failure(conn, sig_tip, "run-001")
            record_failure(conn, sig_volume, "run-001")
            record_failure_chains(conn, [sig_tip, sig_volume], "run-001")
            conn.commit()

        # Now predict: given tip_shortage, what comes next?
        predictions = predict_failures([sig_tip])
        predicted_types = {p.failure_type for p in predictions}
        assert "volume_delivery_failure" in predicted_types
        for p in predictions:
            assert "predicted" in p.tags

    def test_predict_no_chains_returns_empty(self):
        """predict_failures returns empty list when no chains exist."""
        sig = classify_failure("step_1", "robot.pick_up_tip", "no tips available")

        # Ensure tables exist but are empty
        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            conn.commit()

        predictions = predict_failures([sig])
        assert predictions == []


# ---------------------------------------------------------------------------
# get_failure_learning_summary
# ---------------------------------------------------------------------------


class TestFailureSummary:
    def test_learning_summary(self):
        """Summary returns correct counts for patterns and chains."""
        sig_a = classify_failure("step_1", "robot.pick_up_tip", "no tips available")
        sig_b = classify_failure("step_2", "robot.home", "step timeout after 30s")

        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            record_failure(conn, sig_a, "run-001")
            record_failure(conn, sig_b, "run-001")
            record_failure_chains(conn, [sig_a, sig_b], "run-001")
            conn.commit()

        summary = get_failure_learning_summary()
        assert summary["total_patterns"] == 2
        assert summary["chain_count"] == 1
        assert isinstance(summary["top_failures"], list)
        assert len(summary["top_failures"]) == 2
        assert summary["prediction_accuracy"] == 0.0

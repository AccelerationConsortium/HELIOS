"""Tests for failure signature schema and classifier."""
from __future__ import annotations

import os
import tempfile

_tmpdir = tempfile.mkdtemp(prefix="otbot_failure_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "failure_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db  # noqa: E402
from app.services.failure_signatures import (  # noqa: E402
    FAILURE_TYPES,
    LIKELY_CAUSES,
    SEVERITIES,
    FailureSignature,
    RecommendedPatch,
    classify_failure,
    classify_run_failures,
    summarize_failure_signatures,
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
        conn.execute("DELETE FROM runs")
        conn.commit()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_failure_types_nonempty(self):
        assert len(FAILURE_TYPES) >= 10
        assert "unknown" in FAILURE_TYPES

    def test_severities_complete(self):
        assert SEVERITIES == {"CRITICAL", "HIGH", "MEDIUM", "LOW", "BYPASS"}

    def test_likely_causes_nonempty(self):
        assert len(LIKELY_CAUSES) >= 10


# ---------------------------------------------------------------------------
# FailureSignature dataclass
# ---------------------------------------------------------------------------


class TestFailureSignature:
    def test_frozen(self):
        """FailureSignature is immutable."""
        sig = classify_failure("step_1", "robot.pick_up_tip", "no tips available")
        with pytest.raises(AttributeError):
            sig.failure_type = "other"

    def test_to_dict_roundtrip(self):
        """to_dict() -> from_dict() preserves all fields."""
        sig = classify_failure("step_1", "robot.aspirate", "insufficient liquid in well")
        d = sig.to_dict()
        recovered = FailureSignature.from_dict(d)
        assert recovered.failure_type == sig.failure_type
        assert recovered.severity == sig.severity
        assert recovered.retryable == sig.retryable
        assert recovered.step_key == sig.step_key


# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------


class TestClassifyFailure:
    def test_tip_shortage(self):
        sig = classify_failure("step_1", "robot.pick_up_tip", "no tips available in rack")
        assert sig.failure_type == "tip_shortage"
        assert sig.likely_cause == "tip_missing"
        assert sig.retryable is True
        assert sig.severity == "CRITICAL"
        assert sig.confidence > 0.5

    def test_liquid_insufficient(self):
        sig = classify_failure("step_2", "robot.aspirate", "insufficient liquid in source well")
        assert sig.failure_type == "liquid_insufficient"
        assert sig.retryable is True

    def test_temperature_overshoot(self):
        sig = classify_failure(
            "step_3", "heat", "temperature overshoot detected: 450C > 400C target"
        )
        assert sig.failure_type == "temperature_overshoot"
        assert sig.severity == "CRITICAL"
        assert sig.retryable is False

    def test_impedance_anomaly(self):
        sig = classify_failure(
            "step_4", "squidstat.run_experiment", "impedance anomaly: spike detected"
        )
        assert sig.failure_type == "impedance_anomaly"

    def test_disconnection(self):
        sig = classify_failure("step_5", "ssh.connect", "instrument disconnected unexpectedly")
        assert sig.failure_type == "instrument_disconnection"
        assert sig.retryable is True

    def test_timeout(self):
        sig = classify_failure("step_6", "robot.home", "step timeout after 30s")
        assert sig.failure_type == "instrument_timeout"

    def test_unknown_fallback(self):
        """Unrecognized error falls back to 'unknown' with low confidence."""
        sig = classify_failure("step_7", "custom.operation", "something weird happened")
        assert sig.failure_type == "unknown"
        assert sig.confidence < 0.5

    def test_recommended_patch_present(self):
        """Known failures should have recommended patches."""
        sig = classify_failure("step_1", "robot.pick_up_tip", "no tips available")
        assert sig.recommended_patch is not None
        assert isinstance(sig.recommended_patch.action, str)


# ---------------------------------------------------------------------------
# classify_run_failures
# ---------------------------------------------------------------------------


class TestClassifyRunFailures:
    def test_classify_multiple_failures(self):
        steps = [
            {"step_key": "s1", "primitive": "robot.pick_up_tip", "status": "failed", "error": "no tips available"},
            {"step_key": "s2", "primitive": "heat", "status": "succeeded", "error": None},
            {"step_key": "s3", "primitive": "robot.aspirate", "status": "failed", "error": "insufficient liquid"},
        ]
        sigs = classify_run_failures(steps)
        assert len(sigs) == 2  # only failed steps
        types = {s.failure_type for s in sigs}
        assert "tip_shortage" in types
        assert "liquid_insufficient" in types

    def test_classify_all_succeeded(self):
        steps = [
            {"step_key": "s1", "primitive": "robot.home", "status": "succeeded", "error": None},
        ]
        sigs = classify_run_failures(steps)
        assert len(sigs) == 0


# ---------------------------------------------------------------------------
# summarize_failure_signatures
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_summary_structure(self):
        sigs = [
            classify_failure("s1", "robot.pick_up_tip", "no tips available"),
            classify_failure("s2", "robot.aspirate", "insufficient liquid"),
            classify_failure("s3", "heat", "temp overshoot detected"),
        ]
        summary = summarize_failure_signatures(sigs)
        assert summary["total_failures"] == 3
        assert "by_type" in summary
        assert "by_severity" in summary
        assert "retryable_count" in summary
        assert summary["retryable_count"] + summary["non_retryable_count"] == 3

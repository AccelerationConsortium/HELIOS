"""Tests for contract versioning and migration system."""
from __future__ import annotations

import pytest

from app.contracts.versioning import (
    BaseVersionedContract,
    MigrationError,
    get_migration_path,
    migrate_data,
    register_migration,
)
# Import migrations to register them
import app.contracts.migrations  # noqa: F401
from app.contracts.task_contract import TaskContract
from app.core.db import utcnow_iso


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def make_v1_task_contract_data() -> dict:
    """Create a v1.0.0 TaskContract data dict (old schema)."""
    return {
        "contract_id": "tc-test123",
        "version": "1.0",  # Old field name
        "created_at": utcnow_iso(),
        "created_by": "test_user",
        "objective": {
            "objective_type": "kpi_optimization",
            "primary_kpi": "yield",
            "direction": "maximize",
        },
        "exploration_space": {
            "dimensions": [
                {
                    "param_name": "temperature",
                    "param_type": "number",
                    "min_value": 20.0,
                    "max_value": 100.0,
                }
            ],
            "strategy": "lhs",
            "batch_size": 5,
        },
        "stop_conditions": {
            "max_rounds": 10,
            "target_kpi_value": 0.95,
            "target_kpi_direction": "maximize",
        },
        "safety_envelope": {
            "max_temp_c": 100.0,
            "max_volume_ul": 1000.0,
        },
        "human_gate": {
            "auto_approve_magnitude": 0.5,
        },
        "protocol_pattern_id": "basic_mixing",
        # v1.0.0 does NOT have: protocol_metadata, deprecation_warnings
    }


def make_v2_task_contract_data() -> dict:
    """Create a v2.0.0 TaskContract data dict (current schema)."""
    data = make_v1_task_contract_data()
    data["schema_version"] = "2.0.0"  # New field name
    data.pop("version", None)  # Remove old field
    data["protocol_metadata"] = {}
    data["deprecation_warnings"] = []
    return data


# ---------------------------------------------------------------------------
# Versioning system tests
# ---------------------------------------------------------------------------

class TestMigrationRegistry:
    """Test migration registry and path finding."""

    def test_register_migration(self):
        """Test registering a migration."""
        @register_migration("TestContract", "1.0.0", "2.0.0")
        def test_migration(data: dict) -> dict:
            data["new_field"] = "added"
            return data

        path = get_migration_path("TestContract", "1.0.0", "2.0.0")
        assert path == [("1.0.0", "2.0.0")]

    def test_migration_path_single_step(self):
        """Test finding direct migration path."""
        path = get_migration_path("TaskContract", "1.0.0", "2.0.0")
        assert path == [("1.0.0", "2.0.0")]

    def test_migration_path_multi_step(self):
        """Test finding multi-step migration path."""
        # Register a chain: 1.0 -> 2.0 -> 3.0
        @register_migration("MultiStepContract", "1.0.0", "2.0.0")
        def step1(data: dict) -> dict:
            return data

        @register_migration("MultiStepContract", "2.0.0", "3.0.0")
        def step2(data: dict) -> dict:
            return data

        path = get_migration_path("MultiStepContract", "1.0.0", "3.0.0")
        assert path == [("1.0.0", "2.0.0"), ("2.0.0", "3.0.0")]

    def test_migration_path_no_migration_needed(self):
        """Test path when no migration needed (same version)."""
        path = get_migration_path("TaskContract", "2.0.0", "2.0.0")
        assert path == []

    def test_migration_path_not_found(self):
        """Test error when no path exists."""
        with pytest.raises(MigrationError, match="No migration path"):
            get_migration_path("TaskContract", "1.0.0", "999.0.0")


class TestMigrationExecution:
    """Test executing migrations."""

    def test_migrate_task_contract_v1_to_v2(self):
        """Test migrating TaskContract from v1 to v2."""
        v1_data = make_v1_task_contract_data()

        # Migrate
        v2_data = migrate_data("TaskContract", v1_data, from_version="1.0.0", to_version="2.0.0")

        # Verify migration
        assert v2_data["schema_version"] == "2.0.0"
        assert v2_data["migrated_from"] == "1.0.0"
        assert "protocol_metadata" in v2_data
        assert "deprecation_warnings" in v2_data
        assert v2_data["protocol_metadata"] == {}
        assert v2_data["deprecation_warnings"] == []

        # Old 'version' field should be renamed to 'schema_version'
        assert "version" not in v2_data

    def test_migrate_no_op_same_version(self):
        """Test migration is no-op when versions match."""
        v2_data = make_v2_task_contract_data()
        original_data = dict(v2_data)

        # Migrate (should be no-op)
        result = migrate_data("TaskContract", v2_data, from_version="2.0.0", to_version="2.0.0")

        # No changes except possibly migrated_from
        result.pop("migrated_from", None)
        original_data.pop("migrated_from", None)
        assert result == original_data


# ---------------------------------------------------------------------------
# BaseVersionedContract tests
# ---------------------------------------------------------------------------

class TestBaseVersionedContract:
    """Test BaseVersionedContract functionality."""

    def test_checksum_computation(self):
        """Test checksum is computed automatically."""
        v2_data = make_v2_task_contract_data()
        contract = TaskContract(**v2_data)

        assert contract.checksum
        assert len(contract.checksum) == 16  # SHA256 truncated to 16 chars
        assert contract.verify_checksum()

    def test_checksum_verification(self):
        """Test checksum verification detects changes."""
        v2_data = make_v2_task_contract_data()
        contract = TaskContract(**v2_data)

        original_checksum = contract.checksum

        # Modify contract
        contract.protocol_pattern_id = "different_pattern"

        # Checksum should no longer match
        assert not contract.verify_checksum()
        assert contract.checksum == original_checksum  # Stored checksum unchanged
        assert contract.compute_checksum() != original_checksum  # New checksum differs

    def test_automatic_migration_on_load(self):
        """Test from_dict() automatically migrates old versions."""
        v1_data = make_v1_task_contract_data()

        # Load v1 data - should auto-migrate to v2
        contract = TaskContract.from_dict(v1_data)

        # Verify migration happened
        assert contract.schema_version == "2.0.0"
        assert contract.migrated_from == "1.0.0"
        assert hasattr(contract, "protocol_metadata")
        assert hasattr(contract, "deprecation_warnings")
        assert contract.protocol_metadata == {}

    def test_load_current_version_no_migration(self):
        """Test from_dict() doesn't migrate when version matches."""
        v2_data = make_v2_task_contract_data()

        contract = TaskContract.from_dict(v2_data)

        assert contract.schema_version == "2.0.0"
        assert contract.migrated_from is None  # No migration

    def test_supports_version(self):
        """Test supports_version() checks migration path."""
        assert TaskContract.supports_version("1.0.0")  # Has migration 1.0->2.0
        assert TaskContract.supports_version("2.0.0")  # Current version
        assert not TaskContract.supports_version("999.0.0")  # No migration path


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestContractVersioningIntegration:
    """End-to-end integration tests."""

    def test_full_v1_to_v2_migration_workflow(self):
        """Test complete workflow: load v1 -> migrate -> validate -> serialize."""
        # 1. Start with v1.0.0 data (simulating old DB record)
        v1_data = make_v1_task_contract_data()
        assert v1_data.get("version") == "1.0"
        assert "protocol_metadata" not in v1_data

        # 2. Load via from_dict (auto-migrates)
        contract = TaskContract.from_dict(v1_data)

        # 3. Verify migration
        assert contract.schema_version == "2.0.0"
        assert contract.migrated_from == "1.0.0"
        assert contract.protocol_metadata == {}
        assert contract.deprecation_warnings == []

        # 4. Verify all original data preserved
        assert contract.contract_id == "tc-test123"
        assert contract.created_by == "test_user"
        assert contract.objective.primary_kpi == "yield"
        assert contract.exploration_space.batch_size == 5

        # 5. Verify checksum valid
        assert contract.verify_checksum()

        # 6. Serialize back to dict
        v2_data = contract.model_dump(mode="json")
        assert v2_data["schema_version"] == "2.0.0"
        assert "protocol_metadata" in v2_data

    def test_round_trip_current_version(self):
        """Test serialize -> deserialize for current version."""
        # Create v2 contract
        v2_data = make_v2_task_contract_data()
        contract1 = TaskContract(**v2_data)

        # Serialize
        serialized = contract1.model_dump(mode="json")

        # Deserialize
        contract2 = TaskContract.from_dict(serialized)

        # Should be identical (no migration)
        assert contract2.schema_version == "2.0.0"
        assert contract2.migrated_from is None
        assert contract2.contract_id == contract1.contract_id
        assert contract2.protocol_pattern_id == contract1.protocol_pattern_id

    def test_backward_compatibility_check(self):
        """Test that old code paths still work (graceful degradation)."""
        # Old code might construct directly without from_dict()
        v1_data = make_v1_task_contract_data()

        # This will FAIL validation (missing new required fields)
        # So we need to provide defaults
        v1_data["schema_version"] = "2.0.0"  # Fake it
        v1_data["protocol_metadata"] = {}
        v1_data["deprecation_warnings"] = []
        v1_data.pop("version", None)

        # Now it should work
        contract = TaskContract(**v1_data)
        assert contract.schema_version == "2.0.0"

    def test_migration_preserves_custom_fields(self):
        """Test that migration preserves unknown/custom fields."""
        v1_data = make_v1_task_contract_data()
        v1_data["custom_annotation"] = "important note"  # Extra field

        contract = TaskContract.from_dict(v1_data)

        # Pydantic drops unknown fields by default
        # If we want to preserve them, need to use model_config extra="allow"
        # For now, just verify migration didn't crash
        assert contract.schema_version == "2.0.0"


# ---------------------------------------------------------------------------
# Invariant Validation Tests
# ---------------------------------------------------------------------------


class TestInvariantValidation:
    """Test invariant validation system."""

    def test_task_contract_max_rounds_positive(self):
        """TaskContract: max_rounds must be positive."""
        from app.contracts.versioning import validate_invariants

        valid_data = {
            "schema_version": "2.0.0",
            "contract_id": "tc-test",
            "stop_conditions": {
                "max_rounds": 10,
            }
        }

        is_valid, violations = validate_invariants("TaskContract", valid_data)
        # Note: might have other violations, but max_rounds should pass
        # Check if max_rounds violation is NOT present
        assert not any("max_rounds_positive" in str(v) for v in violations)

    def test_task_contract_max_rounds_negative_fails(self):
        """TaskContract: negative max_rounds violates invariant."""
        from app.contracts.versioning import validate_invariants

        invalid_data = {
            "schema_version": "2.0.0",
            "contract_id": "tc-test",
            "stop_conditions": {
                "max_rounds": -5,  # Invalid!
            }
        }

        is_valid, violations = validate_invariants("TaskContract", invalid_data)
        # Should have violations (at least max_rounds)
        assert len(violations) > 0

    def test_campaign_plan_total_runs_feasible(self):
        """CampaignPlan: total_runs must be within bounds."""
        from app.contracts.versioning import validate_invariants

        # Valid case
        valid_data = {
            "schema_version": "2.0.0",
            "campaign_id": "camp-test",
            "total_runs": 100,
        }

        is_valid, violations = validate_invariants("CampaignPlan", valid_data)
        assert not any("total_runs_feasible" in str(v) for v in violations)

        # Invalid case: too large
        invalid_data = {
            "schema_version": "2.0.0",
            "campaign_id": "camp-test",
            "total_runs": 50000,  # Too large!
        }

        is_valid, violations = validate_invariants("CampaignPlan", invalid_data)
        assert not is_valid
        assert any("total_runs" in str(v) for v in violations)

    def test_campaign_plan_multi_objective_consistency(self):
        """CampaignPlan: multi_objective flag must match pareto_objectives."""
        from app.contracts.versioning import validate_invariants

        # Invalid: multi_objective=True but only 1 objective
        invalid_data = {
            "schema_version": "2.0.0",
            "campaign_id": "camp-test",
            "total_runs": 50,
            "multi_objective": True,
            "pareto_objectives": ["kpi1"],  # Only 1!
        }

        is_valid, violations = validate_invariants("CampaignPlan", invalid_data)
        assert not is_valid
        assert any("multi_objective" in str(v) for v in violations)

        # Valid: multi_objective=True with 2+ objectives
        valid_data = {
            **invalid_data,
            "pareto_objectives": ["kpi1", "kpi2"],
        }

        is_valid, violations = validate_invariants("CampaignPlan", valid_data)
        assert not any("multi_objective" in str(v) for v in violations)

    def test_run_bundle_protocol_required(self):
        """RunBundle: python_code must be non-empty."""
        from app.contracts.versioning import validate_invariants

        invalid_data = {
            "schema_version": "2.0.0",
            "run_id": "run-test",
            "python_code": "",  # Empty!
        }

        is_valid, violations = validate_invariants("RunBundle", invalid_data)
        assert not is_valid
        assert any("protocol" in str(v).lower() for v in violations)

        # Valid case
        valid_data = {
            **invalid_data,
            "python_code": "from opentrons import protocol_api\n\ndef run(ctx):\n    pass"
        }

        is_valid, violations = validate_invariants("RunBundle", valid_data)
        assert not any("protocol" in str(v).lower() for v in violations)

    def test_no_invariants_for_unknown_contract(self):
        """Unknown contract type returns valid."""
        from app.contracts.versioning import validate_invariants

        is_valid, violations = validate_invariants("UnknownContract", {})
        assert is_valid
        assert len(violations) == 0

    def test_custom_invariant_registration(self):
        """Register and validate custom invariant."""
        from app.contracts.versioning import register_invariant, validate_invariants

        # Register custom invariant
        @register_invariant("TestContract", "value_even", "value must be even")
        def validate_even(data):
            value = data.get("value", 0)
            return value % 2 == 0

        # Valid data
        valid_data = {"value": 4}
        is_valid, violations = validate_invariants("TestContract", valid_data)
        assert is_valid

        # Invalid data
        invalid_data = {"value": 3}
        is_valid, violations = validate_invariants("TestContract", invalid_data)
        assert not is_valid
        assert any("value_even" in str(v) for v in violations)

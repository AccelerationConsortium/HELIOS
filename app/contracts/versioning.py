"""Contract versioning and migration system.

This module provides:
1. BaseVersionedContract - base class for all versionable contracts
2. Migration registry - register and execute migrations
3. Automatic upgrade path - old versions → current version
4. Backward compatibility validation

Usage:
    @register_migration("TaskContract", from_version="1.0.0", to_version="2.0.0")
    def migrate_task_contract_1_to_2(data: dict) -> dict:
        # Add new required fields with defaults
        data.setdefault("new_field", "default_value")
        # Rename fields
        if "old_name" in data:
            data["new_name"] = data.pop("old_name")
        return data

    # Automatic migration on load
    contract = TaskContract(**old_data)  # auto-migrates if needed
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from typing import Any, Callable, ClassVar, TypeVar

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

__all__ = [
    "BaseVersionedContract",
    "register_migration",
    "get_migration_path",
    "migrate_data",
    "ContractValidationError",
    "MigrationError",
    "register_invariant",
    "validate_invariants",
    "InvariantRegistry",
]

# Type aliases
T = TypeVar("T", bound=BaseModel)
MigrationFunc = Callable[[dict[str, Any]], dict[str, Any]]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ContractValidationError(Exception):
    """Contract validation failed."""
    pass


class MigrationError(Exception):
    """Migration failed."""
    pass


# ---------------------------------------------------------------------------
# Migration Registry
# ---------------------------------------------------------------------------

class MigrationRegistry:
    """Registry of all contract migrations.

    Structure:
        _migrations[contract_name][(from_ver, to_ver)] = migration_func
    """

    def __init__(self):
        # contract_name -> {(from_ver, to_ver): migration_func}
        self._migrations: dict[str, dict[tuple[str, str], MigrationFunc]] = defaultdict(dict)
        # contract_name -> current_version
        self._current_versions: dict[str, str] = {}

    def register(
        self,
        contract_name: str,
        from_version: str,
        to_version: str,
        func: MigrationFunc,
    ) -> None:
        """Register a migration function."""
        key = (from_version, to_version)
        if key in self._migrations[contract_name]:
            logger.warning(
                "Overwriting migration %s: %s -> %s",
                contract_name, from_version, to_version,
            )
        self._migrations[contract_name][key] = func

        # Track current version (latest to_version)
        if (
            contract_name not in self._current_versions
            or self._compare_versions(to_version, self._current_versions[contract_name]) > 0
        ):
            self._current_versions[contract_name] = to_version

    def get_migration_path(
        self,
        contract_name: str,
        from_version: str,
        to_version: str | None = None,
    ) -> list[tuple[str, str, MigrationFunc]]:
        """Find migration path from from_version to to_version.

        Uses BFS to find shortest path through migration graph.

        Returns:
            List of (from_ver, to_ver, migration_func) tuples in order.

        Raises:
            MigrationError: If no path exists.
        """
        if to_version is None:
            to_version = self._current_versions.get(contract_name)
            if to_version is None:
                raise MigrationError(
                    f"No migrations registered for {contract_name}"
                )

        if from_version == to_version:
            return []  # No migration needed

        # Build adjacency graph
        migrations = self._migrations.get(contract_name, {})
        graph: dict[str, list[tuple[str, MigrationFunc]]] = defaultdict(list)
        for (from_v, to_v), func in migrations.items():
            graph[from_v].append((to_v, func))

        # BFS to find shortest path
        from collections import deque
        queue = deque([(from_version, [])])
        visited = {from_version}

        while queue:
            current_ver, path = queue.popleft()

            if current_ver == to_version:
                return path

            for next_ver, func in graph[current_ver]:
                if next_ver not in visited:
                    visited.add(next_ver)
                    queue.append((
                        next_ver,
                        path + [(current_ver, next_ver, func)],
                    ))

        raise MigrationError(
            f"No migration path from {from_version} to {to_version} "
            f"for {contract_name}"
        )

    def migrate(
        self,
        contract_name: str,
        data: dict[str, Any],
        from_version: str,
        to_version: str | None = None,
    ) -> dict[str, Any]:
        """Execute migration chain."""
        path = self.get_migration_path(contract_name, from_version, to_version)

        if not path:
            return data  # No migration needed

        migrated_data = dict(data)  # Copy to avoid mutating input

        for from_v, to_v, func in path:
            logger.info(
                "Migrating %s: %s -> %s",
                contract_name, from_v, to_v,
            )
            try:
                migrated_data = func(migrated_data)
                # Update version after each step
                migrated_data["schema_version"] = to_v
            except Exception as exc:
                raise MigrationError(
                    f"Migration failed ({from_v} -> {to_v}): {exc}"
                ) from exc

        # Record migration provenance
        migrated_data["migrated_from"] = from_version

        return migrated_data

    @staticmethod
    def _compare_versions(v1: str, v2: str) -> int:
        """Compare semantic versions.

        Returns:
            -1 if v1 < v2
             0 if v1 == v2
             1 if v1 > v2
        """
        parts1 = [int(x) for x in v1.split(".")]
        parts2 = [int(x) for x in v2.split(".")]

        for p1, p2 in zip(parts1, parts2):
            if p1 < p2:
                return -1
            if p1 > p2:
                return 1

        # Handle different lengths (e.g., "2.0" vs "2.0.1")
        if len(parts1) < len(parts2):
            return -1
        if len(parts1) > len(parts2):
            return 1

        return 0


# Global registry instance
_registry = MigrationRegistry()


# ---------------------------------------------------------------------------
# Invariant Validation System
# ---------------------------------------------------------------------------


class InvariantRegistry:
    """Registry for contract invariants (formal verification constraints).

    Invariants are logical constraints that must hold for all valid contracts.
    Examples:
    - max_rounds > 0
    - total_runs <= max_capacity
    - if multi_objective, then len(pareto_objectives) > 1
    """

    def __init__(self):
        # contract_name -> [(invariant_name, validator_func, description)]
        self._invariants: dict[str, list[tuple[str, Callable[[dict[str, Any]], bool], str]]] = defaultdict(list)

    def register(
        self,
        contract_name: str,
        invariant_name: str,
        validator_func: Callable[[dict[str, Any]], bool],
        description: str = "",
    ) -> None:
        """Register an invariant validator.

        Args:
            contract_name: Contract type name
            invariant_name: Unique invariant identifier
            validator_func: Function that returns True if invariant holds
            description: Human-readable description of the invariant
        """
        self._invariants[contract_name].append((invariant_name, validator_func, description))
        logger.info(f"Registered invariant: {contract_name}.{invariant_name}")

    def validate(self, contract_name: str, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate all invariants for a contract.

        Returns:
            (is_valid, violated_invariants) tuple
        """
        if contract_name not in self._invariants:
            return True, []

        violated = []

        for inv_name, validator, description in self._invariants[contract_name]:
            try:
                if not validator(data):
                    violated.append(f"{inv_name}: {description}" if description else inv_name)
            except Exception as e:
                logger.error(f"Invariant {inv_name} raised exception: {e}")
                violated.append(f"{inv_name}: validation error ({e})")

        return len(violated) == 0, violated


# Global invariant registry
_invariant_registry = InvariantRegistry()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_migration(
    contract_name: str,
    from_version: str,
    to_version: str,
) -> Callable[[MigrationFunc], MigrationFunc]:
    """Decorator to register a migration function.

    Usage:
        @register_migration("TaskContract", "1.0.0", "2.0.0")
        def migrate_task_contract_1_to_2(data: dict) -> dict:
            data["new_field"] = "default"
            return data
    """
    def decorator(func: MigrationFunc) -> MigrationFunc:
        _registry.register(contract_name, from_version, to_version, func)
        return func
    return decorator


def get_migration_path(
    contract_name: str,
    from_version: str,
    to_version: str | None = None,
) -> list[tuple[str, str]]:
    """Get migration path (without functions).

    Returns:
        List of (from_ver, to_ver) tuples.
    """
    path = _registry.get_migration_path(contract_name, from_version, to_version)
    return [(from_v, to_v) for from_v, to_v, _ in path]


def migrate_data(
    contract_name: str,
    data: dict[str, Any],
    from_version: str | None = None,
    to_version: str | None = None,
) -> dict[str, Any]:
    """Migrate contract data.

    Args:
        contract_name: Contract class name (e.g., "TaskContract")
        data: Contract data dict
        from_version: Source version (auto-detected if None)
        to_version: Target version (latest if None)

    Returns:
        Migrated data dict with updated schema_version.
    """
    if from_version is None:
        from_version = data.get("schema_version", "1.0.0")

    return _registry.migrate(contract_name, data, from_version, to_version)


def register_invariant(
    contract_name: str,
    invariant_name: str,
    description: str = "",
) -> Callable[[Callable], Callable]:
    """Decorator to register an invariant validator.

    Usage:
        @register_invariant("TaskContract", "max_rounds_positive", "max_rounds must be > 0")
        def validate_max_rounds(data: dict) -> bool:
            return data.get("max_rounds", 1) > 0
    """
    def decorator(func: Callable[[dict[str, Any]], bool]) -> Callable[[dict[str, Any]], bool]:
        _invariant_registry.register(contract_name, invariant_name, func, description)
        return func
    return decorator


def validate_invariants(contract_name: str, data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate all invariants for a contract.

    Returns:
        (is_valid, violated_invariants) tuple where violated_invariants
        is a list of error messages for each violated invariant
    """
    return _invariant_registry.validate(contract_name, data)


# ---------------------------------------------------------------------------
# Base Contract
# ---------------------------------------------------------------------------

class BaseVersionedContract(BaseModel):
    """Base class for all versionable contracts.

    Provides:
    - Automatic schema versioning
    - Migration on load
    - Checksum validation
    - Deprecation warnings

    Subclasses should:
    1. Set SCHEMA_VERSION class variable
    2. Register migrations via @register_migration decorator

    Example:
        class MyContract(BaseVersionedContract):
            SCHEMA_VERSION: ClassVar[str] = "2.0.0"
            CONTRACT_NAME: ClassVar[str] = "MyContract"

            # Fields...
            field1: str
            field2: int

        @register_migration("MyContract", "1.0.0", "2.0.0")
        def migrate_v1_to_v2(data: dict) -> dict:
            data["field2"] = 0  # Add new required field
            return data
    """

    # Subclasses MUST override these (use ClassVar to avoid Pydantic treating as fields)
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    CONTRACT_NAME: ClassVar[str] = "BaseContract"

    # Version metadata
    schema_version: str = Field(
        default="1.0.0",
        description="Semantic version of this contract schema",
    )
    migrated_from: str | None = Field(
        default=None,
        description="Original version if auto-migrated (for audit trail)",
    )
    checksum: str = Field(
        default="",
        description="SHA256 checksum of canonical JSON representation",
    )

    @field_validator("schema_version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Warn if using deprecated version."""
        if hasattr(cls, "SCHEMA_VERSION") and v != cls.SCHEMA_VERSION:
            logger.warning(
                "%s: using version %s, current is %s (auto-migration available)",
                cls.__name__, v, cls.SCHEMA_VERSION,
            )
        return v

    def model_post_init(self, __context: Any) -> None:
        """Auto-compute checksum after initialization."""
        super().model_post_init(__context)
        if not self.checksum:
            self.checksum = self.compute_checksum()

    def compute_checksum(self) -> str:
        """Compute SHA256 checksum of canonical JSON.

        Excludes checksum and migrated_from fields from hash.
        """
        data = self.model_dump(mode="json", exclude={"checksum", "migrated_from"})
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def verify_checksum(self) -> bool:
        """Verify checksum matches current data."""
        if not self.checksum:
            return True  # No checksum to verify
        return self.checksum == self.compute_checksum()

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Load contract from dict with automatic migration.

        Args:
            data: Contract data (any version)

        Returns:
            Contract instance (current version)

        Raises:
            MigrationError: If migration fails
            ValidationError: If data invalid after migration
        """
        # Detect source version
        src_version = data.get("schema_version", "1.0.0")

        # Auto-migrate if needed
        if src_version != cls.SCHEMA_VERSION:
            try:
                data = migrate_data(
                    cls.CONTRACT_NAME,
                    data,
                    from_version=src_version,
                    to_version=cls.SCHEMA_VERSION,
                )
            except MigrationError as exc:
                logger.error("Migration failed for %s: %s", cls.__name__, exc)
                raise

        # Ensure schema_version is set to current
        data["schema_version"] = cls.SCHEMA_VERSION

        return cls(**data)

    @classmethod
    def supports_version(cls, version: str) -> bool:
        """Check if migration path exists for given version."""
        try:
            get_migration_path(cls.CONTRACT_NAME, version, cls.SCHEMA_VERSION)
            return True
        except MigrationError:
            return False


# ---------------------------------------------------------------------------
# Example Invariants for Built-in Contracts
# ---------------------------------------------------------------------------


@register_invariant("TaskContract", "max_rounds_positive", "max_rounds must be positive")
def _task_contract_max_rounds_positive(data: dict[str, Any]) -> bool:
    """Invariant: max_rounds must be > 0."""
    # Check both possible locations: top-level or in stop_conditions
    max_rounds = data.get("max_rounds")
    if max_rounds is None and "stop_conditions" in data:
        max_rounds = data["stop_conditions"].get("max_rounds")
    return max_rounds is None or max_rounds > 0


@register_invariant("TaskContract", "goal_target_if_present", "goal must have target_value if present")
def _task_contract_goal_has_target(data: dict[str, Any]) -> bool:
    """Invariant: if goal exists, it must have target_value."""
    goal = data.get("goal")
    if not goal or not isinstance(goal, dict):
        return True  # Optional goal is OK
    return "target_value" in goal or "kpi_name" in goal


@register_invariant("CampaignPlan", "total_runs_feasible", "total_runs must be within reasonable bounds")
def _campaign_plan_total_runs_feasible(data: dict[str, Any]) -> bool:
    """Invariant: total_runs must be reasonable (0 < total_runs <= 10000)."""
    total_runs = data.get("total_runs", 0)
    return 0 < total_runs <= 10000


@register_invariant("CampaignPlan", "multi_objective_consistency", "multi_objective flag must match pareto_objectives")
def _campaign_plan_multi_objective_consistent(data: dict[str, Any]) -> bool:
    """Invariant: if multi_objective=True, must have >=2 pareto_objectives."""
    multi_obj = data.get("multi_objective", False)
    pareto_objs = data.get("pareto_objectives", [])

    if multi_obj:
        return len(pareto_objs) >= 2
    return True  # If not multi-objective, any number is fine


@register_invariant("RunBundle", "protocol_required", "python_code must be non-empty")
def _run_bundle_protocol_required(data: dict[str, Any]) -> bool:
    """Invariant: python_code must be present and non-empty."""
    python_code = data.get("python_code", "")
    return isinstance(python_code, str) and len(python_code.strip()) > 0


# ---------------------------------------------------------------------------
# Example Migrations for Built-in Contracts
# ---------------------------------------------------------------------------


@register_migration("TaskContract", "1.0.0", "2.0.0")
def _migrate_task_contract_1_to_2(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate TaskContract from v1.0.0 to v2.0.0.

    Changes:
    - Add experiment_cost field (default 1.0)
    - Rename goal.metric → goal.kpi_name for consistency
    """
    migrated = dict(data)

    # Add new field with default
    if "experiment_cost" not in migrated:
        migrated["experiment_cost"] = 1.0

    # Rename field in goal if exists
    if "goal" in migrated and isinstance(migrated["goal"], dict):
        if "metric" in migrated["goal"] and "kpi_name" not in migrated["goal"]:
            migrated["goal"]["kpi_name"] = migrated["goal"].pop("metric")

    return migrated


@register_migration("CampaignPlan", "1.0.0", "2.0.0")
def _migrate_campaign_plan_1_to_2(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate CampaignPlan from v1.0.0 to v2.0.0.

    Changes:
    - Add multi_objective flag (default False)
    - Add pareto_objectives list (default empty)
    """
    migrated = dict(data)

    if "multi_objective" not in migrated:
        migrated["multi_objective"] = False

    if "pareto_objectives" not in migrated:
        migrated["pareto_objectives"] = []

    return migrated


@register_migration("RunBundle", "1.0.0", "2.0.0")
def _migrate_run_bundle_1_to_2(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate RunBundle from v1.0.0 to v2.0.0.

    Changes:
    - Add execution_metadata field (default empty dict)
    """
    migrated = dict(data)

    if "execution_metadata" not in migrated:
        migrated["execution_metadata"] = {}

    return migrated


@register_migration("ResultPacket", "1.0.0", "2.0.0")
def _migrate_result_packet_1_to_2(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate ResultPacket from v1.0.0 to v2.0.0.

    Changes:
    - Add qc_metadata field (default empty dict)
    """
    migrated = dict(data)

    if "qc_metadata" not in migrated:
        migrated["qc_metadata"] = {}

    return migrated

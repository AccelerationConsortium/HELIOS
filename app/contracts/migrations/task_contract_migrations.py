"""TaskContract migrations.

Version history:
- 1.0.0: Initial version
- 2.0.0: Add protocol_metadata field, rename version -> schema_version
"""
from __future__ import annotations

from typing import Any

from app.contracts.versioning import register_migration

__all__ = []


@register_migration("TaskContract", from_version="1.0.0", to_version="2.0.0")
def migrate_task_contract_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate TaskContract from v1.0.0 to v2.0.0.

    Changes:
    1. Add protocol_metadata field with default empty dict
    2. Rename 'version' field to 'schema_version' (if exists)
    3. Add deprecation_warnings list for tracking issues
    """
    # Rename version -> schema_version (backward compat)
    if "version" in data and "schema_version" not in data:
        data["schema_version"] = data.pop("version")

    # Add new fields with defaults
    data.setdefault("protocol_metadata", {})
    data.setdefault("deprecation_warnings", [])

    # Clean up any old fields that no longer exist
    # (none for this migration, but shown as example)
    # if "obsolete_field" in data:
    #     data.pop("obsolete_field")

    return data


# Future migrations go here:
# @register_migration("TaskContract", from_version="2.0.0", to_version="3.0.0")
# def migrate_task_contract_v2_to_v3(data: dict[str, Any]) -> dict[str, Any]:
#     ...

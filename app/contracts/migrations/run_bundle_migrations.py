"""RunBundle migrations.

Version history:
- 1.0.0: Initial version (protocol_version only)
- 2.0.0: Add schema_version separate from protocol_version
"""
from __future__ import annotations

from typing import Any

from app.contracts.versioning import register_migration

__all__ = []


@register_migration("RunBundle", from_version="1.0.0", to_version="2.0.0")
def migrate_run_bundle_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate RunBundle from v1.0.0 to v2.0.0.

    Changes:
    1. Add schema_version field (separate from protocol_version)
    2. Add execution_metadata field for runtime info
    """
    data.setdefault("schema_version", "2.0.0")
    data.setdefault("execution_metadata", {})

    return data

"""ResultPacket migrations.

Version history:
- 1.0.0: Initial version (no schema_version)
- 2.0.0: Add schema_version, add provenance_chain field
"""
from __future__ import annotations

from typing import Any

from app.contracts.versioning import register_migration

__all__ = []


@register_migration("ResultPacket", from_version="1.0.0", to_version="2.0.0")
def migrate_result_packet_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate ResultPacket from v1.0.0 to v2.0.0.

    Changes:
    1. Add schema_version field
    2. Add provenance_chain for W3C PROV-O compatibility
    """
    data.setdefault("schema_version", "2.0.0")
    data.setdefault("provenance_chain", [])

    return data

"""CampaignPlan migrations.

Version history:
- 1.0.0: Initial version (no schema_version field)
- 2.0.0: Add schema_version, add optimization_history field
"""
from __future__ import annotations

from typing import Any

from app.contracts.versioning import register_migration

__all__ = []


@register_migration("CampaignPlan", from_version="1.0.0", to_version="2.0.0")
def migrate_campaign_plan_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate CampaignPlan from v1.0.0 to v2.0.0.

    Changes:
    1. Add schema_version field
    2. Add optimization_history field for tracking strategy decisions
    """
    data.setdefault("schema_version", "2.0.0")
    data.setdefault("optimization_history", [])

    return data

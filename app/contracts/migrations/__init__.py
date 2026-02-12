"""Contract migration registry.

All migrations are registered here and executed automatically when
loading contracts via BaseVersionedContract.from_dict().

Migration naming convention:
    {contract_name}_v{from}_to_v{to}.py

Example:
    task_contract_v1_to_v2.py
    campaign_plan_v1_to_v2.py
"""
from __future__ import annotations

# Import all migration modules to register them
# New migrations should be imported here
from app.contracts.migrations import (
    task_contract_migrations,
    campaign_plan_migrations,
    run_bundle_migrations,
    result_packet_migrations,
)

__all__ = [
    "task_contract_migrations",
    "campaign_plan_migrations",
    "run_bundle_migrations",
    "result_packet_migrations",
]

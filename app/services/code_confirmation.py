"""Code & Operation Confirmation — user approval flow for generated code and hardware ops.

Manages a lightweight in-memory store of pending confirmations.
Supports multiple confirmation types:
- "code": NLP-generated protocol code review
- "cleaning": Pre/post cleaning workflow approval
- "candidate": Candidate execution approval (handled separately via run_service)

When a confirmable action is generated, it creates a confirmation request.
The orchestrator pauses and emits an SSE event. The user reviews and
approves/rejects/modifies through the API.

All state is in-memory (no DB persistence) since confirmations are
short-lived and tied to the current campaign execution.
"""
from __future__ import annotations

import logging
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "CodeConfirmationRequest",
    "CodeConfirmationResponse",
    "CodeConfirmationStatus",
    "request_code_confirmation",
    "get_pending_confirmation",
    "respond_to_confirmation",
    "get_confirmed_code",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CodeConfirmationStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class CodeConfirmationRequest(BaseModel):
    """Request for user to confirm generated code or an operation."""

    request_id: str = Field(default_factory=lambda: f"ccr-{uuid.uuid4().hex[:12]}")
    confirmation_type: str = Field(
        default="code",
        description="Type of confirmation: 'code' | 'cleaning' | 'operation'",
    )
    python_code: str = ""
    workflow_json: str = ""
    protocol_steps: list[dict[str, Any]] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    plan_candidates: list[dict[str, Any]] = Field(default_factory=list)
    selected_candidate_idx: int = 0
    auto_approve: bool = False
    campaign_id: str = ""
    # Cleaning-specific fields
    workflow_id: str = ""
    skill_ids: list[str] = Field(default_factory=list)
    description: str = ""


class CodeConfirmationResponse(BaseModel):
    """User's response to a code confirmation request."""

    request_id: str
    approved: bool = False
    modified_code: str | None = None
    modified_steps: list[dict[str, Any]] | None = None
    selected_candidate_idx: int | None = None
    rejection_reason: str = ""


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


class _ConfirmationEntry:
    """Internal tracking entry."""

    __slots__ = ("request", "status", "response")

    def __init__(self, request: CodeConfirmationRequest) -> None:
        self.request = request
        self.status = CodeConfirmationStatus.PENDING
        self.response: CodeConfirmationResponse | None = None


_STORE: dict[str, _ConfirmationEntry] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def request_code_confirmation(req: CodeConfirmationRequest) -> str:
    """Store a pending confirmation request.

    If ``auto_approve`` is set, immediately marks as approved.

    Returns the request_id.
    """
    entry = _ConfirmationEntry(req)

    if req.auto_approve:
        entry.status = CodeConfirmationStatus.APPROVED
        entry.response = CodeConfirmationResponse(
            request_id=req.request_id,
            approved=True,
        )
        logger.info("Code confirmation %s auto-approved", req.request_id)
    else:
        logger.info(
            "Code confirmation %s pending user approval (campaign=%s)",
            req.request_id,
            req.campaign_id,
        )

    _STORE[req.request_id] = entry
    return req.request_id


def get_pending_confirmation(request_id: str) -> CodeConfirmationRequest | None:
    """Get a pending confirmation request, or None if not found."""
    entry = _STORE.get(request_id)
    if entry is None:
        return None
    return entry.request


def get_confirmation_status(request_id: str) -> CodeConfirmationStatus | None:
    """Get the current status of a confirmation request."""
    entry = _STORE.get(request_id)
    if entry is None:
        return None
    return entry.status


def respond_to_confirmation(response: CodeConfirmationResponse) -> CodeConfirmationStatus:
    """Process a user's response to a code confirmation request.

    Returns the new status.

    Raises
    ------
    ValueError
        If the request_id is not found.
    """
    entry = _STORE.get(response.request_id)
    if entry is None:
        raise ValueError(f"Confirmation request not found: {response.request_id}")

    entry.response = response

    if response.approved:
        if response.modified_code or response.modified_steps:
            entry.status = CodeConfirmationStatus.MODIFIED
            logger.info("Code confirmation %s approved with modifications", response.request_id)
        else:
            entry.status = CodeConfirmationStatus.APPROVED
            logger.info("Code confirmation %s approved", response.request_id)
    else:
        entry.status = CodeConfirmationStatus.REJECTED
        logger.info(
            "Code confirmation %s rejected: %s",
            response.request_id,
            response.rejection_reason,
        )

    return entry.status


def get_confirmed_code(
    request_id: str,
) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Get the confirmed code (possibly modified).

    Returns ``(python_code, workflow_json, protocol_steps)`` if approved/modified,
    ``None`` if pending or rejected.
    """
    entry = _STORE.get(request_id)
    if entry is None:
        return None

    if entry.status not in (CodeConfirmationStatus.APPROVED, CodeConfirmationStatus.MODIFIED):
        return None

    # If user modified the code, use the modified version
    if entry.response and entry.response.modified_code:
        return (
            entry.response.modified_code,
            entry.request.workflow_json,
            entry.response.modified_steps or entry.request.protocol_steps,
        )

    if entry.response and entry.response.modified_steps:
        return (
            entry.request.python_code,
            entry.request.workflow_json,
            entry.response.modified_steps,
        )

    return (
        entry.request.python_code,
        entry.request.workflow_json,
        entry.request.protocol_steps,
    )


def cleanup_confirmation(request_id: str) -> None:
    """Remove a confirmation entry from the store."""
    _STORE.pop(request_id, None)


def list_pending_confirmations(
    campaign_id: str = "",
    confirmation_type: str = "",
) -> list[CodeConfirmationRequest]:
    """List all pending confirmations, optionally filtered by campaign and type."""
    results = []
    for entry in _STORE.values():
        if entry.status != CodeConfirmationStatus.PENDING:
            continue
        if campaign_id and entry.request.campaign_id != campaign_id:
            continue
        if confirmation_type and entry.request.confirmation_type != confirmation_type:
            continue
        results.append(entry.request)
    return results

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import sys
import sqlite3
import threading
from collections import defaultdict
from typing import Any

from app.adapters.base import InstrumentAdapter
from app.core.config import get_settings
from app.core.constants import RUN_STATUS_FAILED, RUN_STATUS_SUCCEEDED
from app.core.db import run_txn
from app.services.action_contracts import ActionContract, SafetyClass
from app.services.artifact_store import persist_json_artifact, persist_file_artifact
from app.services.audit import record_event
from app.services.error_policy import ErrorPolicy, classify_step_error, classify_step_safety
from app.services.lock_manager import acquire_lock, release_lock
from app.services.primitives_registry import get_registry
from app.services.recovery import RecoveryPolicy, attempt_recovery
from app.services.run_context import RunContext
from app.services.run_service import (
    DomainError,
    worker_append_artifact,
    worker_close_instrument_session,
    worker_complete_run,
    worker_get_completed_step_keys,
    worker_list_steps,
    worker_load_run,
    worker_open_instrument_session,
    worker_set_step_state,
)
from app.services.safety import evaluate_runtime_step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

def _create_adapter() -> InstrumentAdapter:
    """Instantiate the correct adapter based on settings."""
    settings = get_settings()
    mode = settings.adapter_mode

    if mode == "battery_lab":
        from app.adapters.battery_lab import BatteryLabAdapter
        return BatteryLabAdapter(dry_run=settings.adapter_dry_run)

    # Default: simulated
    from app.adapters.simulated_instrument import SimulatedAdapter
    return SimulatedAdapter()


# ---------------------------------------------------------------------------
# Resource lock helpers
# ---------------------------------------------------------------------------

def _acquire_resources(run_id: str, resources: list[str]) -> list[str]:
    settings = get_settings()
    acquired: list[str] = []

    def _txn(conn: sqlite3.Connection) -> list[str]:
        for resource in sorted(set(resources)):
            lock = acquire_lock(
                conn,
                resource_id=resource,
                run_id=run_id,
                ttl_seconds=settings.lock_ttl_seconds,
            )
            if lock is None:
                raise RuntimeError(f"resource busy: {resource}")
            acquired.append(resource)
            record_event(
                conn,
                run_id=run_id,
                actor="lock-manager",
                action="resource.lock_acquired",
                details={"resource_id": resource, "fencing_token": lock["fencing_token"]},
            )
        return acquired

    return run_txn(_txn)


def _release_resources(run_id: str, resources: list[str]) -> None:
    def _txn(conn: sqlite3.Connection) -> None:
        for resource in sorted(set(resources)):
            release_lock(conn, resource_id=resource, run_id=run_id)
            record_event(
                conn,
                run_id=run_id,
                actor="lock-manager",
                action="resource.lock_released",
                details={"resource_id": resource},
            )

    run_txn(_txn)


# ---------------------------------------------------------------------------
# Parallel execution helpers
# ---------------------------------------------------------------------------

def _find_ready_steps(
    steps: list[dict[str, Any]],
    finished_step_keys: set[str],
    failed_step_keys: set[str],
) -> list[dict[str, Any]]:
    """Return all steps whose dependencies are satisfied and haven't run yet."""
    ready: list[dict[str, Any]] = []
    for step in steps:
        if step["step_key"] in finished_step_keys:
            continue
        if step["step_key"] in failed_step_keys:
            continue
        if step["status"] not in ("pending", "running"):
            continue
        deps = set(step.get("depends_on", []))
        if deps.issubset(finished_step_keys):
            ready.append(step)
    return ready


def _partition_by_resources(
    ready_steps: list[dict[str, Any]],
    instrument_id: str,
) -> list[list[dict[str, Any]]]:
    """Partition ready steps into non-conflicting groups by resources.

    Steps that share a resource cannot run in the same parallel batch.
    Returns a list of groups; only the first group runs concurrently,
    remaining groups wait for the next iteration.
    """
    if len(ready_steps) <= 1:
        return [ready_steps] if ready_steps else []

    # Build groups: greedy assignment — each step goes in the first group
    # whose claimed resources don't overlap with the step's resources.
    groups: list[tuple[set[str], list[dict[str, Any]]]] = []
    for step in ready_steps:
        resources = set(step.get("resources", []) or [instrument_id])
        placed = False
        for claimed, group in groups:
            if not claimed & resources:
                claimed.update(resources)
                group.append(step)
                placed = True
                break
        if not placed:
            groups.append((set(resources), [step]))

    return [g[1] for g in groups]


def _get_contract_for_step(primitive: str) -> ActionContract | None:
    """Look up the ActionContract for a primitive from the registry."""
    try:
        registry = get_registry()
        spec = registry.get(primitive)
        if spec is not None:
            return spec.contract
    except Exception:
        pass
    return None


def _execute_step(
    *,
    adapter: InstrumentAdapter,
    run_id: str,
    step: dict[str, Any],
    policy: dict[str, Any],
    instrument_id: str,
    error_policy: ErrorPolicy,
    result_holder: dict[str, Any],
    run_context: RunContext | None = None,
    recovery_policy: RecoveryPolicy | None = None,
    recovery_attempt_counts: dict[str, int] | None = None,
) -> None:
    """Execute a single step — designed to run in a thread.

    Stores outcome in *result_holder* keyed by step_key:
      {"ok": True/False, "bypassed": bool, "error": str | None}

    When *run_context* is provided:
      1. Check preconditions from the primitive's ActionContract
      2. Execute with timeout from the contract's TimeoutConfig
      3. Retry according to safety_class (HAZARDOUS never retries)
      4. On success, apply effects to the RunContext
    """
    step_key = step["step_key"]
    step_id = step["id"]
    primitive = step["primitive"]
    params = step.get("params", {})

    try:
        worker_set_step_state(
            run_id=run_id,
            step_id=step_id,
            status="running",
            actor="worker",
        )

        # --- Contract lookup ---
        contract = _get_contract_for_step(primitive)
        safety_class = classify_step_safety(primitive, contract)

        # --- Precondition check (when RunContext available) ---
        if run_context is not None and contract is not None:
            unmet: list[str] = []
            for precond in contract.preconditions:
                rendered = precond.render(params)
                if not run_context.check_precondition(rendered):
                    unmet.append(rendered)
            if unmet:
                msg = f"precondition(s) not met: {', '.join(unmet)}"
                logger.warning(
                    "step %s (%s): %s", step_key, primitive, msg,
                )
                worker_set_step_state(
                    run_id=run_id,
                    step_id=step_id,
                    status="failed",
                    actor="contract-engine",
                    error=msg,
                )
                result_holder[step_key] = {"ok": False, "bypassed": False, "error": msg}
                return

        # --- Safety gate (runtime interlock checks) ---
        interlock_state = {
            "hardware_interlock_ok": True,
            "cooling_ok": True,
        }
        runtime_gate = evaluate_runtime_step(
            step=step,
            policy_snapshot=policy,
            interlock_state=interlock_state,
        )
        if not runtime_gate.allowed:
            msg = "; ".join(runtime_gate.violations)
            worker_set_step_state(
                run_id=run_id,
                step_id=step_id,
                status="failed",
                actor="safety-engine",
                error=msg,
            )
            result_holder[step_key] = {"ok": False, "bypassed": False, "error": msg}
            return

        resources = list(step.get("resources", []))
        if not resources:
            resources = [instrument_id]

        # --- Timeout + retry config from contract ---
        timeout_seconds: float | None = None
        max_retries: int = 0
        if contract is not None:
            timeout_seconds = contract.timeout.seconds
            # HAZARDOUS primitives never retry
            if safety_class < SafetyClass.HAZARDOUS:
                max_retries = contract.timeout.retries

        try:
            _acquire_resources(run_id, resources)

            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    if timeout_seconds and timeout_seconds > 0:
                        # Execute with timeout using concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            future = pool.submit(
                                adapter.execute_primitive,
                                instrument_id=instrument_id,
                                primitive=primitive,
                                params=params,
                            )
                            result = future.result(timeout=timeout_seconds)
                    else:
                        result = adapter.execute_primitive(
                            instrument_id=instrument_id,
                            primitive=primitive,
                            params=params,
                        )
                    last_exc = None
                    break  # success
                except concurrent.futures.TimeoutError:
                    last_exc = TimeoutError(
                        f"step {step_key} ({primitive}) timed out after {timeout_seconds}s"
                    )
                    if attempt < max_retries:
                        logger.warning(
                            "step %s (%s) attempt %d/%d timed out — retrying",
                            step_key, primitive, attempt + 1, max_retries + 1,
                        )
                    continue
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        logger.warning(
                            "step %s (%s) attempt %d/%d failed — retrying: %s",
                            step_key, primitive, attempt + 1, max_retries + 1, exc,
                        )
                    continue

            if last_exc is not None:
                raise last_exc

            # --- Success: persist artifact ---
            artifact_uri, checksum = persist_json_artifact(run_id, step_id, result)
            worker_append_artifact(
                run_id=run_id,
                step_id=step_id,
                kind="primitive_result",
                uri=artifact_uri,
                checksum=checksum,
                metadata={
                    "primitive": primitive,
                    "step_key": step_key,
                },
            )

            # --- Apply effects to RunContext ---
            if run_context is not None and contract is not None:
                for effect in contract.effects:
                    rendered = effect.render(params)
                    run_context.apply_effect(rendered)

            worker_set_step_state(
                run_id=run_id,
                step_id=step_id,
                status="succeeded",
                actor="worker",
            )
            result_holder[step_key] = {"ok": True, "bypassed": False, "error": None}
        except Exception as exc:
            # --- Attempt adaptive recovery before error classification ---
            recovery_succeeded = False
            if (
                recovery_policy is not None
                and recovery_policy.enabled
                and recovery_attempt_counts is not None
                and run_context is not None
            ):
                recovery_result = attempt_recovery(
                    primitive=primitive,
                    error_text=str(exc),
                    run_id=run_id,
                    instrument_id=instrument_id,
                    adapter=adapter,
                    run_context=run_context,
                    policy_snapshot=policy,
                    step_key=step_key,
                    recovery_policy=recovery_policy,
                    recovery_attempt_counts=recovery_attempt_counts,
                )
                if recovery_result.attempted and recovery_result.succeeded:
                    # Retry original step ONE more time after recovery
                    try:
                        result = adapter.execute_primitive(
                            instrument_id=instrument_id,
                            primitive=primitive,
                            params=params,
                        )
                        # Success: persist artifact + apply effects
                        artifact_uri, checksum = persist_json_artifact(
                            run_id, step_id, result,
                        )
                        worker_append_artifact(
                            run_id=run_id,
                            step_id=step_id,
                            kind="primitive_result",
                            uri=artifact_uri,
                            checksum=checksum,
                            metadata={
                                "primitive": primitive,
                                "step_key": step_key,
                                "recovered": True,
                            },
                        )
                        if run_context is not None and contract is not None:
                            for effect in contract.effects:
                                rendered = effect.render(params)
                                run_context.apply_effect(rendered)
                        worker_set_step_state(
                            run_id=run_id,
                            step_id=step_id,
                            status="succeeded",
                            actor="worker",
                        )
                        result_holder[step_key] = {
                            "ok": True, "bypassed": False,
                            "error": None, "recovered": True,
                        }
                        recovery_succeeded = True
                    except Exception as retry_exc:
                        # Recovery didn't help — use new error for classification
                        exc = retry_exc

            if not recovery_succeeded:
                severity = classify_step_error(primitive, exc)
                if severity == "BYPASS" and error_policy.allow_bypass:
                    logger.warning(
                        "step %s (%s) failed with BYPASS severity — skipping: %s",
                        step_key, primitive, exc,
                    )
                    worker_set_step_state(
                        run_id=run_id,
                        step_id=step_id,
                        status="skipped",
                        actor="worker",
                        error=f"[BYPASS] {exc}",
                    )
                    result_holder[step_key] = {"ok": True, "bypassed": True, "error": str(exc)}
                else:
                    worker_set_step_state(
                        run_id=run_id,
                        step_id=step_id,
                        status="failed",
                        actor="worker",
                        error=str(exc),
                    )
                    result_holder[step_key] = {"ok": False, "bypassed": False, "error": str(exc)}
        finally:
            _release_resources(run_id, resources)
    except Exception as exc:
        result_holder[step_key] = {"ok": False, "bypassed": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------

def execute_run(run_id: str) -> int:
    """Execute a single run using the configured adapter.

    Supports parallel execution: when multiple steps have all dependencies
    satisfied and use non-overlapping resources, they run concurrently via
    threads.  The error policy (CRITICAL / BYPASS) determines whether a
    step failure aborts the entire run.

    Contract-aware execution:
      - Creates a RunContext at the start to track per-run state
      - Each step's preconditions are checked against RunContext
      - Effects are applied to RunContext after successful execution
      - Timeout and retry are configured per-primitive via ActionContract

    Returns 0 on success, 1 on failure.
    """
    run = worker_load_run(run_id)
    policy = run["policy_snapshot"]
    inputs = run["inputs"]
    instrument_id = str(inputs.get("instrument_id", "sim-instrument-1"))

    error_policy = ErrorPolicy.from_policy_snapshot(policy)
    recovery_policy = RecoveryPolicy.from_policy_snapshot(policy)
    recovery_attempt_counts: dict[str, int] = defaultdict(int)

    # Create per-run context for contract precondition/effect tracking
    run_context = RunContext()

    # Create and connect the adapter
    adapter = _create_adapter()
    adapter.connect()

    session_id = worker_open_instrument_session(
        run_id=run_id,
        instrument_id=instrument_id,
        firmware_version=get_settings().default_firmware_version,
        calibration_id=get_settings().default_calibration_id,
    )

    try:
        steps = worker_list_steps(run_id)
        finished_step_keys = worker_get_completed_step_keys(run_id)
        failed_step_keys: set[str] = set()

        while len(finished_step_keys) + len(failed_step_keys) < len(steps):
            ready = _find_ready_steps(steps, finished_step_keys, failed_step_keys)
            if not ready:
                # Check if remaining steps depend on failed steps — deadlock
                remaining = [
                    s for s in steps
                    if s["step_key"] not in finished_step_keys
                    and s["step_key"] not in failed_step_keys
                ]
                if remaining:
                    raise RuntimeError(
                        "no executable step found; dependency on failed step or cycle"
                    )
                break

            groups = _partition_by_resources(ready, instrument_id)
            # Execute first non-conflicting group in parallel
            batch = groups[0]

            if len(batch) == 1:
                # Single step — run directly (no thread overhead)
                result_holder: dict[str, Any] = {}
                _execute_step(
                    adapter=adapter,
                    run_id=run_id,
                    step=batch[0],
                    policy=policy,
                    instrument_id=instrument_id,
                    error_policy=error_policy,
                    result_holder=result_holder,
                    run_context=run_context,
                    recovery_policy=recovery_policy,
                    recovery_attempt_counts=recovery_attempt_counts,
                )
            else:
                # Multiple non-conflicting steps — run in parallel threads
                result_holder = {}
                threads: list[threading.Thread] = []
                for step in batch:
                    t = threading.Thread(
                        target=_execute_step,
                        kwargs={
                            "adapter": adapter,
                            "run_id": run_id,
                            "step": step,
                            "policy": policy,
                            "instrument_id": instrument_id,
                            "error_policy": error_policy,
                            "result_holder": result_holder,
                            "run_context": run_context,
                            "recovery_policy": recovery_policy,
                            "recovery_attempt_counts": recovery_attempt_counts,
                        },
                        name=f"step-{step['step_key']}",
                        daemon=True,
                    )
                    threads.append(t)

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            # Process results
            any_critical_failure = False
            for step in batch:
                sk = step["step_key"]
                outcome = result_holder.get(sk, {"ok": False, "error": "no result"})
                if outcome["ok"]:
                    finished_step_keys.add(sk)
                else:
                    failed_step_keys.add(sk)
                    any_critical_failure = True

            if any_critical_failure:
                first_error = next(
                    (result_holder[sk]["error"] for sk in failed_step_keys if result_holder.get(sk, {}).get("error")),
                    "unknown error",
                )
                worker_complete_run(
                    run_id=run_id,
                    final_status=RUN_STATUS_FAILED,
                    actor="worker",
                    reason=first_error,
                )
                worker_close_instrument_session(
                    run_id=run_id,
                    session_id=session_id,
                    status="failed",
                )
                return 1

        worker_complete_run(
            run_id=run_id,
            final_status=RUN_STATUS_SUCCEEDED,
            actor="worker",
        )
        worker_close_instrument_session(
            run_id=run_id,
            session_id=session_id,
            status="succeeded",
        )
        return 0
    except Exception as exc:
        worker_complete_run(
            run_id=run_id,
            final_status=RUN_STATUS_FAILED,
            actor="worker",
            reason=str(exc),
        )
        worker_close_instrument_session(
            run_id=run_id,
            session_id=session_id,
            status="failed",
        )
        return 1
    finally:
        adapter.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated orchestrator worker")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    try:
        return execute_run(args.run_id)
    except DomainError as exc:
        print(f"worker domain error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

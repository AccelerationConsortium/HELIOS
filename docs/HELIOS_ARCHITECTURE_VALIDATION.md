# HELIOS Architecture Validation

This evidence pack summarizes the current HELIOS architecture and validation boundary.

## Architecture

- HELIOS is agent-native, hierarchical, graph/state-machine routed, and role-based.
- The campaign controller coordinates specialist roles through typed state, traces, outcomes, and safety/proposal contracts.
- The live controller remains rule-based and auditable by default.

## Optimization And Backend Boundaries

- Nexus is an advisor/backend/evidence source, not campaign decision authority.
- BO MCP is a backend/tool, not the top-level controller.
- Backend failures are represented through typed failure attribution and deterministic fallback paths.
- Default BO MCP/Nexus/backend behavior is not changed by the validation-report layer.

## Learned Policy And Self-Evolution Boundaries

- Learned policies are offline/shadow/canary/gated, not default live overrides.
- Learned policies cannot hard-veto, add backends, override action/objective decisions, or auto-apply space revisions.
- Self-evolution is human/config-approved and proposal/approval gated.
- Self-evolution metadata does not alter default live behavior.

## Offline Closed-Loop SDL Evidence

- Offline no-human closed-loop SDL tests cover simulated observations, simulated execution, mocked backends, safety checks, outcomes, replay records, typed failure attribution, and default behavior invariance.
- Offline scenario benchmarks cover tiny-data baseline, high-noise measurement drift, constraint-heavy campaign behavior, backend-unavailable fallback, scientific negative evidence handling, plateau-to-pivot, mechanism validation, generalization transfer, execution instability recovery, and legacy fallback behavior.

## Safety Boundary

- Space revisions are approval-only and are not auto-applied.
- Policy evolution workflow is not connected to `rank_backends` or live `strategy_selector` execution.
- Safety gates and approval requirements are not weakened by reporting or metadata.
- Hardware and measurement failures do not directly penalize optimizer backends.
- `scientific_negative` is evidence, not an optimizer backend failure.

## Remaining Limitations

- Offline tests validate deterministic simulated campaign behavior, not paper-level proof of real-world superiority.
- Real campaign data, shadow/canary outcomes, reward correlation, failure-rate comparison, ablations, and external benchmarks remain required for research-grade claims.
- Report summaries are evidence metadata and do not activate learned policies or self-evolution workflows.

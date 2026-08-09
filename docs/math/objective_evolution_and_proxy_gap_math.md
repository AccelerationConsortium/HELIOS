# Objective Evolution and Proxy-Gap Mathematics for Scientific Campaign Control

**Status:** Technical note for the current implementation and intended design
**Scope:** objective hierarchy, proxy-gap assessment, objective-state revision, campaign-mode routing
**Primary code:** `objective_models.py`, `objective_stack.py`, `objective_state.py`, `campaign_mode.py`, `decision_layer.py`, `strategy_selector.py`

## 1. Scientific campaign control is not fixed-objective optimization

HELIOS should be read as a scientific campaign controller. It may use Bayesian
optimization, DoE, active learning, validation, calibration, or human/literature
context, but those are methods. The controller owns the campaign state.

The central mathematical distinction is:

```text
scientific question != active optimization proxy.
```

A fixed-objective optimizer assumes the objective function is already correct:

```text
choose x_t to maximize f(x).
```

A scientific campaign controller must ask a larger question:

```text
given the current evidence, which objective should the campaign trust,
which proxy should it optimize, and when should it validate or revise that proxy?
```

HELIOS already encodes this distinction in three layers:

- `ObjectiveStack`: static relationship among metrics and scientific goal.
- `ObjectiveState`: evolving belief about the active objective and its proxy gap.
- `CampaignMode`: the per-round activity decision, including validation,
  calibration, diagnosis, and optimization.

## 2. Objective stack

Let the campaign have a set of metrics:

```text
M = {m_1, ..., m_K}.
```

Each metric has:

```text
m_i = (
  name_i,
  level_i,
  direction_i,
  weight_i,
  target_i,
  current_value_i,
  uncertainty_i,
  proxy_risk_i,
  functional_relevance_i
).
```

The implemented levels are:

```text
raw_measurement
material_property
functional_proxy
device_performance
campaign_goal
```

The active metric set is:

```text
A = {m_i : name_i in active_metric_names}.
```

The validation metric set is:

```text
V = {m_i : name_i in validation_metric_names}.
```

This graph lets HELIOS represent common scientific situations:

- a raw sensor value is easy to measure but far from the final scientific goal
- a material property is closer, but still a proxy
- a device-level metric is expensive but more trustworthy
- a campaign goal may combine several functional requirements

The optimizer can operate on an active proxy while the controller tracks how
far that proxy sits from the true campaign goal.

## 3. Proxy-gap score

The proxy-gap analyzer assigns each active metric a base gap by level:

| Metric level | Base gap |
|---|---:|
| `campaign_goal` | `0.0` |
| `device_performance` | `0.1` |
| `functional_proxy` | `0.35` |
| `material_property` | `0.65` |
| `raw_measurement` | `0.8` |

For active metric `m_i`, the score is:

```text
g_i = clamp_01(
  base(level_i)
  + 0.25 * proxy_risk_i
  - 0.25 * functional_relevance_i
).
```

where:

```text
clamp_01(x) = max(0, min(1, x)).
```

The stack-level proxy-gap score is a weighted average:

```text
G = sum_{i in A} weight_i * g_i / sum_{i in A} weight_i.
```

If the active weights sum to zero, HELIOS uses an unweighted average:

```text
G = (1 / |A|) * sum_{i in A} g_i.
```

The severity level is:

```text
LOW     if G < 0.25
MEDIUM  if 0.25 <= G < 0.6
HIGH    if G >= 0.6
UNKNOWN if the objective stack cannot be resolved.
```

This score is interpretable. A raw measurement with high proxy risk and low
functional relevance produces a high gap. A device-performance metric with low
proxy risk produces a low gap. The formula is intentionally simple because the
controller records every component in evidence.

## 4. Objective state

`ObjectiveState` is the evolving belief state for the campaign objective:

```text
O_t = (
  primary_objective,
  scientific_question,
  proxy_objective_names,
  objective_confidence_t,
  proxy_gap_t,
  failure_constraints_t,
  validation_requirements_t,
  stopping_criteria,
  revision_t,
  rounds_observed_t,
  consecutive_failure_count_t,
  revision_history_t
).
```

This state separates:

- the scientific question, which should remain stable across rounds
- the proxy metrics used for optimization
- the controller's confidence that the objective is still valid
- the evidence that triggered objective revisions

The model does not need a fixed objective. It needs a versioned objective state.

## 5. Objective-confidence update

After a round, HELIOS can bind a `CampaignDecisionTrace` to a
`CampaignDecisionOutcome`. The objective-state updater computes a confidence
delta from observable outcome terms.

Let:

```text
e_t = execution_success
d_t = objective_delta
p_t = proxy_gap_delta
v_t = validation_success
f_t = failure_count
```

The implemented confidence delta is:

```text
Delta_conf_t =
  execution_term(e_t)
  + objective_term(d_t)
  + proxy_gap_term(p_t)
  + validation_term(v_t)
  + failure_term(f_t).
```

with:

```text
execution_term(True)  =  0.1
execution_term(False) = -0.1
execution_term(None)  =  0.0

objective_term(d) = 0.2 * clamp_signed(d)

proxy_gap_term(p) = -0.2 * clamp_signed(p)

validation_term(True)  =  0.1
validation_term(False) = -0.1
validation_term(None)  =  0.0

failure_term(f) = -0.05 * f.
```

`clamp_signed` bounds a scalar contribution to a stable finite range before
scaling. The sign convention for proxy gap matters:

```text
p_t < 0 means the proxy gap shrank, so confidence increases.
p_t > 0 means the proxy gap widened, so confidence decreases.
```

The updated confidence is:

```text
confidence_{t+1} = clamp_01(confidence_t + Delta_conf_t).
```

Every update produces an `ObjectiveRevision` with:

- old and new confidence
- old and new proxy-gap state when supplied
- old and new stopping recommendation
- evidence strings for each term
- a trace id linking the revision to the decision outcome

This makes objective evolution replayable.

## 6. Stopping criteria

An objective may carry deterministic stopping criteria:

```text
max_rounds
target_confidence
max_consecutive_failures
```

The updater computes:

```text
stop_recommended_{t+1} =
  (rounds_observed_{t+1} >= max_rounds)
  or (confidence_{t+1} >= target_confidence)
  or (consecutive_failure_count_{t+1} >= max_consecutive_failures).
```

The stop signal belongs to objective state. It does not execute a stop by
itself. The campaign mode and decision layers must consume it.

## 7. Campaign mode as objective-aware control

The campaign mode table is a priority-ordered policy:

```text
mu: (ObjectiveState, FailureAttribution, safety_summary, literature_missing)
    -> CampaignMode.
```

The current order is:

| Rank | Mode | Trigger |
|---:|---|---|
| 1 | `STOP_RECOMMENDED` | objective state recommends stop |
| 2 | `SAFETY_CONSTRAINT_TIGHTENING` | high safety risk |
| 3 | `HUMAN_OBSERVATION_REQUEST` | failure attribution confidence below 0.5 |
| 4 | `CALIBRATION` | confident instrument failure |
| 5 | `FAILURE_DIAGNOSIS` | other confident failure |
| 6 | `VALIDATION` | proxy gap is HIGH |
| 7 | `LITERATURE_CONTEXT_SEEKING` | external context missing |
| 8 | `BO_OPTIMIZATION` | default |

This table matters because it places objective integrity above blind
optimization. If the proxy gap is high, the controller chooses validation before
more BO. If safety risk is high, it tightens constraints before validation or
optimization.

## 8. Objective transition proposal

The dynamic strategy trace can carry an `ObjectiveTransitionProposal`:

```text
tau_t = (from_level, to_level, reason, evidence, confidence, auto_applied).
```

The current mapping is:

```text
DISCOVER  -> baseline
OPTIMIZE  -> performance
VALIDATE  -> mechanism
TRANSFER  -> generalization
PIVOT     -> generalization
```

The current implementation sets:

```text
auto_applied = False.
```

This is the correct default for a scientific campaign controller. A controller
can propose an objective transition, but live objective mutation should pass
through an approval gate or a bounded canary mode.

## 9. How objective level affects strategy selection

The strategy selector adds an objective-level prior to action utility:

```text
U'(a) = U(a) + prior(a, objective_level).
```

Current priors:

| Objective level | Prior behavior |
|---|---|
| `feasibility` | favor `stabilize` |
| `data_quality` | favor `stabilize` |
| `baseline` | favor `explore` |
| `performance` | favor `exploit`, weakly favor `refine` |
| `mechanism` | favor `stabilize` and `refine` |
| `generalization` | favor `explore` and `refine` |

This converts the same observed KPI history into different choices under
different scientific intents. A plateau during mechanism validation should not
mean the same thing as a plateau during performance optimization.

## 10. Decision-layer objective revision

The decision layer can inspect objective summaries. If it detects a high proxy
gap, it returns:

```text
CampaignDecisionAction.REVISE_OBJECTIVE
```

with an `ObjectivePatch` containing the proxy-gap assessment. This path is
shadow-only in the current code. It is a proposal to revise the objective, not
an automatic mutation of the live campaign.

The same layer preserves higher-priority gates:

```text
stop > blocking failure > safety > validation > objective conflict
  > proxy gap > context seeking > failure attribution > propose candidates.
```

That ordering prevents an objective revision rule from overriding a safety or
failure recovery requirement.

## 11. Mathematical invariants

A scientific campaign controller should satisfy these invariants:

### 11.1 Scientific objective identity

The scientific question should not be overwritten by a proxy metric.

```text
scientific_question_t = scientific_question_0
```

unless a human or explicit campaign-level policy revises it.

### 11.2 Proxy-objective versioning

If the active proxy changes, the controller must create a revision record:

```text
active_proxy_t != active_proxy_{t-1}
  => revision_history_t includes change record.
```

### 11.3 Proxy-gap monotonicity is not assumed

HELIOS must allow:

```text
proxy_gap_{t+1} > proxy_gap_t.
```

Scientific campaigns can learn that a proxy was misleading. The controller
should route that evidence to validation or objective revision, not hide it.

### 11.4 Validation outranks exploitation under high proxy gap

If:

```text
proxy_gap.level = HIGH
```

then the campaign mode table should select validation unless a higher-priority
stop, safety, or failure rule applies.

### 11.5 Objective transition is advisory until approved

The current safe invariant is:

```text
ObjectiveTransitionProposal.auto_applied = False.
```

Promotion from proposal to live objective change should require an explicit
approval gate, a feature flag, and replay evidence.

## 12. Current gaps

The current code supports objective evolution as a shadow and traceable design,
but the live campaign still starts from a single `objective_kpi` in many paths.
The orchestrator builds a default performance-level objective from input:

```text
objective_hierarchy = [
  { level: performance, metric: objective_kpi, direction, target }
]
```

That is a useful fallback, not the final controller model.

The main missing pieces are:

- A first-class objective stack in the campaign contract.
- A live `ObjectiveState` field in campaign persistence.
- A real proxy-gap source on live rounds.
- Consumption of `objective_transitions_json` by the next-round context builder.
- An approval path that can apply an objective transition after validation.

### Evidence-backed confidence path

The additive confidence update above remains the backward-compatible
`heuristic_outcome_delta` routing path. HELIOS now also supports a distinct
`scientific_evidence_posterior` path in `scientific_evidence.py` and
`objective_state.py`:

```text
logit confidence_{t+1}
  = logit prior_confidence + sum_i log Bayes factor_i.
```

Only independent evidence blocks with an auditable analysis method enter this
sum. Descriptive evidence is recorded without changing confidence, duplicate
independence keys are rejected, and operational execution success cannot alter
an objective once it is bound to the evidence-posterior path. Promotion remains
shadow-only and requires predeclared evidence/design gates plus explicit human
approval; it never applies an objective transition automatically.

## 13. Recommended implementation path

The next implementation should keep the current safety boundary:

1. Add `ObjectiveStack` and `ObjectiveState` to the campaign contract or
   campaign context.
2. Persist `ObjectiveState` as versioned campaign state.
3. Thread proxy-gap assessments into the adaptive substrate and decision layer.
4. Convert `ObjectiveTransitionProposal` into an approval-gated mutation.
5. Feed approved transitions into `CampaignSnapshot.campaign_context`.
6. Add replay tests that compare fixed-objective behavior against
   objective-aware campaign control.

The mathematical target is simple:

```text
HELIOS should optimize a proxy only while evidence supports that proxy.
```

When evidence weakens that support, HELIOS should validate, diagnose, revise,
or generalize. That is the difference between a fixed-objective optimizer and a
scientific campaign controller.

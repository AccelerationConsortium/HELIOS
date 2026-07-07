# Reward Accounting, Replay, and Policy Evaluation Mathematics in HELIOS

**Status:** Technical note for the current implementation
**Scope:** decision trace, observed outcome, deterministic reward, replay summary, bounded policy evaluation
**Primary code:** `decision_trace.py`, `decision_outcome.py`, `verifiable_reward.py`, `decision_replay.py`, `policy_evaluation.py`, `learned_policy.py`

## 1. Why HELIOS needs reward math

HELIOS cannot improve a campaign policy from traces unless each decision has a
stable target. The target cannot be a single KPI improvement because scientific
campaigns care about safety, failures, validation, proxy-gap reduction, context
gathering, and execution success.

The current reward layer binds three objects:

```text
CampaignDecisionTrace
CampaignDecisionOutcome
CampaignDecisionReward
```

into:

```text
CampaignDecisionAccounting = (trace, outcome, reward).
```

This accounting record is replayable. It carries enough information to ask:

- Did the proposed action match the runtime action?
- Did execution succeed?
- Did the objective improve?
- Did the proxy gap shrink?
- Did validation succeed?
- Did the campaign avoid safety incidents and repeated failures?
- Did a requested context action get fulfilled?

## 2. Trace, outcome, reward

Let a shadow decision trace at round `t` be:

```text
T_t = (
  campaign_id,
  round_index,
  context,
  decision_plan,
  actual_action,
  shadow_action,
  would_change_route,
  evidence
).
```

Let the observed post-decision outcome be:

```text
O_t = (
  execution_success,
  failure_count,
  safety_incident_count,
  objective_delta,
  proxy_gap_delta,
  validation_success,
  context_request_fulfilled,
  human_override
).
```

The reward calculator maps:

```text
R: O_t -> (reward_t, regret_t, component_scores_t, verifications_t).
```

The calculator is deterministic. It has no database writes, external calls,
training side effects, or promotion gates.

## 3. Component scores

The raw reward is a sum of verifiable component scores:

```text
r_raw =
  r_execution
  + r_failure
  + r_safety
  + r_objective
  + r_proxy_gap
  + r_validation
  + r_context.
```

### 3.1 Execution

```text
r_execution =
  0.2   if execution_success is True
 -0.3   if execution_success is False
  0.0   if execution_success is None.
```

Execution failure receives a larger magnitude than execution success because a
scientific campaign controller must protect the runtime path.

### 3.2 Failure count

```text
r_failure = -0.1 * failure_count.
```

The implementation rounds the component to ten decimal places. This prevents
floating-point artifacts such as `-0.30000000000000004` from leaking into exact
serialization and equality tests.

### 3.3 Safety incidents

```text
r_safety = -0.5 * safety_incident_count.
```

Safety dominates ordinary objective reward. One safety incident has larger
penalty magnitude than a unit objective improvement reward.

### 3.4 Objective improvement

```text
r_objective =
  0.0                         if objective_delta is None
  0.3 * min(objective_delta, 1) if objective_delta > 0 and positive_clamp=True
  0.3 * objective_delta        otherwise.
```

The decision-outcome calculator uses `positive_clamp=True`. The loop layer can
use raw scaling for historical compatibility. Both variants share one reward
core.

### 3.5 Proxy-gap reduction

```text
r_proxy_gap =
  0.0                 if proxy_gap_delta is None
  0.3 * abs(delta)    if proxy_gap_delta < 0
 -0.3 * delta         if proxy_gap_delta >= 0.
```

The sign convention is:

```text
proxy_gap_delta < 0  means the proxy gap shrank.
proxy_gap_delta > 0  means the proxy gap widened.
```

This term lets HELIOS reward a decision that improves scientific validity even
when it does not immediately improve the active KPI.

### 3.6 Validation

```text
r_validation =
  0.2   if validation_success is True
 -0.2   if validation_success is False
  0.0   if validation_success is None.
```

Validation reward is symmetric because validation is a test of whether the
campaign can trust the active proxy or mechanism claim.

### 3.7 Context fulfillment

```text
r_context =
  0.1   if context_request_fulfilled is True
  0.0   otherwise.
```

The implementation does not penalize unobserved context fulfillment. A missing
context signal is not treated as failure because many rounds do not request
context.

## 4. Clamping and regret

The public reward is clamped:

```text
reward = clamp(r_raw, -1, 1).
```

where:

```text
clamp(x, -1, 1) = max(-1, min(1, x)).
```

Regret is:

```text
regret = max(0, -reward).
```

This regret is not classical BO simple regret. It is a decision-quality regret
under the campaign reward rubric. A negative reward means the decision harmed
the campaign under the current accounting model.

## 5. Verifiable reward records

Each component has a verification record:

```text
v_i = (
  name_i,
  passed_i,
  score_i,
  verifier_type_i,
  evidence_i
).
```

The `passed` field is tri-state:

```text
True   means the desirable condition occurred.
False  means the undesirable condition occurred.
None   means the signal was unobserved or not applicable.
```

Examples:

```text
verify_execution(True)
  -> passed=True, score=0.2, evidence={execution_success: True}

verify_proxy_gap(-0.4)
  -> passed=True, score=0.12, evidence={proxy_gap_delta: -0.4}

verify_context(None)
  -> passed=None, score=0.0, evidence={context_request_fulfilled: None}
```

A verifier exception becomes:

```text
passed=None, score=0.0, evidence={error: "..."}
```

The system records the failure instead of hiding it.

## 6. Process reward and outcome reward

HELIOS splits raw reward into process and outcome terms:

```text
process_signals = {proxy_gap, context}
outcome_signals = {execution, objective, validation, failure, safety, recovery}
```

Then:

```text
process_reward = sum_{i in process_signals} score_i
outcome_reward = sum_{i in outcome_signals} score_i
```

and:

```text
process_reward + outcome_reward = r_raw
```

before clamping.

This split matters for learning. A decision may have weak immediate KPI gain
but strong process value, for example reducing a proxy gap or requesting
missing context. A campaign controller should be able to learn from those
decisions instead of treating them as failed optimization.

## 7. Replay summary

Given accounting records:

```text
A = {a_1, ..., a_N},
```

the replay analyzer computes:

```text
average_reward = (1 / N) * sum_i reward_i
positive_reward_count = |{i : reward_i > 0}|
negative_reward_count = |{i : reward_i < 0}|
neutral_reward_count  = |{i : reward_i = 0}|
```

It also computes route-change metrics:

```text
route_change_count = sum_i 1[trace_i.would_change_route]
route_change_rate = route_change_count / N.
```

Action distribution:

```text
action_distribution(a) = sum_i 1[decision_plan_i.action_type = a].
```

Average component rewards:

```text
avg_safety_penalty    = mean_i safety_penalty_i
avg_failure_penalty   = mean_i failure_penalty_i
avg_objective_reward  = mean_i objective_reward_i
avg_proxy_gap_reward  = mean_i proxy_gap_reward_i
avg_validation_reward = mean_i validation_reward_i
avg_context_reward    = mean_i context_reward_i.
```

Optional rates ignore unobserved values:

```text
rate(values) =
  (# True values among observed booleans) / (# observed booleans).
```

If no values are observed, the rate is `None`, not zero. This distinction keeps
the math honest: "we saw no validation outcomes" is not the same as "validation
never succeeded."

## 8. Policy evaluation variants

The policy evaluation runner compares bounded policy-influence variants on the
same replay snapshots:

```text
V = {
  baseline,
  action_policy_rerank,
  backend_memory_rerank,
  transition_guard_penalty,
  combined_safe_influence
}.
```

For each variant `v`, HELIOS reruns:

```text
trace_{i,v} = select_strategy(snapshot_i, policy_influence=v).strategy_trace.
```

It compares each variant to baseline:

```text
backend_changed_count_v =
  sum_i 1[selected_backend_{i,v} != selected_backend_{i,baseline}]

backend_changed_rate_v =
  backend_changed_count_v / N

reward_delta_vs_baseline_v =
  mean_reward_v - mean_reward_baseline.
```

This is counterfactual replay. It does not prove a policy will improve real
campaigns, but it filters unsafe or unstable policy influence before live use.

## 9. Ranking-change replay

For each baseline trace and variant trace, HELIOS extracts the ranked backend
list. It computes:

```text
top1_changed_i = ranked_baseline_i[0] != ranked_variant_i[0]
topk_changed_i = set(top_k_baseline_i) != set(top_k_variant_i).
```

A change is explained only if the variant trace contains a nonzero ranking
influence record:

```text
explained_i = exists influence record j such that abs(score_delta_j) > 0.
```

The replay summary counts unexplained ranking changes:

```text
unexplained_change_count =
  sum_i 1[(top1_changed_i or topk_changed_i) and not explained_i].
```

Unexplained changes are dangerous because the policy changed behavior without a
recorded causal mechanism.

## 10. Influence caps

Safe policy influence is bounded. For a ranking influence record:

```text
record = (source, target, raw_signal, applied_weight, score_delta, capped).
```

The safety check verifies:

```text
abs(score_delta) <= cap(source).
```

It also checks aggregate target-level deltas against a total cap:

```text
abs(sum_{records targeting b} score_delta) <= max_total_score_delta.
```

Caps prevent learned or bandit policy paths from silently overriding the rule
policy. A learned policy can nudge a backend within a compatible pool. It cannot
invent a backend, bypass safety, or hide the reason for a rank change.

## 11. Promotion logic implied by the math

The reward and replay layers do not promote policies by themselves. They supply
evidence for a promotion gate. A reasonable promotion gate should require:

```text
mean_reward_variant >= mean_reward_baseline - tolerance
unexplained_change_count = 0
cap_violation_count = 0
safety_warning_rate <= threshold
top1_changed_rate <= threshold
calibration_score >= threshold
```

HELIOS currently keeps learned and bandit influence conservative. That matches
the math: a policy that cannot explain its rank changes and stay within caps
should remain shadow-only.

## 12. Relationship to RLVR

The current reward core is compatible with a verifiable-reward interpretation:

```text
reward = f(verifications)
```

where each verification has:

```text
(passed, score, verifier_type, evidence).
```

This is stronger than an opaque scalar reward. A future offline RL or
meta-policy learner can train on:

```text
(state_features, context_features, action, backend, reward, verifications).
```

A reviewer can then inspect which part of the reward changed: objective,
proxy-gap, safety, validation, failure, execution, or context.

## 13. Mathematical limitations

The current reward is a hand-built scalarization. That has three limits:

1. Coefficients encode policy values. For example, one safety incident costs
   more than a unit positive objective delta. This is intentional, but it is a
   design choice.
2. Component additivity assumes no interaction terms. In reality, validation
   success after a safety incident may deserve a different interpretation than
   validation success in a clean round.
3. Reward does not estimate long-horizon scientific value unless process terms
   such as proxy-gap reduction and context fulfillment capture it.

These limits are acceptable for a first deterministic reward layer because the
system records all components. Replay can later test whether different weights
better predict long-run campaign success.

## 14. Recommended next mathematical checks

A math-heavy validation document should test:

```text
1. Reward sensitivity:
   vary each coefficient and measure policy-ranking stability.

2. Calibration:
   compare predicted strategy confidence with realized reward.

3. Ablation:
   remove proxy-gap, failure, safety, context, and validation terms one at a
   time and measure replay degradation.

4. Counterfactual safety:
   verify that no policy variant changes top-1 backend without a recorded,
   bounded influence record.

5. Long-horizon consistency:
   check whether process reward predicts later outcome reward.
```

The reward model should earn its place through replay evidence. Until that
evidence exists, HELIOS should keep policy learning behind shadow, canary, and
approval gates.

# Scientific Evidence Loop

## Purpose

HELIOS separates three quantities that are easy to conflate:

1. operational success — an experiment executed without hardware or workflow failure;
2. optimization progress — a configured KPI improved;
3. scientific evidence — an observation discriminated a falsifiable claim from a declared alternative.

Only the third quantity may update a scientific claim posterior. The implementation is pure, deterministic, shadow-only, and incapable of mutating a live objective, constraint, search space, or hardware route.

## Core modules

| Module | Responsibility |
|---|---|
| `app/services/scientific_evidence.py` | Typed claims, evidence, posterior-odds updates, evidence requirements, and promotion gates |
| `app/services/hypothesis_experiment_planner.py` | Prior-sensitive expected-information-gain scoring of experiments that distinguish competing hypotheses |
| `app/services/objective_state.py` | Binds an evidence posterior to objective state without allowing later operational outcomes to corrupt it |
| `app/services/scientific_ledger.py` | Writes reviewable claim assessments and experiment plans under the campaign evidence tree |

## Claim posterior

For claim `H`, prior probability `p(H)`, and independent evidence blocks `D_i`, HELIOS accepts an externally audited log Bayes factor for each block:

```text
log_BF_i = log p(D_i | H) - log p(D_i | not H)
```

The update is:

```text
logit p(H | D) = logit p(H) + sum_i log_BF_i
```

HELIOS does not infer a Bayes factor from generic success, objective delta, an LLM narrative, or an arbitrary evidence grade. An evidence item without a log Bayes factor is stored as descriptive evidence and does not move the posterior.

Every item has an `independence_key`. Duplicate keys are rejected so repeated summaries of the same plate, batch, dataset, or analysis cannot be counted as independent support. Raw dependent observations should first be combined by the declared statistical model and submitted as one evidence item.

The claim also records a prior version and rationale. Once an `ObjectiveState` is bound to a claim posterior, changing that prior is rejected; a scientifically justified revision must create a new versioned claim rather than silently rewriting prior odds after observing results.

## Evidence status

A posterior threshold alone is insufficient. `EvidenceAssessmentPolicy` can require:

- a minimum number of scored evidence blocks;
- prospective evidence;
- distinct experimental blocks;
- at least one interventional or independent-replication result;
- no triggered predeclared falsifier;
- no unresolved block on the claim, such as unvalidated placeholder metadata.

The resulting status is `proposed`, `inconclusive`, `supported`, `refuted`, or `blocked`.

## Promotion gate

`PromotionCriteria` is stricter than claim assessment. It can require additional preregistration, independent blocks, interventional evidence, safety history, predictive calibration checks, and human approval.

```text
evidence_criteria_satisfied = true
human_approved = true
promotion_allowed = true
auto_applied = false
```

`auto_applied` is structurally forbidden. Promotion means that an authorized operator may review an objective or policy change; it is not the change itself.

## Hypothesis-discrimination planning

Each candidate experiment declares a categorical outcome likelihood under every competing hypothesis. For prior scenario `s`, the expected information gain is:

```text
EIG_s(experiment) = H(H | prior_s) - E_y[H(H | y, experiment, prior_s)]
```

Because EIG can be sensitive to a subjective prior, HELIOS computes it under multiple predeclared prior scenarios and uses:

```text
robust_EIG = min_s EIG_s
ranking_utility = robust_EIG / experiment_cost
```

Experiments without source-backed safety approval remain visible but are excluded from the actionable ranking. Every plan requires operator approval and remains shadow-only.

## Minimal example

```python
import math

from app.services.scientific_evidence import (
    EvidenceAssessmentPolicy,
    EvidenceDesign,
    EvidenceItem,
    EvidenceSet,
    ScientificClaim,
    assess_claim_evidence,
)

claim = ScientificClaim(
    claim_id="scalarization-cliff",
    statement="Hard scalarization destroys useful feasibility signal.",
    scope="multi-drug solubilization campaign",
    prior_probability=0.5,
    prior_rationale="Balanced prior registered before prospective validation.",
    falsifying_observations=[
        "No held-out calibration gain from an explicitly constrained model."
    ],
)

evidence = EvidenceSet(
    claim_id=claim.claim_id,
    items=[
        EvidenceItem(
            evidence_id="plate-a",
            claim_id=claim.claim_id,
            independence_key="plate-a",
            design=EvidenceDesign.PROSPECTIVE_INTERVENTIONAL,
            source="preregistered held-out comparison",
            log_bayes_factor=math.log(9),
            analysis_method="predictive likelihood ratio",
            dataset_hash="sha256:...",
            registered_before_observation=True,
            block_ids=["plate-a"],
        )
    ],
)

assessment = assess_claim_evidence(
    claim,
    evidence,
    policy=EvidenceAssessmentPolicy(
        min_prospective_evidence=1,
        min_independent_blocks=1,
        require_interventional_evidence=True,
    ),
)
```

## Objective-state boundary

`apply_evidence_to_objective_state()` replaces `objective_confidence` with the audited posterior and records `objective_confidence_method=scientific_evidence_posterior`. Once bound, ordinary execution success, failure counts, or objective deltas continue to update operational state but cannot numerically alter the scientific posterior.

The legacy `heuristic_outcome_delta` path remains available for backward compatibility. It is a routing heuristic, not a probability statement.

## Ledger artifacts

```text
data/scientific_ledger/campaigns/<campaign-id>/evidence/
  index.md
  claims/<claim-id>.md
  plans/<plan-id>.md
```

Claim artifacts record falsifiers, every evidence block, log Bayes factors, design quality, warnings, unmet requirements, and promotion state. Plan artifacts record robust and mean information gain, cost-normalized ranking, and safety exclusions.

## What this still does not prove

- A supplied likelihood model may be scientifically wrong; its method and dataset hash therefore remain part of the evidence record.
- Posterior probability is conditional on declared hypotheses, priors, likelihoods, and independence assumptions.
- An experiment plan is only as meaningful as its hypothesis-specific outcome predictions.
- Retrospective evidence cannot substitute for prospective independent validation.
- No LLM output is scientific evidence by itself.

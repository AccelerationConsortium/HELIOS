# Mathematics of Dynamic Strategy Selection in HELIOS

**Status:** Technical note for the current implementation
**Scope:** `CampaignSnapshot -> DiagnosticSignals -> PhasePosterior -> ActionCandidate -> BackendSelection -> StrategyTrace`
**Primary code:** `app/services/strategy_selector.py`, `strategy_diagnostics.py`, `strategy_scoring.py`, `strategy_actions.py`, `backend_selection.py`

## 1. Problem statement

HELIOS chooses a strategy once per campaign round. The strategy is not a free
text choice and it is not delegated to an LLM. The selector maps a typed state
snapshot into:

- a campaign-level action family: `explore`, `exploit`, `refine`, or `stabilize`
- an optimization backend inside the action-compatible backend pool
- a confidence score
- an evidence table
- a replayable `StrategyTrace`

The selector solves a constrained decision problem:

```text
given state S_t, choose action a_t and backend b_t
subject to availability, safety, failure history, and bounded policy influence.
```

The implemented policy is deterministic after the input snapshot is fixed. It
uses rule scores, softmax normalization, utility scoring, backend ranking, and
bounded ranking deltas. Learned and bandit paths can influence ranking only
when explicit gates allow them.

## 2. State and notation

Let round `t` have a campaign snapshot:

```text
S_t = (
  t,
  T,
  n_t,
  d,
  H_t,
  B_t,
  X_t,
  Y_t,
  q_t,
  F_t,
  C_t,
  A_t
)
```

where:

- `t` is the round number.
- `T` is the max round budget.
- `n_t` is the number of completed observations.
- `d` is the search-space dimensionality.
- `H_t = (y_1, ..., y_n)` is the KPI history.
- `B_t = {(x_i, y_i)}` is the last batch.
- `X_t = (x_1, ..., x_n)` is the full parameter history.
- `Y_t = (y_1, ..., y_n)` is the full objective history.
- `q_t` is the QC failure rate.
- `F_t` is typed failure history and backend failure counts.
- `C_t` is campaign context, including objective level and scientific context.
- `A_t` is the available-backend map.

The selector assumes that higher objective values are better after the local
orientation step. For a minimization task, diagnostics that depend on trend
orientation use `-y_i` internally where needed.

## 3. Feature extraction

The diagnostic map is:

```text
phi: S_t -> z_t
```

where `z_t` is a `DiagnosticSignals` record. The current implementation computes
the following components.

### 3.1 Space coverage

The selector uses observations per dimension as a crude coverage proxy:

```text
coverage_t = min(1, (n_t / max(d, 1)) / 10).
```

This says that about ten observations per dimension counts as full coverage for
the rule policy. It is not a geometric coverage measure over the hypercube. It
is a sample-complexity proxy.

### 3.2 Model uncertainty

When enough observations and batch points exist, HELIOS builds a bootstrap KNN
ensemble. For each bootstrap sample `k`, it predicts the KPI at each last-batch
point by averaging nearby observations. Let:

```text
\hat{y}_{j,k} = KNNPredictor_k(x_j)
```

for batch point `x_j`. The uncertainty estimate is the mean bootstrap standard
deviation:

```text
u_t = mean_j std_k(\hat{y}_{j,k}).
```

This is not a GP posterior variance. It is a cheap disagreement estimate that
works without optional BO dependencies.

### 3.3 Noise ratio

For each observed point `x_i`, HELIOS finds a small nearest-neighbor set
`N_k(i)` in parameter space. It compares local KPI variation with global KPI
variation:

```text
local_var_i = Var({y_i} union {y_j : j in N_k(i)})
noise_ratio_t = mean_i local_var_i / Var(Y_t).
```

If the global variance is numerically zero, the ratio is defined as zero. High
values mean that nearby points disagree almost as much as arbitrary points, so
measurement noise or process instability dominates the response surface.

### 3.4 Replicate need

The selector combines available noise indicators:

```text
replicate_need_t =
  weighted_mean(
    min(1, noise_ratio_t),
    min(1, batch_cv_t / 0.3),
    min(1, qc_fail_rate_t / 0.3)
  )
```

with weights `0.4`, `0.3`, and `0.3` when all three components exist.
Missing components are skipped and the remaining weights are renormalized.

### 3.5 Batch coefficient of variation

For a last batch with at least two KPI values:

```text
batch_cv_t = std(B_t.y) / abs(mean(B_t.y)).
```

If the batch mean is near zero, the CV is left unavailable. The selector treats
unavailable as "no evidence", not as zero.

### 3.6 Improvement velocity

HELIOS computes a rolling improvement rate over oriented KPI history. Abstractly:

```text
velocity_t = rolling_improvement_rate(oriented(Y_t), window=5).
```

Positive velocity supports exploitation. Near-zero velocity supports refinement
when other plateau signals agree.

### 3.7 EI decay proxy

The selector computes an expected-improvement decay proxy from recent observed
improvement relative to earlier improvement:

```text
ei_decay_t ~= recent_improvement / historical_improvement.
```

Small values mean that the current acquisition surface no longer finds much
gain. The implementation treats low EI decay as evidence for `refine`.

### 3.8 Convergence status

The convergence detector returns:

```text
(status_t, confidence_t) in
  {improving, plateau, diverging, insufficient_data} x [0, 1].
```

The strategy selector does not collapse convergence into a stop decision. It
uses it as one signal among uncertainty, noise, coverage, and campaign context.

### 3.9 Local smoothness and batch spread

Local smoothness is a KNN consistency score. Nearby points should have similar
KPI rankings on a smooth landscape. Low smoothness suggests rugged or
multimodal structure, unless high noise explains the same behavior.

Batch spread estimates diversity in the last proposed batch. Low spread can
support refinement or plateau diagnoses; high spread can support exploration
or multimodal search.

### 3.10 Drift

The drift score compares recent observations with historical observations. In
the current implementation it is a bounded heuristic based on changes in recent
and older KPI distributions. A high drift score demotes exploitation.

## 4. Phase posterior

HELIOS maps diagnostics into four unnormalized phase scores:

```text
s_t = (s_E, s_X, s_R, s_S)
```

where:

- `E` is explore.
- `X` is exploit.
- `R` is refine.
- `S` is stabilize.

The current score table is explicit:

```text
if n_t < min_obs_for_exploitation:
    s_E += 3.0
if coverage_t < min_coverage_for_exploitation:
    s_E += 2.0
if model_uncertainty_t > 0.3:
    s_E += 1.5

if noise_ratio_t > noise_ratio_high:
    s_S += 2.0
if replicate_need_t > replicate_need_threshold:
    s_S += 1.5
if qc_fail_rate_t > 0.2:
    s_S += 1.0

if status_t == plateau and convergence_confidence_t > 0.5:
    s_R += 2.0 * convergence_confidence_t
if ei_decay_t < ei_decay_threshold:
    s_R += 1.5
if abs(velocity_t) < stall_velocity_threshold:
    s_R += 1.0
if batch_cv_t < batch_cv_convergence:
    s_R += 0.8

if status_t == improving and convergence_confidence_t > 0.4:
    s_X += 2.5 * convergence_confidence_t
if ei_decay_t > 0.3:
    s_X += 1.0
if batch_cv_t > 0.2:
    s_X += 0.8

if status_t == diverging and convergence_confidence_t > 0.5:
    s_E += 2.5

if n_t >= min_obs_for_exploitation and max(s_t) < 1.0:
    s_X += 1.0
```

The posterior is a softmax:

```text
p_i = exp(s_i - max_j s_j) / sum_k exp(s_k - max_j s_j).
```

The entropy is:

```text
H(p) = - sum_i p_i log(max(p_i, epsilon)).
```

The selector keeps entropy because a high top probability and high uncertainty
mean different things. High entropy signals that the controller does not have a
clean phase assignment.

## 5. Adaptive utility weights

Each action candidate has three predicted components:

```text
g_a = expected objective improvement
i_a = expected information gain
r_a = expected risk
```

The utility uses normalized weights:

```text
U(a | S_t) = w_imp * g_a + w_info * i_a - w_risk * r_a + prior(a, C_t).
```

The default weights are:

```text
w_imp = 0.45
w_info = 0.35
w_risk = 0.20
```

HELIOS adjusts them before scoring:

- High noise increases `w_risk` and `w_info`, and decreases `w_imp`.
- High posterior entropy increases `w_info`, and slightly decreases `w_imp`.
- High improvement velocity increases `w_imp`, and slightly decreases `w_info`.

After adjustment, the implementation clamps the weights to lower bounds and
renormalizes them:

```text
w_imp >= 0.1
w_info >= 0.1
w_risk >= 0.05
w_imp + w_info + w_risk = 1.
```

This keeps utility signs interpretable. Objective improvement and information
gain add value; risk subtracts value.

## 6. Action candidate models

The selector constructs four action candidates.

### 6.1 Explore

Explore uses a space-filling backend, usually LHS:

```text
g_explore = max(0, 1 - coverage_t)
i_explore = max(0.3, 1 - coverage_t) + uncertainty_bonus
r_explore = 0.1 + noise_penalty.
```

The mathematical intent is to buy coverage and information when the controller
does not trust the local model.

### 6.2 Exploit

Exploit uses the current best model-backed optimizer:

```text
g_exploit = base + improving_bonus + ei_decay_bonus + velocity_bonus
i_exploit = 0.3
r_exploit = 0.2 + uncertainty_penalty + noise_penalty.
```

If the surface looks multimodal and an evolutionary backend is available, the
candidate can switch to `pymoo_nsga2`. Otherwise the exploit backend comes from
the configured exploitation pool.

### 6.3 Refine

Refine targets local search or acquisition/batch adjustment after apparent
saturation:

```text
g_refine = 0.3 + plateau_bonus + low_ei_bonus
i_refine = 0.2
r_refine = 0.3 + noise_penalty.
```

The selected refinement backend depends on dimension, smoothness, and backend
availability.

### 6.4 Stabilize

Stabilize means repeat or verify points instead of proposing a new region:

```text
g_stabilize = 0.1
i_stabilize = 0.2 + high_noise_bonus + replicate_need_bonus
r_stabilize = 0.1.
```

If stabilize wins, `build_stabilize_spec()` creates a concrete replicate plan.
The replicate plan caps its cost by remaining budget:

```text
max_stabilize_budget = max(1, floor((T - t) * stabilize_budget_fraction)).
```

It prefers replicating top points or high-variance batch points, then trims
replicates and points until the plan fits the cap.

## 7. Objective-level priors

HELIOS treats the current scientific objective level as a prior over action
families:

```text
prior(a, level)
```

The implementation uses small additive shifts:

| Objective level | Preferred actions |
|---|---|
| `feasibility` | `stabilize` |
| `data_quality` | `stabilize` |
| `baseline` | `explore` |
| `performance` | `exploit`, then `refine` |
| `mechanism` | `stabilize`, `refine` |
| `generalization` | `explore`, `refine` |

This is the first place where the selector behaves like a campaign controller
instead of a fixed-objective optimizer. The same KPI trend can map to different
actions when the scientific intent changes.

## 8. Governance gates

The selector applies two major gates after action scoring.

### 8.1 Entropy gate

The entropy threshold depends on observations per dimension:

```text
obs_per_dim = n_t / max(d, 1).
```

Early campaigns use a lower threshold for exploitation. Mature campaigns use a
threshold close to the maximum entropy of a four-way distribution:

```text
H_max = log(4).
```

If entropy exceeds the adaptive threshold, the selector demotes exploit by
moving non-exploit actions ahead of exploit in the candidate order.

### 8.2 Drift gate

If:

```text
drift_score_t > drift_high_threshold,
```

the selector demotes exploit and favors stabilize or explore. This prevents a
stale model from exploiting a response surface that no longer resembles the
historical data.

## 9. Backend ranking

The best action chooses an action-compatible backend pool:

```text
P_a = (b_1, ..., b_m).
```

Backend ranking filters unavailable backends and vetoes backends whose failure
count reaches the threshold:

```text
f_b >= failure_veto_threshold.
```

For each backend that survives:

```text
phase_score_b = (m - index_P(b)) / m
fingerprint_boost_b =
  fingerprint_weight * (len(R) - rank_R(b)) / len(R), if b in R
  0, otherwise
failure_penalty_b =
  failure_penalty * min(1, f_b / failure_veto_threshold)
total_b =
  phase_weight * phase_score_b
  + fingerprint_boost_b
  - failure_penalty_b
  + influence_delta_b.
```

`R` is the tuple of recommended backends from Nexus, method advice, or
campaign-specific meta-learning. `influence_delta_b` is a bounded policy
influence term. It cannot add a backend to the pool. It can only adjust scores
inside the candidate pool, and the selector records every applied delta.

The selected backend is:

```text
b_t = argmax_b total_b,
```

with deterministic tie-breaking by the original pool order.

## 10. Evidence decomposition

The selector records evidence as signed contributions. Examples:

```text
low coverage -> explore contribution
high uncertainty -> explore contribution
high noise -> stabilize contribution
high noise -> exploit penalty
positive velocity -> exploit contribution
plateau -> refine contribution
low EI decay -> refine contribution
low smoothness -> explore contribution
high drift -> stabilize contribution
```

This evidence table is not a proof that the chosen action is optimal. It is an
audit trail for why the current rule policy chose the action.

## 11. Confidence

HELIOS computes confidence from four terms:

```text
confidence =
  0.15 * observation_confidence
  + 0.25 * signal_richness
  + 0.25 * convergence_confidence
  + 0.35 * phase_agreement.
```

where:

```text
observation_confidence = min(1, n_t / 20)
signal_richness = min(1, number_of_observed_diagnostic_signals / 8).
```

`phase_agreement` is high when the selected phase agrees with the strongest
diagnostic class, for example `exploit` with `improving`, `refine` with
`plateau`, `explore` with low coverage, or `stabilize` with high noise.

## 12. What this math is, and what it is not

This selector is a deterministic controller policy. It is not a Bayes-optimal
POMDP solution, and it does not claim that the phase posterior is a calibrated
posterior over latent scientific states. The posterior is a normalized rule
score that gives the controller an interpretable four-way uncertainty measure.

The design tradeoff is deliberate:

- The math is transparent enough to audit after each round.
- Every score can be replayed from a `CampaignSnapshot`.
- Optional learned policies remain bounded and gated.
- Hardware execution does not depend on an LLM or an opaque learned controller.

The natural next mathematical improvement is calibration. Replay data can test
whether `p_explore`, `p_exploit`, `p_refine`, and `p_stabilize` predict future
reward better than a simpler rule table. If they do not, the system should tune
or replace score increments using replay evidence, while keeping the same trace
contract.

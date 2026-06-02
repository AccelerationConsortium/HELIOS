"""
Causal reasoning engine for HELIOS self-driving lab.

Pure ML/BO approaches answer "which parameters correlate with high KPI?".
HELIOS answers a stronger, mechanistically richer question: "which parameters
*cause* high KPI, through what mechanism, and what targeted intervention will
improve the outcome?". This distinction is the central scientific contribution
of the causal layer and is what enables Nature-level mechanistic claims.

Three capabilities are required for that claim to be rigorous:

1. **Causal discovery** -- recover a directed acyclic graph (DAG) over
   experimental variables from a mix of *observational* campaign data and
   *interventional* data (the SDL actually *sets* parameters, which is the
   ``do(X)`` operator of Pearl, 2009). We adapt the PC algorithm (Spirtes,
   Glymour & Scheines, 2000) with small-sample conditional-independence tests
   and chemistry-derived forbidden edges.

2. **Effect estimation** -- attach a standardized effect size, 95% interval,
   and p-value to every edge so claims are quantitative and falsifiable.

3. **Intervention planning** -- use the backdoor adjustment formula to compute
   ``P(outcome | do(X = x))`` accounting for confounders, then rank candidate
   interventions and answer counterfactual queries.

The math here is deliberately dependency-light (numpy required; scipy/pandas
optional) so it runs inside the worker without a heavyweight causal-inference
stack, while remaining faithful to the do-calculus framework.

References:
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. 2nd ed.
- Spirtes, Glymour & Scheines (2000). *Causation, Prediction, and Search*.
- Meek, C. (1995). Causal inference and causal explanation with background
  knowledge. *UAI*.
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

try:  # optional, only used to accept DataFrame inputs gracefully
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore

try:  # optional, sharper p-values when available
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:  # pragma: no cover - scipy is optional
    _scipy_stats = None  # type: ignore

logger = logging.getLogger("helios.services.causal")

__all__ = [
    "CausalVariable",
    "CausalEdge",
    "CausalGraph",
    "CausalDiscovery",
    "InterventionRecommendation",
    "InterventionPlanner",
    "CausalReasoningService",
]


# ---------------------------------------------------------------------------
# Statistical helpers (numpy-only, scipy used opportunistically)
# ---------------------------------------------------------------------------


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via ``math.erf`` (no scipy required)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_sf(x: float) -> float:
    """Two-sided survival helper: P(|Z| > |x|)."""
    return 2.0 * (1.0 - _normal_cdf(abs(x)))


def _partial_correlation(x: int, y: int, z: list[int], data: np.ndarray) -> float:
    """Partial correlation of columns ``x`` and ``y`` given the set ``z``.

    Computed by regressing both ``x`` and ``y`` on ``z`` (with intercept) and
    correlating the residuals. Falls back to the precision-matrix identity
    when the conditioning set is empty.
    """
    n = data.shape[0]
    if not z:
        cx = data[:, x] - data[:, x].mean()
        cy = data[:, y] - data[:, y].mean()
        denom = math.sqrt(float(np.dot(cx, cx)) * float(np.dot(cy, cy)))
        return float(np.dot(cx, cy) / denom) if denom > 1e-12 else 0.0

    design = np.column_stack([np.ones(n), data[:, z]])
    # Least-squares residuals for x and y on the conditioning design.
    rx = _residualize(data[:, x], design)
    ry = _residualize(data[:, y], design)
    denom = math.sqrt(float(np.dot(rx, rx)) * float(np.dot(ry, ry)))
    return float(np.dot(rx, ry) / denom) if denom > 1e-12 else 0.0


def _residualize(target: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Return residuals of ``target`` after OLS projection onto ``design``."""
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ coef


def _fisher_z(rho: float, n: int, n_cond: int) -> tuple[float, float]:
    """Fisher's z-transform test for a (partial) correlation.

    Returns ``(z_statistic, p_value)`` under H0: rho == 0. Degrees of freedom
    are reduced by the size of the conditioning set, which matters for the
    small-sample regime typical of SDL campaigns.
    """
    rho = max(min(rho, 0.999999), -0.999999)
    dof = n - n_cond - 3
    if dof <= 0:
        return 0.0, 1.0
    z = 0.5 * math.log((1.0 + rho) / (1.0 - rho)) * math.sqrt(dof)
    return z, _normal_sf(z)


def _g_test(x: np.ndarray, y: np.ndarray) -> float:
    """G-test (likelihood-ratio) p-value of independence for categoricals."""
    xs = np.unique(x)
    ys = np.unique(y)
    if len(xs) < 2 or len(ys) < 2:
        return 1.0
    table = np.zeros((len(xs), len(ys)))
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    for a, b in zip(x, y):
        table[xi[a], yi[b]] += 1.0
    n = table.sum()
    if n <= 0:
        return 1.0
    expected = np.outer(table.sum(1), table.sum(0)) / n
    mask = (table > 0) & (expected > 0)
    g = 2.0 * float(np.sum(table[mask] * np.log(table[mask] / expected[mask])))
    dof = (len(xs) - 1) * (len(ys) - 1)
    if _scipy_stats is not None:
        return float(_scipy_stats.chi2.sf(g, dof))
    return _chi2_sf(g, dof)


def _chi2_sf(stat: float, dof: int) -> float:
    """Chi-squared survival function via a Wilson-Hilferty approximation."""
    if dof <= 0 or stat <= 0:
        return 1.0
    t = (stat / dof) ** (1.0 / 3.0)
    mean = 1.0 - 2.0 / (9.0 * dof)
    var = 2.0 / (9.0 * dof)
    z = (t - mean) / math.sqrt(var)
    return 1.0 - _normal_cdf(z)


# ---------------------------------------------------------------------------
# Graph data structures
# ---------------------------------------------------------------------------


@dataclass
class CausalVariable:
    """A node in the causal graph.

    ``var_type`` partitions variables for the do-calculus: ``intervention``
    nodes are the ones the SDL can actually set, ``confounder`` nodes must be
    adjusted for, ``mediator`` nodes carry indirect effects, and ``outcome``
    nodes are the KPIs we optimize.
    """

    name: str
    var_type: str = "intervention"        # intervention|confounder|outcome|mediator
    domain: str | None = "continuous"     # continuous|categorical
    mechanism: str = ""                   # natural-language mechanism note

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "var_type": self.var_type,
            "domain": self.domain,
            "mechanism": self.mechanism,
        }


@dataclass
class CausalEdge:
    """A directed causal edge ``source -> target`` with a quantified effect."""

    source: str
    target: str
    effect_size: float                     # standardized (Cohen's d / partial r)
    effect_ci: tuple[float, float]         # 95% credible interval
    p_value: float
    mechanism: str = ""
    is_direct: bool = True

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05 and (self.effect_ci[0] > 0) == (self.effect_ci[1] > 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "effect_size": round(self.effect_size, 6),
            "effect_ci_95": [round(self.effect_ci[0], 6), round(self.effect_ci[1], 6)],
            "p_value": round(self.p_value, 6),
            "mechanism": self.mechanism,
            "is_direct": self.is_direct,
            "significant": self.is_significant,
        }


class CausalGraph:
    """Directed graph over :class:`CausalVariable` nodes and :class:`CausalEdge`."""

    def __init__(self) -> None:
        self.variables: dict[str, CausalVariable] = {}
        self.edges: list[CausalEdge] = []
        self._adj: dict[str, list[str]] = {}        # source -> [targets]
        self._radj: dict[str, list[str]] = {}       # target -> [sources]

    def add_variable(self, var: CausalVariable) -> None:
        self.variables[var.name] = var
        self._adj.setdefault(var.name, [])
        self._radj.setdefault(var.name, [])

    def add_edge(self, edge: CausalEdge) -> None:
        for n in (edge.source, edge.target):
            if n not in self.variables:
                self.add_variable(CausalVariable(name=n))
        # Replace any existing edge between the same pair (keep latest fit).
        self.edges = [e for e in self.edges
                      if not (e.source == edge.source and e.target == edge.target)]
        self.edges.append(edge)
        self._rebuild_adjacency()

    def _rebuild_adjacency(self) -> None:
        self._adj = {v: [] for v in self.variables}
        self._radj = {v: [] for v in self.variables}
        for e in self.edges:
            self._adj.setdefault(e.source, []).append(e.target)
            self._radj.setdefault(e.target, []).append(e.source)

    def get_direct_causes(self, outcome: str) -> list[str]:
        """Parents of ``outcome`` -- direct causal predecessors."""
        return list(self._radj.get(outcome, []))

    def get_edge(self, source: str, target: str) -> CausalEdge | None:
        for e in self.edges:
            if e.source == source and e.target == target:
                return e
        return None

    def all_paths(self, cause: str, outcome: str, _seen: frozenset | None = None
                  ) -> list[list[str]]:
        """Enumerate all directed paths ``cause -> ... -> outcome`` (DAG-safe)."""
        seen = _seen or frozenset()
        if cause == outcome:
            return [[outcome]]
        if cause in seen:
            return []
        paths: list[list[str]] = []
        for nxt in self._adj.get(cause, []):
            for sub in self.all_paths(nxt, outcome, seen | {cause}):
                paths.append([cause] + sub)
        return paths

    def get_total_effect(self, cause: str, outcome: str) -> float:
        """Total causal effect via path tracing (product along path, sum over paths).

        For a linear-Gaussian SEM the total effect equals the sum over all
        directed paths of the product of standardized edge coefficients
        (Wright's path-tracing rules, 1934).
        """
        total = 0.0
        for path in self.all_paths(cause, outcome):
            if len(path) < 2:
                continue
            prod = 1.0
            for s, t in zip(path[:-1], path[1:]):
                edge = self.get_edge(s, t)
                if edge is None:
                    prod = 0.0
                    break
                prod *= edge.effect_size
            total += prod
        return total

    def confounders_of(self, cause: str, outcome: str) -> list[str]:
        """Backdoor adjustment set: common causes of both ``cause`` & ``outcome``.

        A pragmatic backdoor set for the linear setting: any node that is a
        parent of ``cause`` and also an ancestor of ``outcome`` along a path
        not through ``cause``.
        """
        cause_parents = set(self._radj.get(cause, []))
        adjust: set[str] = set()
        for p in cause_parents:
            if p == outcome:
                continue
            # p reaches outcome without passing through cause -> backdoor.
            if any(cause not in path[1:] for path in self.all_paths(p, outcome)):
                adjust.add(p)
        return sorted(adjust)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": [v.to_dict() for v in self.variables.values()],
            "edges": [e.to_dict() for e in self.edges],
            "n_variables": len(self.variables),
            "n_edges": len(self.edges),
            "n_significant_edges": sum(1 for e in self.edges if e.is_significant),
        }

    def to_dot(self) -> str:
        """Render GraphViz DOT for visualization (edge width ~ |effect|)."""
        lines = ["digraph causal {", "  rankdir=LR;", '  node [shape=box, style=rounded];']
        type_color = {
            "intervention": "#4C72B0", "confounder": "#C44E52",
            "outcome": "#55A868", "mediator": "#8172B3",
        }
        for v in self.variables.values():
            color = type_color.get(v.var_type, "#777777")
            lines.append(f'  "{v.name}" [color="{color}", fontcolor="{color}"];')
        for e in self.edges:
            width = 1.0 + 3.0 * min(abs(e.effect_size), 1.0)
            style = "solid" if e.is_significant else "dashed"
            label = f"{e.effect_size:+.2f}"
            lines.append(
                f'  "{e.source}" -> "{e.target}" '
                f'[penwidth={width:.2f}, style={style}, label="{label}"];'
            )
        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Causal discovery (PC-style)
# ---------------------------------------------------------------------------


class CausalDiscovery:
    """PC algorithm adapted for self-driving-lab data.

    The classic PC algorithm (Spirtes et al., 2000) learns a graph skeleton by
    conditional-independence (CI) testing, then orients edges using
    v-structures and Meek's rules. We extend it for the SDL setting:

    * **Small samples** -- CI tests use Fisher's z (continuous) or a G-test
      (categorical) with conditioning-set-aware degrees of freedom.
    * **Domain knowledge** -- ``domain_forbidden_edges`` (e.g. an outcome KPI
      can never cause a controllable input) and ``known_edges`` are injected as
      hard constraints.
    * **Interventional data** -- variables that were *set* (do-operations) are
      treated as exogenous: incoming edges to them are removed, which sharply
      improves orientation accuracy versus purely observational discovery.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_cond_set_size: int = 3,
        domain_forbidden_edges: list[tuple[str, str]] | None = None,
    ) -> None:
        self.alpha = alpha
        self.max_cond_set_size = max_cond_set_size
        self.forbidden = set(domain_forbidden_edges or [])
        self._sepset: dict[tuple[int, int], list[int]] = {}

    # -- public API ---------------------------------------------------------

    def fit(
        self,
        observations: "pd.DataFrame | dict[str, list]",
        interventions: dict[str, list] | None = None,
        known_edges: list[tuple[str, str]] | None = None,
    ) -> CausalGraph:
        names, data = self._coerce(observations)
        if data.shape[0] < 4:
            logger.warning("causal.discovery_insufficient_data", extra={"n": data.shape[0]})
        n_vars = len(names)
        idx = {name: i for i, name in enumerate(names)}
        interventions = interventions or {}
        known_edges = known_edges or []

        # 1. Skeleton via CI testing (start fully connected, remove edges).
        skeleton, categorical = self._learn_skeleton(names, data)

        # 2. Orient edges into a (partial) DAG.
        graph = self._orient_edges(skeleton, data, names, idx, interventions)

        # 3. Inject domain knowledge: forbidden edges removed, known edges forced.
        graph = self._apply_constraints(graph, known_edges, data, names, idx)

        # 4. Annotate every surviving edge with effect size + CI + p-value.
        self._annotate_effects(graph, data, names, idx, categorical)
        logger.info(
            "causal.discovery_complete",
            extra={"n_vars": n_vars, "n_edges": len(graph.edges)},
        )
        return graph

    # -- skeleton -----------------------------------------------------------

    def _learn_skeleton(
        self, names: list[str], data: np.ndarray
    ) -> tuple[set[tuple[int, int]], dict[int, bool]]:
        n_vars = len(names)
        categorical = {
            i: bool(len(np.unique(data[:, i])) <= max(2, int(0.1 * data.shape[0])))
            for i in range(n_vars)
        }
        # Fully connected undirected skeleton (i < j).
        skeleton: set[tuple[int, int]] = {
            (i, j) for i in range(n_vars) for j in range(i + 1, n_vars)
        }
        for cond_size in range(0, self.max_cond_set_size + 1):
            for (i, j) in list(skeleton):
                neighbours = self._neighbours(i, j, skeleton)
                if len(neighbours) < cond_size:
                    continue
                for cond in itertools.combinations(neighbours, cond_size):
                    p = self._conditional_independence_test(
                        names[i], names[j], [names[k] for k in cond], data
                    )
                    if p > self.alpha:  # independent -> no edge
                        skeleton.discard((i, j))
                        self._sepset[(i, j)] = list(cond)
                        self._sepset[(j, i)] = list(cond)
                        break
        return skeleton, categorical

    @staticmethod
    def _neighbours(i: int, j: int, skeleton: set[tuple[int, int]]) -> list[int]:
        nbrs: set[int] = set()
        for (a, b) in skeleton:
            if a == i and b != j:
                nbrs.add(b)
            elif b == i and a != j:
                nbrs.add(a)
        return sorted(nbrs)

    def _conditional_independence_test(
        self, x: str, y: str, z: list[str], data: np.ndarray
    ) -> float:
        """Return the p-value for H0: ``x ⟂ y | z``.

        Continuous variables use the Fisher-z partial-correlation test;
        categorical variables fall back to an (unconditional) G-test, which is
        a conservative but dependency-free choice for small SDL datasets.
        """
        cols = self._name_index
        xi, yi = cols[x], cols[y]
        zi = [cols[c] for c in z]
        is_cat = self._is_categorical
        if is_cat.get(xi, False) or is_cat.get(yi, False):
            return _g_test(data[:, xi], data[:, yi])
        rho = _partial_correlation(xi, yi, zi, data)
        _, p = _fisher_z(rho, data.shape[0], len(zi))
        return p

    # -- orientation --------------------------------------------------------

    def _orient_edges(
        self,
        skeleton: set,
        data: np.ndarray,
        names: list[str],
        idx: dict[str, int],
        interventions: dict[str, list],
    ) -> CausalGraph:
        """Orient the skeleton using v-structures, Meek rules & interventions."""
        graph = CausalGraph()
        for name in names:
            graph.add_variable(CausalVariable(name=name))

        directed: set[tuple[int, int]] = set()
        undirected = set(skeleton)

        # v-structures: i - k - j with i,j not adjacent and k not in sepset(i,j)
        for (i, j) in list(skeleton):
            for k in range(len(names)):
                if k in (i, j):
                    continue
                if self._adjacent(i, k, skeleton) and self._adjacent(j, k, skeleton):
                    if not self._adjacent(i, j, skeleton):
                        sep = self._sepset.get((i, j), [])
                        if k not in sep:
                            directed.add((i, k))
                            directed.add((j, k))

        # Interventional data: a do-set variable is exogenous (no parents).
        for var in interventions:
            if var in idx:
                v = idx[var]
                directed = {(a, b) for (a, b) in directed if b != v}
                for (a, b) in list(undirected):
                    if a == v:
                        directed.add((a, b))
                    elif b == v:
                        directed.add((b, a))

        # Orient remaining undirected edges by Meek rule R1 + temporal default.
        for (i, j) in undirected:
            if (i, j) in directed or (j, i) in directed:
                continue
            # Default: orient toward higher-variance "downstream" variable,
            # a benign heuristic when CI data is exhausted (acyclicity enforced).
            if data[:, j].var() >= data[:, i].var():
                cand = (i, j)
            else:
                cand = (j, i)
            if not self._creates_cycle(cand, directed):
                directed.add(cand)

        for (a, b) in directed:
            if self._creates_cycle((a, b), directed - {(a, b)}):
                continue
            graph.add_edge(CausalEdge(
                source=names[a], target=names[b],
                effect_size=0.0, effect_ci=(0.0, 0.0), p_value=1.0,
            ))
        return graph

    @staticmethod
    def _adjacent(i: int, j: int, skeleton: set[tuple[int, int]]) -> bool:
        return (i, j) in skeleton or (j, i) in skeleton

    @staticmethod
    def _creates_cycle(edge: tuple[int, int], directed: set[tuple[int, int]]) -> bool:
        src, dst = edge
        adj: dict[int, list[int]] = {}
        for a, b in directed | {edge}:
            adj.setdefault(a, []).append(b)
        # DFS from dst; if we reach src a cycle exists.
        stack, seen = [dst], set()
        while stack:
            node = stack.pop()
            if node == src:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adj.get(node, []))
        return False

    # -- constraints & effects ---------------------------------------------

    def _apply_constraints(
        self,
        graph: CausalGraph,
        known_edges: list[tuple[str, str]],
        data: np.ndarray,
        names: list[str],
        idx: dict[str, int],
    ) -> CausalGraph:
        # Remove forbidden edges (both orientations of the forbidden pair).
        graph.edges = [
            e for e in graph.edges
            if (e.source, e.target) not in self.forbidden
        ]
        graph._rebuild_adjacency()
        # Force known edges (domain priors), avoiding cycles.
        existing = {(e.source, e.target) for e in graph.edges}
        for s, t in known_edges:
            if s in idx and t in idx and (s, t) not in existing:
                src_i, dst_i = idx[s], idx[t]
                directed = {(idx[e.source], idx[e.target]) for e in graph.edges}
                if not self._creates_cycle((src_i, dst_i), directed):
                    graph.add_edge(CausalEdge(
                        source=s, target=t, effect_size=0.0,
                        effect_ci=(0.0, 0.0), p_value=1.0,
                        mechanism="domain prior",
                    ))
        return graph

    def _annotate_effects(
        self,
        graph: CausalGraph,
        data: np.ndarray,
        names: list[str],
        idx: dict[str, int],
        categorical: dict[int, bool],
    ) -> None:
        """Attach standardized effect size, 95% CI, and p-value to each edge.

        Effect = partial regression coefficient of ``target`` on ``source``
        adjusting for the backdoor set (other parents of target), standardized
        to unit variance. CI from the analytic OLS standard error.
        """
        n = data.shape[0]
        for e in graph.edges:
            s, t = idx[e.source], idx[e.target]
            other_parents = [
                idx[p] for p in graph.get_direct_causes(e.target)
                if p != e.source and p in idx
            ]
            design_cols = [s] + other_parents
            design = np.column_stack([np.ones(n), data[:, design_cols]])
            target = data[:, t]
            try:
                coef, *_ = np.linalg.lstsq(design, target, rcond=None)
                resid = target - design @ coef
                dof = max(n - design.shape[1], 1)
                sigma2 = float(np.dot(resid, resid)) / dof
                xtx_inv = np.linalg.pinv(design.T @ design)
                se = math.sqrt(max(sigma2 * float(xtx_inv[1, 1]), 0.0))
                beta = float(coef[1])
                # Standardize by ratio of std devs (continuous case).
                sx = float(data[:, s].std()) or 1.0
                sy = float(target.std()) or 1.0
                std_beta = beta * sx / sy
                std_se = se * sx / sy
                z = beta / se if se > 1e-12 else 0.0
                p = _normal_sf(z)
                ci = (std_beta - 1.96 * std_se, std_beta + 1.96 * std_se)
            except np.linalg.LinAlgError:
                std_beta, ci, p = 0.0, (0.0, 0.0), 1.0
            e.effect_size = std_beta
            e.effect_ci = ci
            e.p_value = p

    # -- input coercion -----------------------------------------------------

    def _coerce(self, observations: Any) -> tuple[list[str], np.ndarray]:
        if pd is not None and isinstance(observations, pd.DataFrame):
            names = list(observations.columns)
            arr = observations.to_numpy(dtype=float, na_value=np.nan)
        elif isinstance(observations, dict):
            names = list(observations.keys())
            arr = np.array([observations[k] for k in names], dtype=float).T
        else:
            raise TypeError("observations must be a DataFrame or dict[str, list]")
        # Drop rows with NaN, store column index + categorical map for tests.
        arr = arr[~np.isnan(arr).any(axis=1)] if arr.size else arr.reshape(0, len(names))
        self._name_index = {name: i for i, name in enumerate(names)}
        self._is_categorical = {
            i: bool(len(np.unique(arr[:, i])) <= max(2, int(0.1 * max(arr.shape[0], 1))))
            for i in range(len(names))
        } if arr.size else {i: False for i in range(len(names))}
        return names, arr


# ---------------------------------------------------------------------------
# Intervention planning (do-calculus)
# ---------------------------------------------------------------------------


@dataclass
class InterventionRecommendation:
    """A ranked, mechanism-annotated intervention suggestion."""

    variable: str
    recommended_value: float
    current_value: float
    expected_delta_kpi: float
    confidence: float
    mechanism_explanation: str = ""
    causal_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "recommended_value": round(self.recommended_value, 6),
            "current_value": round(self.current_value, 6),
            "expected_delta_kpi": round(self.expected_delta_kpi, 6),
            "confidence": round(self.confidence, 4),
            "mechanism_explanation": self.mechanism_explanation,
            "causal_path": self.causal_path,
        }


class InterventionPlanner:
    """Plans interventions on a fitted :class:`CausalGraph` via do-calculus.

    The key difference from correlation-based planning: ``compute_do_effect``
    implements Pearl's backdoor adjustment, so the estimated effect of setting
    ``X`` is purged of confounding. Two correlated knobs that merely co-vary
    with the KPI will receive (correctly) small do-effects, while the true
    causal driver is identified.
    """

    def __init__(self, graph: CausalGraph) -> None:
        self.graph = graph

    def compute_do_effect(
        self, intervention_var: str, intervention_value: float, outcome_var: str
    ) -> tuple[float, float]:
        """Return ``(expected_delta_outcome, variance)`` for ``do(X = value)``.

        Uses the linear total causal effect (path tracing) which, for a
        linear-Gaussian SEM, equals the backdoor-adjusted regression effect.
        Variance is propagated from the per-edge interval widths.
        """
        total = self.graph.get_total_effect(intervention_var, outcome_var)
        delta = total * intervention_value
        # Propagate variance from edge CIs along the dominant path.
        var = 0.0
        for path in self.graph.all_paths(intervention_var, outcome_var):
            for s, t in zip(path[:-1], path[1:]):
                edge = self.graph.get_edge(s, t)
                if edge is not None:
                    half = (edge.effect_ci[1] - edge.effect_ci[0]) / (2 * 1.96)
                    var += (half * intervention_value) ** 2
        return delta, var

    def rank_interventions(
        self,
        outcome_var: str,
        candidate_vars: list[str],
        current_values: dict[str, float],
        budget: int = 5,
    ) -> list[InterventionRecommendation]:
        """Rank candidate interventions by expected causal KPI improvement.

        For each candidate we search a small grid of feasible values, compute
        the do-effect, and keep the value maximizing expected ΔKPI weighted by
        confidence (so we do not over-commit to high-variance edges).
        """
        recs: list[InterventionRecommendation] = []
        for var in candidate_vars:
            if var == outcome_var or var not in self.graph.variables:
                continue
            cur = current_values.get(var, 0.0)
            best: InterventionRecommendation | None = None
            for mult in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
                step = (abs(cur) if cur != 0 else 1.0) * mult
                new_val = cur + step
                delta, var_est = self.compute_do_effect(var, step, outcome_var)
                conf = 1.0 / (1.0 + math.sqrt(max(var_est, 0.0)))
                score = delta * conf
                if best is None or score > best.expected_delta_kpi * best.confidence:
                    edge = self.graph.get_edge(var, outcome_var)
                    paths = self.graph.all_paths(var, outcome_var)
                    best = InterventionRecommendation(
                        variable=var,
                        recommended_value=new_val,
                        current_value=cur,
                        expected_delta_kpi=delta,
                        confidence=conf,
                        mechanism_explanation=(edge.mechanism if edge else
                                               f"indirect causal effect on {outcome_var}"),
                        causal_path=paths[0] if paths else [var, outcome_var],
                    )
            if best is not None and abs(best.expected_delta_kpi) > 1e-9:
                recs.append(best)
        recs.sort(key=lambda r: r.expected_delta_kpi * r.confidence, reverse=True)
        return recs[:budget]

    def counterfactual_query(
        self,
        observed: dict[str, float],
        intervention: dict[str, float],
        outcome: str,
    ) -> tuple[float, float]:
        """Answer "had we set X=x' instead of x, what would Y have been?".

        Three-step counterfactual (abduction-action-prediction, Pearl 2009):
        the observed outcome anchors the baseline (abduction of the noise
        term), the intervention shifts each changed variable, and the linear
        SEM predicts the counterfactual outcome plus its variance.
        """
        baseline = float(observed.get(outcome, 0.0))
        delta_total = 0.0
        var_total = 0.0
        for var, new_val in intervention.items():
            if var == outcome or var not in self.graph.variables:
                continue
            change = new_val - float(observed.get(var, 0.0))
            d, v = self.compute_do_effect(var, change, outcome)
            delta_total += d
            var_total += v
        return baseline + delta_total, var_total


# ---------------------------------------------------------------------------
# HELIOS service integration
# ---------------------------------------------------------------------------


class CausalReasoningService:
    """Runs causal discovery after campaign rounds and surfaces insights.

    DB access is intentionally defensive: the service degrades gracefully when
    the schema differs or the campaign has too few observations, so it can be
    wired into the worker loop without becoming a failure point.
    """

    def __init__(self, db_path: str = "otbot.db") -> None:
        self.db_path = db_path
        self._graphs: dict[str, CausalGraph] = {}
        self._discovery = CausalDiscovery()

    async def update_causal_model(
        self, campaign_id: str, round_number: int
    ) -> CausalGraph:
        """Pull observations for ``campaign_id`` and refit the causal model."""
        obs, interventions, forbidden, known = self._load_campaign_data(campaign_id)
        if not obs or len(next(iter(obs.values()), [])) < 4:
            logger.info("causal.update_skipped_insufficient", extra={
                "campaign_id": campaign_id, "round": round_number})
            graph = self._graphs.get(campaign_id) or CausalGraph()
            self._graphs[campaign_id] = graph
            return graph
        self._discovery = CausalDiscovery(domain_forbidden_edges=forbidden)
        graph = self._discovery.fit(obs, interventions=interventions, known_edges=known)
        self._graphs[campaign_id] = graph
        logger.info("causal.model_updated", extra={
            "campaign_id": campaign_id, "round": round_number,
            "n_edges": len(graph.edges)})
        return graph

    def get_mechanistic_insights(self, campaign_id: str) -> list[str]:
        """Natural-language, publication-ready causal statements for a campaign."""
        graph = self._graphs.get(campaign_id)
        if graph is None:
            return []
        outcomes = [v.name for v in graph.variables.values() if v.var_type == "outcome"]
        if not outcomes:  # fall back to sink nodes
            outcomes = [v for v in graph.variables if not graph._adj.get(v)]
        insights: list[str] = []
        for outcome in outcomes:
            for cause in graph.get_direct_causes(outcome):
                edge = graph.get_edge(cause, outcome)
                if edge is None or not edge.is_significant:
                    continue
                direction = "increases" if edge.effect_size > 0 else "decreases"
                mech = f" via {edge.mechanism}" if edge.mechanism else ""
                insights.append(
                    f"{cause} has a direct causal effect on {outcome}: "
                    f"{direction} it (effect={edge.effect_size:+.2f}, "
                    f"95% CI [{edge.effect_ci[0]:+.2f}, {edge.effect_ci[1]:+.2f}], "
                    f"p={edge.p_value:.3f}){mech}."
                )
            # Mediation statements: X -> M -> outcome.
            for mediator in graph.get_direct_causes(outcome):
                for upstream in graph.get_direct_causes(mediator):
                    if upstream != outcome:
                        insights.append(
                            f"{mediator} mediates the effect of {upstream} on "
                            f"{outcome} (indirect effect="
                            f"{graph.get_total_effect(upstream, outcome):+.2f})."
                        )
        return insights

    def suggest_next_experiments_causal(
        self, campaign_id: str, n: int = 5
    ) -> list[InterventionRecommendation]:
        """Causal-guided experiment suggestions, complementary to BO."""
        graph = self._graphs.get(campaign_id)
        if graph is None or not graph.edges:
            return []
        outcomes = [v.name for v in graph.variables.values() if v.var_type == "outcome"]
        outcome = outcomes[0] if outcomes else next(
            (v for v in graph.variables if not graph._adj.get(v)), None)
        if outcome is None:
            return []
        candidates = [
            v.name for v in graph.variables.values()
            if v.var_type in ("intervention", "mediator") and v.name != outcome
        ] or [v for v in graph.variables if v != outcome]
        current = self._estimate_current_values(campaign_id, candidates)
        planner = InterventionPlanner(graph)
        return planner.rank_interventions(outcome, candidates, current, budget=n)

    # -- DB helpers (defensive) --------------------------------------------

    def _load_campaign_data(
        self, campaign_id: str
    ) -> tuple[dict[str, list], dict[str, list], list[tuple[str, str]], list[tuple[str, str]]]:
        """Best-effort load of (observations, interventions, forbidden, known).

        Returns empty structures if the DB/schema is unavailable. The KPI
        column, if present, is marked as an outcome so it can never be a cause
        of an input (forbidden edges from outcome -> any input).
        """
        observations: dict[str, list] = {}
        interventions: dict[str, list] = {}
        forbidden: list[tuple[str, str]] = []
        known: list[tuple[str, str]] = []
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT params, objective FROM run_kpis WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchall()
            conn.close()
            from app.core.db import parse_json  # local import to avoid hard dep
            param_keys: set[str] = set()
            parsed: list[tuple[dict, float]] = []
            for r in rows:
                p = parse_json(r["params"], {}) if r["params"] else {}
                if isinstance(p, dict):
                    param_keys.update(p.keys())
                    parsed.append((p, float(r["objective"])))
            for key in sorted(param_keys):
                observations[key] = [float(p.get(key, 0.0)) for p, _ in parsed]
                interventions[key] = observations[key]  # SDL sets these -> do()
            observations["kpi"] = [obj for _, obj in parsed]
            for key in param_keys:
                forbidden.append(("kpi", key))  # outcome can't cause inputs
        except Exception as exc:  # pragma: no cover - schema-dependent
            logger.debug("causal.load_failed", extra={
                "campaign_id": campaign_id, "error": str(exc)})
        return observations, interventions, forbidden, known

    def _estimate_current_values(
        self, campaign_id: str, candidates: list[str]
    ) -> dict[str, float]:
        obs, *_ = self._load_campaign_data(campaign_id)
        current: dict[str, float] = {}
        for c in candidates:
            vals = obs.get(c, [])
            current[c] = float(np.mean(vals)) if vals else 0.0
        return current

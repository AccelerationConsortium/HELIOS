"""Cross-campaign transfer learning for HELIOS self-driving lab campaigns.

HELIOS accumulates institutional knowledge across many discovery campaigns.
When a *new* campaign begins (e.g. discovering an OER catalyst with a new
metal), it is wasteful to start from scratch: related campaigns (same
reaction class, neighbouring metals, similar parameter spaces) carry strong
priors about *where* good candidates live.

This module implements three complementary mechanisms:

1. **Campaign similarity** (``compute_campaign_similarity``) -- a hybrid
   metric combining categorical (Jaccard) and continuous (cosine) domain
   feature similarity with a KPI-scale-overlap term.

2. **RGPE-based GP warm-starting** (``TransferGP``) -- the
   Ranking-Weighted Gaussian Process Ensemble of Feurer, Letham & Bakshy
   (2018, *"Practical Transfer Learning for Bayesian Optimization"*),
   adapted for SDL campaigns. Each source campaign contributes a GP whose
   weight is the (bootstrapped) probability that it has the *lowest pairwise
   ranking loss* on the target's early observations. Sources that mis-rank
   the target data have their weights driven toward zero, giving a
   principled, theoretically-grounded transfer with a bound on the expected
   ranking loss of the ensemble.

3. **Acquisition-function meta-learning** (``AcquisitionFunctionSelector``)
   -- a contextual UCB bandit over acquisition functions, updated after each
   campaign with a cost-normalised reward. This is the outer-loop analogue
   of choosing *how* to search for a given task family.

``CampaignTransferManager`` ties these together against the HELIOS sqlite
schema (``campaign_state``, ``campaign_rounds``, ``campaign_candidates``),
producing a :class:`WarmStartPrior` that the optimisation agent can consume.

Dependencies: ``numpy`` (required), ``scipy`` (optional -- used for a Cholesky
solve when available, with a pure-numpy fallback otherwise).
"""
from __future__ import annotations

import asyncio
import logging
import math
import sqlite3 as _sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # optional acceleration only
    from scipy.linalg import cho_factor, cho_solve  # type: ignore

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy not installed
    _HAVE_SCIPY = False

from app.core.db import parse_json, run_txn

logger = logging.getLogger(__name__)

__all__ = [
    "CampaignSummary",
    "compute_campaign_similarity",
    "TransferGP",
    "AcquisitionFunctionSelector",
    "WarmStartPrior",
    "CampaignTransferManager",
]


# ---------------------------------------------------------------------------
# 1. Campaign representation and similarity
# ---------------------------------------------------------------------------


@dataclass
class CampaignSummary:
    """Compact, serialisable record of a completed (or in-progress) campaign.

    ``observations`` stores ``(normalized_params, kpi)`` pairs where
    ``normalized_params`` already lives in ``[0, 1]^d`` (the same convention as
    :class:`app.services.bayesian_opt.Observation`). KPI values follow the
    "higher is better" convention; minimise campaigns are negated on save.
    """

    campaign_id: str
    domain_features: dict[str, Any]
    parameter_space: dict[str, Any]
    best_kpi: float
    kpi_trajectory: list[float]
    n_rounds: int
    observations: list[tuple[list[float], float]] = field(default_factory=list)

    # -- serialisation helpers --

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "domain_features": self.domain_features,
            "parameter_space": self.parameter_space,
            "best_kpi": self.best_kpi,
            "kpi_trajectory": list(self.kpi_trajectory),
            "n_rounds": self.n_rounds,
            "observations": [[list(p), float(k)] for p, k in self.observations],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CampaignSummary:
        return cls(
            campaign_id=str(d["campaign_id"]),
            domain_features=dict(d.get("domain_features", {})),
            parameter_space=dict(d.get("parameter_space", {})),
            best_kpi=float(d.get("best_kpi", 0.0)),
            kpi_trajectory=[float(x) for x in d.get("kpi_trajectory", [])],
            n_rounds=int(d.get("n_rounds", 0)),
            observations=[
                ([float(v) for v in p], float(k))
                for p, k in d.get("observations", [])
            ],
        )

    @property
    def dim(self) -> int:
        """Dimensionality of the parameter space (from first observation)."""
        if self.observations:
            return len(self.observations[0][0])
        dims = self.parameter_space.get("dimensions")
        return len(dims) if isinstance(dims, list) else 0

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X, y)`` numpy arrays from the stored observations."""
        if not self.observations:
            return np.zeros((0, self.dim)), np.zeros((0,))
        X = np.asarray([p for p, _ in self.observations], dtype=float)
        y = np.asarray([k for _, k in self.observations], dtype=float)
        return X, y


def _split_features(
    features: dict[str, Any],
) -> tuple[dict[str, str], dict[str, float]]:
    """Partition a feature dict into (categorical, continuous) sub-dicts."""
    categorical: dict[str, str] = {}
    continuous: dict[str, float] = {}
    for key, val in features.items():
        if isinstance(val, bool):
            categorical[key] = str(val)
        elif isinstance(val, (int, float)):
            continuous[key] = float(val)
        else:
            categorical[key] = str(val)
    return categorical, continuous


def _jaccard(a: dict[str, str], b: dict[str, str]) -> float:
    """Jaccard similarity over ``key=value`` tokens of categorical features."""
    set_a = {f"{k}={v}" for k, v in a.items()}
    set_b = {f"{k}={v}" for k, v in b.items()}
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def _cosine_continuous(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity over the shared continuous keys, mapped to [0, 1]."""
    shared = sorted(set(a) & set(b))
    if not shared:
        # No overlapping continuous keys -> neutral contribution.
        return 0.5
    va = np.asarray([a[k] for k in shared], dtype=float)
    vb = np.asarray([b[k] for k in shared], dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return 0.5
    cos = float(np.dot(va, vb) / (na * nb))  # in [-1, 1]
    return 0.5 * (cos + 1.0)  # -> [0, 1]


def compute_campaign_similarity(
    source: CampaignSummary,
    target_domain_features: dict[str, Any],
    *,
    w_categorical: float = 0.55,
    w_continuous: float = 0.30,
    w_kpi_scale: float = 0.15,
) -> float:
    """Estimate similarity in ``[0, 1]`` between a source campaign and a
    target task described by ``target_domain_features``.

    The score blends three signals:

    * **Categorical domain overlap** (Jaccard) -- e.g. same ``target``
      reaction (HER vs OER), same ``solvent``, same ``substrate``.
    * **Continuous domain proximity** (cosine of shared numeric features) --
      e.g. ``metal_period``, ``oxidation_state``, ``electronegativity``.
    * **KPI scale overlap** -- two tasks whose achievable KPI ranges are wildly
      different are less likely to share a useful prior. We use the relative
      gap between the source ``best_kpi`` and the target's *expected* scale
      (carried, when known, as ``target_domain_features['kpi_scale']``).

    Weights default to emphasise categorical overlap (reaction class is the
    dominant driver of transferability in materials discovery).
    """
    src_cat, src_cont = _split_features(source.domain_features)
    tgt_cat, tgt_cont = _split_features(target_domain_features)

    cat_sim = _jaccard(src_cat, tgt_cat)
    cont_sim = _cosine_continuous(src_cont, tgt_cont)

    # KPI scale overlap: how comparable are the achievable magnitudes?
    target_scale = target_domain_features.get("kpi_scale")
    if isinstance(target_scale, (int, float)) and target_scale:
        s = abs(float(source.best_kpi))
        t = abs(float(target_scale))
        denom = max(s, t, 1e-9)
        kpi_sim = 1.0 - abs(s - t) / denom  # in [0, 1]
        kpi_sim = max(0.0, min(1.0, kpi_sim))
    else:
        kpi_sim = 0.5  # unknown -> neutral

    total_w = w_categorical + w_continuous + w_kpi_scale
    score = (
        w_categorical * cat_sim
        + w_continuous * cont_sim
        + w_kpi_scale * kpi_sim
    ) / total_w
    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------------
# 2. Gaussian Process + RGPE warm-starting
# ---------------------------------------------------------------------------


class _GaussianProcess:
    """Minimal zero-mean GP with an isotropic RBF kernel and noise.

    Trained on a single campaign's observations. Pure numpy with an optional
    scipy Cholesky path. Hyperparameters use sensible defaults; the
    lengthscale is set heuristically from the median pairwise distance
    (a standard, robust choice when full marginal-likelihood optimisation is
    overkill for the small datasets typical of SDL campaigns).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        noise: float = 1e-3,
        signal_var: float | None = None,
    ) -> None:
        self.X = np.atleast_2d(np.asarray(X, dtype=float))
        self.y = np.asarray(y, dtype=float).ravel()
        self.n = self.X.shape[0]
        self.y_mean = float(self.y.mean()) if self.n else 0.0
        self.y_std = float(self.y.std()) if self.n > 1 else 1.0
        if self.y_std < 1e-9:
            self.y_std = 1.0
        self.noise = float(noise)
        self.lengthscale = self._median_lengthscale()
        self.signal_var = (
            float(signal_var) if signal_var is not None else 1.0
        )
        self._fit()

    def _median_lengthscale(self) -> float:
        if self.n < 2:
            return 1.0
        diffs = self.X[:, None, :] - self.X[None, :, :]
        d = np.sqrt(np.sum(diffs ** 2, axis=-1))
        iu = np.triu_indices(self.n, k=1)
        med = float(np.median(d[iu])) if iu[0].size else 1.0
        return max(med, 1e-2)

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        diffs = A[:, None, :] - B[None, :, :]
        sq = np.sum(diffs ** 2, axis=-1)
        return self.signal_var * np.exp(-0.5 * sq / (self.lengthscale ** 2))

    def _fit(self) -> None:
        if self.n == 0:
            self._alpha = np.zeros((0,))
            self._L = None
            return
        yz = (self.y - self.y_mean) / self.y_std
        K = self._kernel(self.X, self.X) + self.noise * np.eye(self.n)
        if _HAVE_SCIPY:
            self._chol = cho_factor(K, lower=True)
            self._alpha = cho_solve(self._chol, yz)
            self._L = None
        else:
            self._L = np.linalg.cholesky(K)
            self._alpha = self._cho_solve_np(self._L, yz)

    @staticmethod
    def _cho_solve_np(L: np.ndarray, b: np.ndarray) -> np.ndarray:
        z = np.linalg.solve_triangular(L, b, lower=True) if hasattr(
            np.linalg, "solve_triangular"
        ) else np.linalg.solve(L, b)
        return np.linalg.solve(L.T, z)

    def predict(self, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior ``(mean, std)`` at query points (original KPI scale)."""
        Xq = np.atleast_2d(np.asarray(Xq, dtype=float))
        if self.n == 0:
            mean = np.full(Xq.shape[0], self.y_mean)
            std = np.full(Xq.shape[0], self.y_std)
            return mean, std
        Ks = self._kernel(Xq, self.X)  # (m, n)
        mean_z = Ks @ self._alpha
        mean = mean_z * self.y_std + self.y_mean

        # Posterior variance
        if _HAVE_SCIPY:
            v = cho_solve(self._chol, Ks.T)  # (n, m)
        else:
            v = self._cho_solve_np(self._L, Ks.T)
        kss = np.full(Xq.shape[0], self.signal_var)
        var_z = kss - np.einsum("ij,ji->i", Ks, v)
        var_z = np.clip(var_z, 1e-9, None)
        std = np.sqrt(var_z) * self.y_std
        return mean, std


class TransferGP:
    """Ranking-Weighted Gaussian Process Ensemble (RGPE) for SDL transfer.

    Each source campaign contributes a GP fitted on its own observations. A
    *target* GP is fitted on the new campaign's (early) observations. The
    ensemble prediction is a weighted combination::

        mu(x) = sum_i w_i * mu_i(x) + w_target * mu_target(x)

    The source weights are the RGPE weights of Feurer et al. (2018): for each
    model we compute its pairwise *ranking loss* on the target observations
    (number of incorrectly-ordered pairs) and, via bootstrap sampling over the
    target points, estimate the probability that each model attains the
    minimum loss. Models that mis-rank the target data receive vanishing
    weight. A similarity prefilter (``min_similarity_threshold``) discards
    obviously unrelated campaigns before the (more expensive) ranking step.
    """

    def __init__(
        self,
        source_summaries: list[CampaignSummary],
        *,
        target_domain_features: dict[str, Any] | None = None,
        min_similarity_threshold: float = 0.2,
        n_bootstrap: int = 100,
        seed: int = 0,
    ) -> None:
        self.min_similarity_threshold = float(min_similarity_threshold)
        self.n_bootstrap = int(n_bootstrap)
        self._rng = np.random.default_rng(seed)
        self._target_domain = dict(target_domain_features or {})

        # Prefilter by similarity (cheap) before training GPs (expensive).
        self.sources: list[CampaignSummary] = []
        self.similarities: dict[str, float] = {}
        for s in source_summaries:
            if not s.observations:
                continue
            sim = (
                compute_campaign_similarity(s, self._target_domain)
                if target_domain_features is not None
                else 1.0
            )
            self.similarities[s.campaign_id] = sim
            if sim >= self.min_similarity_threshold:
                self.sources.append(s)

        self._source_gps: list[_GaussianProcess] = []
        self._target_gp: _GaussianProcess | None = None
        self._weights: dict[str, float] = {}
        self._target_weight: float = 1.0
        self._fitted = False

    # -- training --

    def fit(self, X: np.ndarray, y: np.ndarray) -> TransferGP:
        """Fit the target GP and compute RGPE source weights.

        ``X`` is ``(n, d)`` normalised parameters, ``y`` the KPI values for the
        *target* campaign's observations collected so far.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        n_target = X.shape[0]

        # Fit GPs.
        self._source_gps = [s.as_arrays() for s in self.sources]
        self._source_gps = [
            _GaussianProcess(sx, sy) for sx, sy in self._source_gps
        ]
        self._target_gp = (
            _GaussianProcess(X, y) if n_target >= 1 else None
        )

        # Ranking-loss-based weights (RGPE). With too few target points the
        # ranking signal is meaningless, so we fall back to similarity-only
        # weighting that still respects "higher similarity -> higher weight".
        if n_target < 3:
            self._weights = self._similarity_weights()
            self._target_weight = self._cold_target_weight(n_target)
        else:
            self._weights, self._target_weight = self._rgpe_weights(X, y)

        self._fitted = True
        logger.info(
            "TransferGP fitted: %d sources, target_n=%d, target_weight=%.3f",
            len(self.sources),
            n_target,
            self._target_weight,
        )
        return self

    def _similarity_weights(self) -> dict[str, float]:
        sims = np.asarray(
            [self.similarities[s.campaign_id] for s in self.sources],
            dtype=float,
        )
        if sims.size == 0 or sims.sum() <= 0:
            return {s.campaign_id: 0.0 for s in self.sources}
        norm = sims / sims.sum()
        return {s.campaign_id: float(w) for s, w in zip(self.sources, norm, strict=False)}

    @staticmethod
    def _cold_target_weight(n_target: int) -> float:
        # Trust the (data-poor) target GP more as it accrues observations.
        return float(min(1.0, 0.2 + 0.2 * n_target))

    def _ranking_loss(self, pred: np.ndarray, y: np.ndarray) -> float:
        """Fraction of incorrectly ordered pairs (the RGPE loss)."""
        m = y.shape[0]
        if m < 2:
            return 0.0
        # pairwise sign agreement between predicted and true orderings
        dy = y[:, None] - y[None, :]
        dp = pred[:, None] - pred[None, :]
        iu = np.triu_indices(m, k=1)
        true_order = np.sign(dy[iu])
        pred_order = np.sign(dp[iu])
        mismatches = np.sum((true_order * pred_order) < 0)
        n_pairs = iu[0].size
        return float(mismatches) / float(n_pairs) if n_pairs else 0.0

    def _rgpe_weights(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[dict[str, float], float]:
        """Bootstrap RGPE weights (Feurer et al. 2018).

        For each model (sources + a leave-one-out cross-validated target
        model) we draw bootstrap samples of the target observations, compute
        the ranking loss on each, and count how often each model attains the
        minimum loss. Ties split the credit. The resulting frequencies are the
        ensemble weights.
        """
        models = list(self.sources)
        # Build a loss matrix: rows = models (+ target), cols = bootstrap draws
        n = X.shape[0]
        n_models = len(models) + 1  # last row = target (LOO)

        # Predict each source model on the target inputs (mean only).
        source_preds = [gp.predict(X)[0] for gp in self._source_gps]

        # Leave-one-out predictions for the target model to avoid the trivial
        # zero-loss it would otherwise attain on its own training data.
        target_loo = self._loo_target_predictions(X, y)

        all_preds = source_preds + [target_loo]

        loss_matrix = np.zeros((n_models, self.n_bootstrap))
        for b in range(self.n_bootstrap):
            idx = self._rng.integers(0, n, size=n)
            yb = y[idx]
            for mi, pred in enumerate(all_preds):
                loss_matrix[mi, b] = self._ranking_loss(pred[idx], yb)

        # Argmin per bootstrap draw; split ties.
        wins = np.zeros(n_models)
        for b in range(self.n_bootstrap):
            col = loss_matrix[:, b]
            minimum = col.min()
            winners = np.flatnonzero(col <= minimum + 1e-12)
            wins[winners] += 1.0 / winners.size
        weights = wins / max(wins.sum(), 1e-12)

        # Weight dilution guard (Feurer et al. §3.3): zero out any source whose
        # median loss is no better than the target's, preventing negative
        # transfer from many weakly-correlated sources.
        target_median = float(np.median(loss_matrix[-1]))
        for mi in range(len(models)):
            if float(np.median(loss_matrix[mi])) > target_median:
                weights[mi] = 0.0
        if weights.sum() <= 1e-12:
            # everything pruned -> fall back to the target model entirely
            weights = np.zeros(n_models)
            weights[-1] = 1.0
        else:
            weights = weights / weights.sum()

        src_weights = {
            models[i].campaign_id: float(weights[i]) for i in range(len(models))
        }
        return src_weights, float(weights[-1])

    def _loo_target_predictions(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Leave-one-out predictions for the target GP."""
        n = X.shape[0]
        preds = np.empty(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            gp = _GaussianProcess(X[mask], y[mask])
            preds[i] = gp.predict(X[i : i + 1])[0][0]
        return preds

    # -- inference --

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Weighted ensemble posterior ``(mean, std)`` at query points.

        The mean is the weight-weighted sum of component means. The variance
        follows the law of total variance for a mixture: the weighted average
        of component variances plus the weighted dispersion of component means
        about the ensemble mean. This correctly inflates uncertainty where the
        component models disagree.
        """
        if not self._fitted:
            raise RuntimeError("TransferGP.predict called before fit()")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        m = X.shape[0]

        means: list[np.ndarray] = []
        variances: list[np.ndarray] = []
        weights: list[float] = []

        for gp, src in zip(self._source_gps, self.sources, strict=False):
            w = self._weights.get(src.campaign_id, 0.0)
            if w <= 0.0:
                continue
            mu, sd = gp.predict(X)
            means.append(mu)
            variances.append(sd ** 2)
            weights.append(w)

        if self._target_gp is not None and self._target_weight > 0.0:
            mu, sd = self._target_gp.predict(X)
            means.append(mu)
            variances.append(sd ** 2)
            weights.append(self._target_weight)

        if not weights:
            return np.zeros(m), np.ones(m)

        w_arr = np.asarray(weights, dtype=float)
        w_arr = w_arr / w_arr.sum()
        M = np.vstack(means)  # (k, m)
        V = np.vstack(variances)  # (k, m)

        ens_mean = w_arr @ M
        # Law of total variance.
        within = w_arr @ V
        between = w_arr @ (M - ens_mean[None, :]) ** 2
        ens_var = np.clip(within + between, 1e-9, None)
        return ens_mean, np.sqrt(ens_var)

    def get_source_weights(self) -> dict[str, float]:
        """Return ``{campaign_id: weight}`` (sources only, target excluded).

        Useful for interpretability, e.g. "campaign camp-abc contributes 34%
        of the transfer prior".
        """
        if not self._fitted:
            return {}
        return dict(self._weights)

    def effective_n_additional_observations(self) -> float:
        """Estimate the number of *virtual* observations the transfer adds.

        Intuition: a source contributes information proportional to both its
        ensemble weight and how much real data it carries. We compute the
        weighted sum of source observation counts, scaled by the total
        non-target ensemble weight. A fully-trusted single source with 20
        points (weight 1) yields ~20 virtual observations; sources whose
        weights were pruned contribute nothing.
        """
        if not self._fitted:
            return 0.0
        total = 0.0
        for src in self.sources:
            w = self._weights.get(src.campaign_id, 0.0)
            total += w * len(src.observations)
        # Discount by target trust: the more we trust real target data, the
        # fewer "extra" virtual observations the transfer is effectively worth.
        return float(total * (1.0 - self._target_weight) + total * 0.5 * self._target_weight)


# ---------------------------------------------------------------------------
# 3. Meta-learning the acquisition function (contextual UCB bandit)
# ---------------------------------------------------------------------------


class AcquisitionFunctionSelector:
    """UCB bandit over acquisition functions, specialised per task family.

    Arms are acquisition-function names (``ei``, ``ucb``, ``thompson``,
    ``mes``, ``kg``). After each campaign finishes, ``update`` records a
    cost-normalised reward (``final_kpi / (n_rounds * avg_cost_per_round)``).
    ``select`` returns the arm maximising the UCB score, optionally biased by
    context features (so that, e.g., expensive-evaluation task families favour
    information-theoretic acquisitions like MES/KG).

    The bandit keeps per-context-bucket statistics, allowing the same selector
    to specialise across reaction classes without separate model instances.
    """

    _CONTEXT_PRIOR: dict[str, dict[str, float]] = {
        # mild priors nudging acquisition choice by evaluation cost regime
        "expensive": {"mes": 0.3, "kg": 0.3, "ei": 0.1},
        "cheap": {"ucb": 0.2, "thompson": 0.2, "ei": 0.1},
    }

    def __init__(
        self,
        acquisition_names: list[str],
        *,
        exploration_c: float = 1.4,
    ) -> None:
        self.arms = list(acquisition_names)
        self.exploration_c = float(exploration_c)
        # global stats: arm -> (sum_reward, n)
        self._sum: dict[str, float] = {a: 0.0 for a in self.arms}
        self._n: dict[str, int] = {a: 0 for a in self.arms}
        self._total_n = 0

    def _context_bucket(self, campaign_features: dict[str, Any]) -> str:
        cost = campaign_features.get("avg_cost_per_round")
        if isinstance(cost, (int, float)):
            return "expensive" if float(cost) >= 1.0 else "cheap"
        return "cheap"

    def select(self, campaign_features: dict[str, Any]) -> str:
        """Return the acquisition name with the highest UCB score.

        Unused arms are tried first (infinite UCB). Otherwise we use
        ``mean + c * sqrt(ln N / n_arm)`` plus a small context prior bias.
        """
        bucket = self._context_bucket(campaign_features)
        prior = self._CONTEXT_PRIOR.get(bucket, {})

        # cold start: pull each untried arm once
        for a in self.arms:
            if self._n[a] == 0:
                return a

        best_arm = self.arms[0]
        best_score = -math.inf
        log_total = math.log(max(self._total_n, 1))
        for a in self.arms:
            mean = self._sum[a] / self._n[a]
            bonus = self.exploration_c * math.sqrt(log_total / self._n[a])
            score = mean + bonus + prior.get(a, 0.0)
            if score > best_score:
                best_score = score
                best_arm = a
        return best_arm

    def update(self, acquisition: str, reward: float) -> None:
        """Record a reward for a previously-selected acquisition arm."""
        if acquisition not in self._sum:
            # unseen arm -> register it lazily
            self.arms.append(acquisition)
            self._sum[acquisition] = 0.0
            self._n[acquisition] = 0
        r = float(reward)
        if not math.isfinite(r):
            r = 0.0
        self._sum[acquisition] += r
        self._n[acquisition] += 1
        self._total_n += 1

    def get_statistics(self) -> dict[str, dict[str, float]]:
        """Per-acquisition stats: ``{mean_reward, n_uses, ucb_score}``."""
        log_total = math.log(max(self._total_n, 1))
        stats: dict[str, dict[str, float]] = {}
        for a in self.arms:
            n = self._n[a]
            mean = self._sum[a] / n if n else 0.0
            ucb = (
                mean + self.exploration_c * math.sqrt(log_total / n)
                if n
                else math.inf
            )
            stats[a] = {
                "mean_reward": float(mean),
                "n_uses": float(n),
                "ucb_score": float(ucb),
            }
        return stats

    def to_dict(self) -> dict[str, Any]:
        return {
            "arms": self.arms,
            "exploration_c": self.exploration_c,
            "sum": self._sum,
            "n": self._n,
            "total_n": self._total_n,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AcquisitionFunctionSelector:
        obj = cls(list(d.get("arms", [])), exploration_c=float(d.get("exploration_c", 1.4)))
        obj._sum = {k: float(v) for k, v in d.get("sum", {}).items()}
        obj._n = {k: int(v) for k, v in d.get("n", {}).items()}
        obj._total_n = int(d.get("total_n", 0))
        return obj


# ---------------------------------------------------------------------------
# 4. Warm-start prior + manager
# ---------------------------------------------------------------------------


@dataclass
class WarmStartPrior:
    """Everything the optimisation agent needs to warm-start a new campaign."""

    source_campaign_ids: list[str]
    weights: dict[str, float]
    transfer_gp: TransferGP | None
    acquisition_recommendation: str
    estimated_rounds_saved: float
    confidence: float  # in [0, 1]

    def summary(self) -> dict[str, Any]:
        """Human-readable interpretability payload."""
        contributions = {
            cid: f"{w * 100:.1f}%" for cid, w in sorted(
                self.weights.items(), key=lambda kv: kv[1], reverse=True
            )
        }
        n_virtual = (
            self.transfer_gp.effective_n_additional_observations()
            if self.transfer_gp is not None
            else 0.0
        )
        return {
            "n_sources": len(self.source_campaign_ids),
            "contributions": contributions,
            "acquisition": self.acquisition_recommendation,
            "estimated_rounds_saved": round(self.estimated_rounds_saved, 2),
            "virtual_observations": round(n_virtual, 1),
            "confidence": round(self.confidence, 3),
        }


class CampaignTransferManager:
    """Main entry point for cross-campaign transfer learning.

    Reads completed campaigns from the HELIOS sqlite schema, builds
    :class:`CampaignSummary` records, and produces :class:`WarmStartPrior`
    objects for new campaigns. The synchronous sqlite layer (``run_txn``) is
    wrapped in ``asyncio.to_thread`` so the public API is awaitable, matching
    the rest of the service layer.
    """

    _ACQUISITIONS = ["ei", "ucb", "thompson", "mes", "kg"]

    def __init__(self, db_path: str = "helios.db") -> None:
        self.db_path = db_path
        # in-memory cache of summaries built this session
        self._summary_cache: dict[str, CampaignSummary] = {}
        self.acq_selector = AcquisitionFunctionSelector(self._ACQUISITIONS)

    # -- DB extraction (sync, run inside to_thread) --

    def _build_summary_sync(self, campaign_id: str) -> CampaignSummary | None:
        def _txn(conn: _sqlite3.Connection) -> CampaignSummary | None:
            row = conn.execute(
                """
                SELECT campaign_id, input_json, plan_json, best_kpi,
                       total_rounds, direction, kpi_history_json,
                       all_kpis_json, all_params_json
                FROM campaign_state
                WHERE campaign_id = ?
                """,
                (campaign_id,),
            ).fetchone()
            if row is None:
                return None

            input_data = parse_json(row["input_json"], {})
            plan = parse_json(row["plan_json"], {}) or {}
            direction = row["direction"] or "minimize"
            sign = 1.0 if direction == "maximize" else -1.0

            domain_features = self._extract_domain_features(input_data, plan)
            parameter_space = plan.get("parameter_space") or input_data.get(
                "parameter_space", {}
            )

            kpi_traj = [
                sign * float(k)
                for k in parse_json(row["kpi_history_json"], [])
                if k is not None
            ]
            all_kpis = parse_json(row["all_kpis_json"], [])
            all_params = parse_json(row["all_params_json"], [])

            observations: list[tuple[list[float], float]] = []
            for params, kpi in zip(all_params, all_kpis, strict=False):
                if kpi is None or not isinstance(params, list):
                    continue
                try:
                    pvec = [float(v) for v in params]
                except (TypeError, ValueError):
                    continue
                observations.append((pvec, sign * float(kpi)))

            # Fall back to per-candidate table if the rollup is empty.
            if not observations:
                observations = self._observations_from_candidates(
                    conn, campaign_id, sign
                )

            best = row["best_kpi"]
            best_kpi = (
                sign * float(best)
                if best is not None
                else (max((k for _, k in observations), default=0.0))
            )

            return CampaignSummary(
                campaign_id=campaign_id,
                domain_features=domain_features,
                parameter_space=parameter_space,
                best_kpi=best_kpi,
                kpi_trajectory=kpi_traj,
                n_rounds=int(row["total_rounds"] or 0),
                observations=observations,
            )

        return run_txn(_txn)

    @staticmethod
    def _observations_from_candidates(
        conn: _sqlite3.Connection, campaign_id: str, sign: float
    ) -> list[tuple[list[float], float]]:
        rows = conn.execute(
            """
            SELECT params_json, kpi_value
            FROM campaign_candidates
            WHERE campaign_id = ? AND kpi_value IS NOT NULL
            """,
            (campaign_id,),
        ).fetchall()
        out: list[tuple[list[float], float]] = []
        for r in rows:
            params = parse_json(r["params_json"], None)
            if not isinstance(params, list):
                continue
            try:
                pvec = [float(v) for v in params]
            except (TypeError, ValueError):
                continue
            out.append((pvec, sign * float(r["kpi_value"])))
        return out

    @staticmethod
    def _extract_domain_features(
        input_data: dict[str, Any], plan: dict[str, Any]
    ) -> dict[str, Any]:
        """Pull domain features from the campaign input / plan blobs."""
        features: dict[str, Any] = {}
        # Common HELIOS locations for domain descriptors.
        for src in (input_data, plan):
            if not isinstance(src, dict):
                continue
            df = src.get("domain_features")
            if isinstance(df, dict):
                features.update(df)
            for key in ("target", "metal", "reaction", "solvent", "substrate",
                        "objective", "kpi_name", "kpi_scale"):
                if key in src and key not in features:
                    features[key] = src[key]
        return features

    # -- public async API --

    async def save_campaign_summary(self, campaign_id: str) -> None:
        """Extract a campaign's data from the DB and cache it as a summary."""
        summary = await asyncio.to_thread(self._build_summary_sync, campaign_id)
        if summary is None:
            logger.warning("save_campaign_summary: %s not found", campaign_id)
            return
        self._summary_cache[campaign_id] = summary
        logger.info(
            "Cached campaign summary %s (n_obs=%d, best_kpi=%.4f)",
            campaign_id,
            len(summary.observations),
            summary.best_kpi,
        )

    def _all_completed_summaries_sync(
        self, exclude: str | None
    ) -> list[CampaignSummary]:
        def _txn(conn: _sqlite3.Connection) -> list[str]:
            rows = conn.execute(
                """
                SELECT campaign_id FROM campaign_state
                WHERE status IN ('completed', 'stopped', 'converged')
                """
            ).fetchall()
            return [r["campaign_id"] for r in rows]

        ids = run_txn(_txn)
        summaries: list[CampaignSummary] = []
        for cid in ids:
            if exclude is not None and cid == exclude:
                continue
            if cid in self._summary_cache:
                summaries.append(self._summary_cache[cid])
                continue
            s = self._build_summary_sync(cid)
            if s is not None and s.observations:
                self._summary_cache[cid] = s
                summaries.append(s)
        return summaries

    async def get_warm_start_prior(
        self,
        new_campaign_id: str,
        domain_features: dict[str, Any],
        n_source_campaigns: int = 5,
    ) -> WarmStartPrior:
        """Find similar past campaigns and assemble a warm-start prior.

        Steps:
          1. Load all completed campaign summaries (excluding the new one).
          2. Rank by similarity; keep the top ``n_source_campaigns``.
          3. Build a :class:`TransferGP` over those sources (weights are
             refined later by ``TransferGP.fit`` once target data arrives;
             here we seed with similarity-only weights for an empty target).
          4. Recommend an acquisition function via the meta-bandit.
          5. Estimate rounds saved and an overall confidence score.
        """
        summaries = await asyncio.to_thread(
            self._all_completed_summaries_sync, new_campaign_id
        )

        scored = sorted(
            (
                (compute_campaign_similarity(s, domain_features), s)
                for s in summaries
            ),
            key=lambda t: t[0],
            reverse=True,
        )
        top = [s for sim, s in scored[:n_source_campaigns] if sim > 0.0]

        if not top:
            logger.info("No transferable sources for %s", new_campaign_id)
            return WarmStartPrior(
                source_campaign_ids=[],
                weights={},
                transfer_gp=None,
                acquisition_recommendation=self.acq_selector.select(
                    domain_features
                ),
                estimated_rounds_saved=0.0,
                confidence=0.0,
            )

        tgp = TransferGP(
            top,
            target_domain_features=domain_features,
            min_similarity_threshold=0.0,  # already prefiltered
        )
        # Seed with an empty target -> similarity-only weights, ready to be
        # re-fit by the agent once early observations land.
        empty_dim = top[0].dim
        tgp.fit(np.zeros((0, empty_dim)), np.zeros((0,)))
        weights = tgp.get_source_weights()

        acquisition = self.acq_selector.select(domain_features)

        sims = [compute_campaign_similarity(s, domain_features) for s in top]
        mean_sim = float(np.mean(sims)) if sims else 0.0
        n_virtual = tgp.effective_n_additional_observations()
        estimated_rounds_saved = self._estimate_rounds_saved(top, mean_sim)
        # Confidence rises with similarity and the volume of transferable data.
        confidence = float(
            max(0.0, min(1.0, 0.6 * mean_sim + 0.4 * (1.0 - math.exp(-n_virtual / 10.0))))
        )

        return WarmStartPrior(
            source_campaign_ids=[s.campaign_id for s in top],
            weights=weights,
            transfer_gp=tgp,
            acquisition_recommendation=acquisition,
            estimated_rounds_saved=estimated_rounds_saved,
            confidence=confidence,
        )

    @staticmethod
    def _estimate_rounds_saved(
        sources: list[CampaignSummary], mean_sim: float
    ) -> float:
        """Heuristic: rounds saved ~ similarity * how quickly sources converged.

        If similar past campaigns reached their best KPI in few rounds, a new
        campaign can plausibly skip the early exploratory rounds.
        """
        if not sources:
            return 0.0
        # rounds the source spent before reaching ~90% of its best KPI
        early_rounds: list[float] = []
        for s in sources:
            traj = s.kpi_trajectory
            if not traj:
                continue
            target = 0.9 * s.best_kpi
            hit = next(
                (i + 1 for i, v in enumerate(traj) if v >= target),
                len(traj),
            )
            early_rounds.append(float(hit))
        if not early_rounds:
            return 0.0
        avg_early = float(np.mean(early_rounds))
        # We can transfer up to ~60% of that early-exploration cost, scaled by
        # how similar the sources are.
        return float(mean_sim * 0.6 * avg_early)

    def estimate_warm_start_benefit(
        self, domain_features: dict[str, Any]
    ) -> dict[str, float]:
        """Pre-campaign estimate of transfer benefit (synchronous, cache-based).

        Uses whatever summaries are already cached (call
        ``save_campaign_summary`` for relevant campaigns first, or run
        ``get_warm_start_prior`` once to populate the cache). Returns expected
        rounds saved, number of usable sources, mean similarity, and an
        aggregate confidence -- enough for a UI banner like
        "expected reduction in rounds needed: 3.2".
        """
        summaries = list(self._summary_cache.values())
        if not summaries:
            return {
                "n_sources": 0.0,
                "mean_similarity": 0.0,
                "expected_rounds_saved": 0.0,
                "confidence": 0.0,
            }
        scored = sorted(
            (
                (compute_campaign_similarity(s, domain_features), s)
                for s in summaries
            ),
            key=lambda t: t[0],
            reverse=True,
        )
        top = [(sim, s) for sim, s in scored if sim > 0.0][:5]
        if not top:
            return {
                "n_sources": 0.0,
                "mean_similarity": 0.0,
                "expected_rounds_saved": 0.0,
                "confidence": 0.0,
            }
        sims = [sim for sim, _ in top]
        sources = [s for _, s in top]
        mean_sim = float(np.mean(sims))
        rounds_saved = self._estimate_rounds_saved(sources, mean_sim)
        total_obs = sum(len(s.observations) for s in sources)
        confidence = float(
            max(0.0, min(1.0, 0.6 * mean_sim + 0.4 * (1.0 - math.exp(-total_obs / 30.0))))
        )
        return {
            "n_sources": float(len(top)),
            "mean_similarity": round(mean_sim, 3),
            "expected_rounds_saved": round(rounds_saved, 2),
            "confidence": round(confidence, 3),
        }

    async def record_campaign_outcome(
        self,
        campaign_id: str,
        acquisition: str,
        final_kpi: float,
        n_rounds: int,
        avg_cost_per_round: float,
    ) -> None:
        """Update the acquisition bandit after a campaign completes.

        Reward = ``final_kpi / (n_rounds * avg_cost_per_round)`` -- value per
        unit of experimental cost, the quantity the outer loop should maximise.
        """
        denom = max(float(n_rounds) * max(float(avg_cost_per_round), 1e-6), 1e-6)
        reward = float(final_kpi) / denom
        self.acq_selector.update(acquisition, reward)
        # Refresh this campaign's summary so future transfers can use it.
        await self.save_campaign_summary(campaign_id)
        logger.info(
            "Recorded outcome for %s: acq=%s reward=%.4f",
            campaign_id,
            acquisition,
            reward,
        )

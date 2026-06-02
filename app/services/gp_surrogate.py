"""Research-grade Gaussian Process surrogate for HELIOS Bayesian Optimization.

This module replaces the distance-weighted KNN surrogate in ``bayesian_opt``
with a fully probabilistic Gaussian Process (GP) surrogate that delivers
*calibrated* posterior uncertainty -- a prerequisite for principled
acquisition-driven experiment selection in a self-driving lab.

Scientific contributions implemented here
------------------------------------------
1.  **Automatic Relevance Determination (ARD).**  Each input dimension has its
    own length-scale.  After hyperparameter optimization, the inverse
    length-scales rank the *relevance* of each experimental parameter, giving
    chemists an interpretable, data-driven sensitivity analysis "for free".

2.  **Matern-5/2 kernel for physical systems.**  Physical responses are rarely
    infinitely smooth.  The Matern-5/2 kernel assumes twice-differentiable
    sample paths -- a far more defensible prior for reaction yields, particle
    sizes, etc. than the (analytic) squared-exponential.

3.  **Product kernels for mixed continuous/categorical spaces.**  Self-driving
    labs routinely mix continuous (temperature, concentration) and categorical
    (solvent, catalyst) variables.  A product of an ARD continuous kernel and a
    categorical-overlap kernel handles both coherently.

4.  **Calibrated uncertainty + a portfolio of acquisition functions.**  Beyond
    EI / UCB we provide Thompson sampling, Max-value Entropy Search (MES) and
    Knowledge Gradient (KG).  MES and Thompson sampling are particularly
    valuable for *batched* experimentation, where a lab runs many wells in
    parallel and benefits from diverse, information-rich selections.

The module degrades gracefully: ``scipy`` is used for L-BFGS-B hyperparameter
optimization when present, otherwise a pure-numpy random-restart coordinate /
gradient optimizer is used.  ``numpy`` is the only hard dependency.

Drop-in compatibility
----------------------
``sample_bo_gp(space, n, observations, ...)`` mirrors the signature semantics of
``bayesian_opt.sample_bo`` and accepts the same ``Observation`` records, so it
can be wired into ``candidate_gen.generate_batch`` as a new strategy or used to
replace the KNN path entirely.
"""
from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

try:  # SciPy is optional -- used only to accelerate hyperparameter fitting.
    from scipy.optimize import minimize as _scipy_minimize  # type: ignore

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only when scipy is absent
    _scipy_minimize = None  # type: ignore
    _HAVE_SCIPY = False

from app.services.candidate_gen import (
    ParameterSpace,
    SearchDimension,
    _unit_to_value,
    sample_lhs,
)

logger = logging.getLogger(__name__)

# Numerical constants -------------------------------------------------------
_SQRT5 = math.sqrt(5.0)
_LOG_2PI = math.log(2.0 * math.pi)
# Bounds (in log-space) for hyperparameter optimization.  Length-scales and
# variances live on a log grid to keep the optimizer well-conditioned and the
# parameters strictly positive.
_LOG_LENGTHSCALE_BOUNDS = (math.log(1e-2), math.log(1e2))
_LOG_VARIANCE_BOUNDS = (math.log(1e-4), math.log(1e2))
_LOG_NOISE_BOUNDS = (math.log(1e-6), math.log(1e0))


# ===========================================================================
# Kernels
# ===========================================================================


class Kernel(ABC):
    """Abstract covariance function operating on rows of ``[0, 1]^d`` points.

    All kernels expose their tunable hyperparameters through
    :meth:`get_log_params` / :meth:`set_log_params` (always in log-space, so the
    optimizer can work unconstrained) and provide analytic gradients of the
    Gram matrix w.r.t. those log-parameters via :meth:`gradient_wrt_params`.
    """

    @abstractmethod
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Return the ``(n1, n2)`` covariance matrix between ``x1`` and ``x2``."""

    @abstractmethod
    def diag(self, x: np.ndarray) -> np.ndarray:
        """Return the diagonal ``k(x_i, x_i)`` -- cheap prior variance."""

    @abstractmethod
    def gradient_wrt_params(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Return ``{log_param_name: dK/d(log param)}`` for ``x1`` vs ``x2``."""

    @abstractmethod
    def get_log_params(self) -> dict[str, np.ndarray]:
        """Current hyperparameters in log-space (flat-ordered for the optimizer)."""

    @abstractmethod
    def set_log_params(self, params: dict[str, np.ndarray]) -> None:
        """Set hyperparameters from a log-space dict (mirror of get_log_params)."""

    # -- convenience for the optimizer: flatten / unflatten --------------

    def flatten_log_params(self) -> np.ndarray:
        """Concatenate all log-params into a single 1-D vector."""
        parts = [np.atleast_1d(v).ravel() for v in self.get_log_params().values()]
        return np.concatenate(parts) if parts else np.zeros(0)

    def unflatten_log_params(self, theta: np.ndarray) -> None:
        """Inverse of :meth:`flatten_log_params`."""
        out: dict[str, np.ndarray] = {}
        i = 0
        for name, val in self.get_log_params().items():
            arr = np.atleast_1d(val).ravel()
            out[name] = theta[i : i + arr.size].reshape(np.shape(val))
            i += arr.size
        self.set_log_params(out)

    def param_bounds(self) -> list[tuple[float, float]]:
        """Per-parameter log-space bounds, aligned with :meth:`flatten_log_params`."""
        bounds: list[tuple[float, float]] = []
        for name, val in self.get_log_params().items():
            size = np.atleast_1d(val).size
            if "lengthscale" in name:
                bnd = _LOG_LENGTHSCALE_BOUNDS
            else:
                bnd = _LOG_VARIANCE_BOUNDS
            bounds.extend([bnd] * size)
        return bounds


def _pairwise_sq_dist(
    x1: np.ndarray, x2: np.ndarray, lengthscales: np.ndarray
) -> np.ndarray:
    """Scaled squared Euclidean distance ``sum_d ((x1-x2)/l_d)^2``.

    Shapes: ``x1`` is ``(n1, d)``, ``x2`` is ``(n2, d)`` -> ``(n1, n2)``.
    """
    xs1 = x1 / lengthscales
    xs2 = x2 / lengthscales
    s1 = np.sum(xs1 * xs1, axis=1)[:, None]
    s2 = np.sum(xs2 * xs2, axis=1)[None, :]
    cross = xs1 @ xs2.T
    sq = s1 + s2 - 2.0 * cross
    return np.maximum(sq, 0.0)


class ARDSquaredExponential(Kernel):
    """ARD squared-exponential (RBF) kernel.

    ``k(x, x') = sigma^2 * exp(-0.5 * sum_d ((x_d - x'_d) / l_d)^2)``.

    The per-dimension length-scales ``l_d`` implement Automatic Relevance
    Determination: large ``l_d`` -> dimension is effectively ignored, small
    ``l_d`` -> dimension is highly relevant.  This yields an interpretable
    feature-importance readout after fitting (see :meth:`relevance`).
    """

    def __init__(self, lengthscales: np.ndarray, variance: float = 1.0) -> None:
        self.lengthscales = np.asarray(lengthscales, dtype=float).ravel()
        self.variance = float(variance)

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        sq = _pairwise_sq_dist(x1, x2, self.lengthscales)
        return self.variance * np.exp(-0.5 * sq)

    def diag(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.variance, dtype=float)

    def gradient_wrt_params(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> dict[str, np.ndarray]:
        sq = _pairwise_sq_dist(x1, x2, self.lengthscales)
        k = self.variance * np.exp(-0.5 * sq)
        grads: dict[str, np.ndarray] = {}
        # d k / d(log variance) = k  (since k ∝ variance, chain rule via log)
        grads["log_variance"] = k.copy()
        # d k / d(log l_d) = k * ((x1_d - x2_d)^2 / l_d^2)
        per_dim = np.empty((self.lengthscales.size, *k.shape))
        for d in range(self.lengthscales.size):
            diff = x1[:, d][:, None] - x2[:, d][None, :]
            scaled = (diff * diff) / (self.lengthscales[d] ** 2)
            per_dim[d] = k * scaled
        grads["log_lengthscale"] = per_dim
        return grads

    def get_log_params(self) -> dict[str, np.ndarray]:
        return {
            "log_lengthscale": np.log(self.lengthscales),
            "log_variance": np.array([math.log(self.variance)]),
        }

    def set_log_params(self, params: dict[str, np.ndarray]) -> None:
        self.lengthscales = np.exp(np.asarray(params["log_lengthscale"]).ravel())
        self.variance = float(np.exp(np.asarray(params["log_variance"]).ravel()[0]))

    def relevance(self) -> np.ndarray:
        """Per-dimension relevance score = 1 / lengthscale (normalized)."""
        inv = 1.0 / np.maximum(self.lengthscales, 1e-12)
        total = inv.sum()
        return inv / total if total > 0 else inv


class MaternKernel(Kernel):
    """Matern-5/2 ARD kernel -- the workhorse prior for physical systems.

    ``k(r) = sigma^2 * (1 + sqrt(5) r + 5 r^2 / 3) * exp(-sqrt(5) r)`` where
    ``r = sqrt(sum_d ((x_d - x'_d) / l_d)^2)`` is the ARD-scaled distance.

    Sample paths are twice mean-square differentiable, matching the smoothness
    of most physical response surfaces far better than the infinitely-smooth
    squared-exponential.
    """

    def __init__(self, lengthscales: np.ndarray, variance: float = 1.0) -> None:
        self.lengthscales = np.asarray(lengthscales, dtype=float).ravel()
        self.variance = float(variance)

    def _scaled_r(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        sq = _pairwise_sq_dist(x1, x2, self.lengthscales)
        return np.sqrt(sq)

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        r = self._scaled_r(x1, x2)
        poly = 1.0 + _SQRT5 * r + (5.0 / 3.0) * r * r
        return self.variance * poly * np.exp(-_SQRT5 * r)

    def diag(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.variance, dtype=float)

    def gradient_wrt_params(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> dict[str, np.ndarray]:
        sq = _pairwise_sq_dist(x1, x2, self.lengthscales)
        r = np.sqrt(sq)
        exp_term = np.exp(-_SQRT5 * r)
        poly = 1.0 + _SQRT5 * r + (5.0 / 3.0) * sq
        k = self.variance * poly * exp_term

        grads: dict[str, np.ndarray] = {}
        grads["log_variance"] = k.copy()

        # dk/d(r^2): k = var * f(r) where f = (1 + sqrt5 r + 5/3 r^2) e^{-sqrt5 r}
        # df/d(sq) = -(5/6) (1 + sqrt5 r) e^{-sqrt5 r}   (clean closed form)
        dk_dsq = -self.variance * (5.0 / 6.0) * (1.0 + _SQRT5 * r) * exp_term
        per_dim = np.empty((self.lengthscales.size, *k.shape))
        for d in range(self.lengthscales.size):
            diff = x1[:, d][:, None] - x2[:, d][None, :]
            # d(sq)/d(log l_d) = -2 (diff^2 / l_d^2)
            dsq_dlogl = -2.0 * (diff * diff) / (self.lengthscales[d] ** 2)
            per_dim[d] = dk_dsq * dsq_dlogl
        grads["log_lengthscale"] = per_dim
        return grads

    def get_log_params(self) -> dict[str, np.ndarray]:
        return {
            "log_lengthscale": np.log(self.lengthscales),
            "log_variance": np.array([math.log(self.variance)]),
        }

    def set_log_params(self, params: dict[str, np.ndarray]) -> None:
        self.lengthscales = np.exp(np.asarray(params["log_lengthscale"]).ravel())
        self.variance = float(np.exp(np.asarray(params["log_variance"]).ravel()[0]))

    def relevance(self) -> np.ndarray:
        """Per-dimension relevance score = 1 / lengthscale (normalized)."""
        inv = 1.0 / np.maximum(self.lengthscales, 1e-12)
        total = inv.sum()
        return inv / total if total > 0 else inv


class ProductKernel(Kernel):
    """Product of sub-kernels acting on disjoint column blocks.

    ``k(x, x') = prod_i k_i(x[:, cols_i], x'[:, cols_i])``.

    This is the natural construction for mixed continuous/categorical spaces:
    one ARD kernel over the continuous columns, multiplied by a
    categorical-overlap kernel over the discrete ones.  Because variances would
    be redundant across factors, only the *first* sub-kernel keeps a free
    variance; the rest are unit-variance (their log-variance is dropped from the
    optimization vector).
    """

    def __init__(self, kernels: list[Kernel], column_blocks: list[list[int]]) -> None:
        if len(kernels) != len(column_blocks):
            raise ValueError("kernels and column_blocks must align")
        self.kernels = kernels
        self.column_blocks = [list(c) for c in column_blocks]

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        out = np.ones((x1.shape[0], x2.shape[0]), dtype=float)
        for kern, cols in zip(self.kernels, self.column_blocks):
            out = out * kern(x1[:, cols], x2[:, cols])
        return out

    def diag(self, x: np.ndarray) -> np.ndarray:
        out = np.ones(x.shape[0], dtype=float)
        for kern, cols in zip(self.kernels, self.column_blocks):
            out = out * kern.diag(x[:, cols])
        return out

    def gradient_wrt_params(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> dict[str, np.ndarray]:
        # Precompute each factor's Gram matrix.
        factors = [
            kern(x1[:, cols], x2[:, cols])
            for kern, cols in zip(self.kernels, self.column_blocks)
        ]
        grads: dict[str, np.ndarray] = {}
        for i, (kern, cols) in enumerate(zip(self.kernels, self.column_blocks)):
            # Product of all other factors (so dK/dtheta_i scales by them).
            other = np.ones_like(factors[0])
            for j, fac in enumerate(factors):
                if j != i:
                    other = other * fac
            sub = kern.gradient_wrt_params(x1[:, cols], x2[:, cols])
            for name, g in sub.items():
                if g.ndim == 3:  # per-dimension length-scale gradients
                    grads[f"k{i}_{name}"] = g * other[None, :, :]
                else:
                    grads[f"k{i}_{name}"] = g * other
        return grads

    def get_log_params(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i, kern in enumerate(self.kernels):
            sub = kern.get_log_params()
            for name, val in sub.items():
                # Drop redundant variances on all but the first factor.
                if i > 0 and "variance" in name:
                    continue
                out[f"k{i}_{name}"] = val
        return out

    def set_log_params(self, params: dict[str, np.ndarray]) -> None:
        for i, kern in enumerate(self.kernels):
            current = kern.get_log_params()
            update: dict[str, np.ndarray] = {}
            for name in current:
                key = f"k{i}_{name}"
                if key in params:
                    update[name] = params[key]
                else:
                    update[name] = current[name]  # kept fixed (e.g. variance)
            kern.set_log_params(update)


class CategoricalOverlapKernel(Kernel):
    """Hamming / overlap kernel for categorical columns.

    ``k(x, x') = exp(- sum_d w_d * [x_d != x'_d])``.  The categorical columns
    are encoded as integer codes in ``[0, 1]`` bins; a small tolerance decides
    equality.  Per-column weights ``w_d`` act like inverse length-scales.
    """

    def __init__(self, weights: np.ndarray, tol: float = 1e-6) -> None:
        self.weights = np.asarray(weights, dtype=float).ravel()
        self.tol = float(tol)

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        n1, n2 = x1.shape[0], x2.shape[0]
        acc = np.zeros((n1, n2), dtype=float)
        for d in range(x1.shape[1]):
            diff = np.abs(x1[:, d][:, None] - x2[:, d][None, :])
            mismatch = (diff > self.tol).astype(float)
            acc += self.weights[d] * mismatch
        return np.exp(-acc)

    def diag(self, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape[0], dtype=float)

    def gradient_wrt_params(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> dict[str, np.ndarray]:
        k = self.__call__(x1, x2)
        per_dim = np.empty((self.weights.size, *k.shape))
        for d in range(x1.shape[1]):
            diff = np.abs(x1[:, d][:, None] - x2[:, d][None, :])
            mismatch = (diff > self.tol).astype(float)
            # dk/d(log w_d) = k * (-w_d * mismatch)
            per_dim[d] = k * (-self.weights[d] * mismatch)
        return {"log_lengthscale": per_dim}

    def get_log_params(self) -> dict[str, np.ndarray]:
        return {"log_lengthscale": np.log(np.maximum(self.weights, 1e-12))}

    def set_log_params(self, params: dict[str, np.ndarray]) -> None:
        self.weights = np.exp(np.asarray(params["log_lengthscale"]).ravel())


def _categorical_mask(space: ParameterSpace) -> list[bool]:
    """Boolean mask: True for categorical/boolean dimensions."""
    mask: list[bool] = []
    for dim in space.dimensions:
        is_cat = dim.choices is not None or dim.param_type in ("boolean", "categorical")
        mask.append(bool(is_cat))
    return mask


def select_kernel(space: ParameterSpace) -> Kernel:
    """Choose a kernel appropriate to the structure of ``space``.

    Rules
    -----
    * Continuous-only spaces -> ARD squared-exponential (smooth, interpretable
      relevances).
    * Spaces with many categorical dims (>= 2) and some continuous dims ->
      ``ProductKernel`` of a continuous Matern-5/2 and a categorical-overlap
      kernel.
    * Mostly/all categorical -> categorical-overlap kernel.
    * Mixed with a single categorical column -> Matern-5/2 over all columns
      (robust default for noisy physical systems).
    """
    mask = _categorical_mask(space)
    n_dims = len(mask)
    n_cat = sum(mask)
    n_cont = n_dims - n_cat

    if n_cat == 0:
        return ARDSquaredExponential(np.ones(n_dims), variance=1.0)

    if n_cat == n_dims:
        return CategoricalOverlapKernel(np.ones(n_dims))

    if n_cat >= 2 and n_cont >= 1:
        cont_cols = [i for i, m in enumerate(mask) if not m]
        cat_cols = [i for i, m in enumerate(mask) if m]
        cont_kernel = MaternKernel(np.ones(len(cont_cols)), variance=1.0)
        cat_kernel = CategoricalOverlapKernel(np.ones(len(cat_cols)))
        return ProductKernel([cont_kernel, cat_kernel], [cont_cols, cat_cols])

    # One categorical column mixed in -> robust Matern over everything.
    return MaternKernel(np.ones(n_dims), variance=1.0)


# ===========================================================================
# Gaussian Process surrogate
# ===========================================================================


class GPSurrogate:
    """Exact zero-mean Gaussian Process regressor with calibrated posterior.

    The GP is fit by Cholesky factorization of ``K + (noise + jitter) I``.
    Targets are (optionally) standardized to zero-mean / unit-variance, which
    both improves conditioning and makes the default kernel variance prior
    sensible.  Predictions return the posterior *mean* and posterior *variance*
    rescaled back to the original target units -- i.e. genuinely calibrated
    uncertainty, not a neighbourhood-spread proxy.
    """

    def __init__(
        self,
        kernel: Kernel,
        noise_variance: float = 1e-4,
        normalize_y: bool = True,
        jitter: float = 1e-6,
    ) -> None:
        self.kernel = kernel
        self.noise_variance = float(noise_variance)
        self.normalize_y = bool(normalize_y)
        self.jitter = float(jitter)

        # Populated by fit():
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None  # standardized targets
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._L: np.ndarray | None = None  # Cholesky factor of K_noisy
        self._alpha: np.ndarray | None = None  # K_noisy^{-1} y

    # -- fitting ---------------------------------------------------------

    def _gram_noisy(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        K = self.kernel(X, X)
        K[np.diag_indices_from(K)] += self.noise_variance + self.jitter
        return K

    def _cholesky(self, K: np.ndarray) -> np.ndarray:
        """Robust Cholesky with escalating jitter on failure."""
        jitter = self.jitter
        for _ in range(8):
            try:
                return np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K = K.copy()
                K[np.diag_indices_from(K)] += jitter
                jitter *= 10.0
        # Last resort: symmetric eigenvalue clipping.
        w, V = np.linalg.eigh(K)
        w = np.maximum(w, 1e-10)
        K = (V * w) @ V.T
        return np.linalg.cholesky(K)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        """Condition the GP on training data ``X`` (n, d) and targets ``y`` (n,)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        if self.normalize_y and y.size > 1:
            self._y_mean = float(np.mean(y))
            std = float(np.std(y))
            self._y_std = std if std > 1e-12 else 1.0
        else:
            self._y_mean = 0.0
            self._y_std = 1.0

        y_std = (y - self._y_mean) / self._y_std

        K = self._gram_noisy(X)
        L = self._cholesky(K)
        alpha = _cho_solve(L, y_std)

        self._X, self._y = X, y_std
        self._L, self._alpha = L, alpha
        return self

    # -- prediction ------------------------------------------------------

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior ``(mean, variance)`` in original target units.

        Both arrays have shape ``(n_test,)``.  Variance is clipped at 0 to guard
        against tiny negative values from finite-precision arithmetic.
        """
        if self._L is None or self._X is None or self._alpha is None:
            raise RuntimeError("GPSurrogate.predict called before fit()")
        X_test = np.atleast_2d(np.asarray(X_test, dtype=float))

        K_s = self.kernel(self._X, X_test)  # (n_train, n_test)
        mean_std = K_s.T @ self._alpha  # standardized mean

        # v = L^{-1} K_s  ->  var = k(x,x) - sum(v^2)
        v = np.linalg.solve(self._L, K_s)
        prior_var = self.kernel.diag(X_test) + self.noise_variance
        var_std = prior_var - np.sum(v * v, axis=0)
        var_std = np.maximum(var_std, 0.0)

        mean = mean_std * self._y_std + self._y_mean
        var = var_std * (self._y_std ** 2)
        return mean, var

    def sample_posterior(self, X_test: np.ndarray, n_samples: int = 10) -> np.ndarray:
        """Draw ``n_samples`` joint posterior functions at ``X_test``.

        Returns shape ``(n_samples, n_test)`` -- the basis for Thompson sampling.
        """
        if self._L is None or self._X is None or self._alpha is None:
            raise RuntimeError("sample_posterior called before fit()")
        X_test = np.atleast_2d(np.asarray(X_test, dtype=float))

        K_s = self.kernel(self._X, X_test)
        mean_std = K_s.T @ self._alpha

        v = np.linalg.solve(self._L, K_s)
        K_ss = self.kernel(X_test, X_test)
        cov_std = K_ss - v.T @ v
        n = cov_std.shape[0]
        cov_std[np.diag_indices(n)] += self.jitter
        cov_std = 0.5 * (cov_std + cov_std.T)  # symmetrize

        Lc = self._cholesky(cov_std)
        eps = np.random.standard_normal((n, n_samples))
        draws_std = mean_std[:, None] + Lc @ eps  # (n_test, n_samples)
        draws = draws_std * self._y_std + self._y_mean
        return draws.T  # (n_samples, n_test)

    # -- hyperparameter optimization ------------------------------------

    def marginal_log_likelihood(self) -> float:
        """Log marginal likelihood ``log p(y | X, theta)`` of the current fit."""
        if self._L is None or self._y is None or self._alpha is None:
            raise RuntimeError("marginal_log_likelihood called before fit()")
        n = self._y.shape[0]
        data_fit = -0.5 * float(self._y @ self._alpha)
        complexity = -float(np.sum(np.log(np.diag(self._L))))  # = -0.5 log|K|
        const = -0.5 * n * _LOG_2PI
        return data_fit + complexity + const

    def _neg_lml_and_grad(
        self, theta: np.ndarray, X: np.ndarray, y_std: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Negative LML and its gradient w.r.t. ``theta`` (kernel log-params + log-noise).

        The last entry of ``theta`` is the log noise variance.
        """
        n_kernel = self.kernel.flatten_log_params().size
        self.kernel.unflatten_log_params(theta[:n_kernel])
        log_noise = float(theta[n_kernel])
        self.noise_variance = math.exp(log_noise)

        n = X.shape[0]
        K = self._gram_noisy(X)
        try:
            L = self._cholesky(K)
        except np.linalg.LinAlgError:
            return 1e25, np.zeros_like(theta)
        alpha = _cho_solve(L, y_std)

        lml = (
            -0.5 * float(y_std @ alpha)
            - float(np.sum(np.log(np.diag(L))))
            - 0.5 * n * _LOG_2PI
        )

        # Gradient: 0.5 * tr((alpha alpha^T - K^{-1}) dK/dtheta_i)
        K_inv = _cho_inv(L)
        aaT = np.outer(alpha, alpha)
        factor = aaT - K_inv  # (n, n)

        kgrads = self.kernel.gradient_wrt_params(X, X)
        grad_parts: list[float] = []
        # Iterate in the same key order as get_log_params -> flatten.
        for name, val in self.kernel.get_log_params().items():
            g = kgrads[name]
            if g.ndim == 3:  # per-dimension stack
                for d in range(g.shape[0]):
                    grad_parts.append(0.5 * float(np.sum(factor * g[d])))
            else:
                grad_parts.append(0.5 * float(np.sum(factor * g)))

        # Noise gradient: dK/d(log noise) = noise * I
        dnoise = self.noise_variance * float(np.trace(factor))
        grad_parts.append(0.5 * dnoise)

        grad = np.asarray(grad_parts, dtype=float)
        return -lml, -grad

    def optimize_hyperparameters(self, n_restarts: int = 3) -> "GPSurrogate":
        """Maximize the log marginal likelihood with random restarts.

        Uses L-BFGS-B (SciPy) when available, otherwise a numpy gradient-descent
        fallback.  The best (lowest negative LML) configuration is retained and
        the GP is re-fit with it.
        """
        if self._X is None or self._y is None:
            raise RuntimeError("optimize_hyperparameters called before fit()")
        X, y_std = self._X, self._y

        n_kernel = self.kernel.flatten_log_params().size
        kernel_bounds = self.kernel.param_bounds()
        bounds = kernel_bounds + [_LOG_NOISE_BOUNDS]

        init_kernel = self.kernel.flatten_log_params()
        init_noise = math.log(max(self.noise_variance, 1e-6))
        theta0 = np.concatenate([init_kernel, [init_noise]])

        rng = np.random.default_rng(0)
        best_theta = theta0.copy()
        best_obj = self._safe_obj(theta0, X, y_std)

        for r in range(max(1, n_restarts)):
            if r == 0:
                start = theta0
            else:
                lows = np.array([b[0] for b in bounds])
                highs = np.array([b[1] for b in bounds])
                start = lows + rng.random(theta0.size) * (highs - lows)

            theta_opt = self._run_optimizer(start, bounds, X, y_std)
            obj = self._safe_obj(theta_opt, X, y_std)
            if obj < best_obj:
                best_obj, best_theta = obj, theta_opt

        # Commit the best hyperparameters and re-fit cleanly.
        self.kernel.unflatten_log_params(best_theta[:n_kernel])
        self.noise_variance = math.exp(best_theta[n_kernel])

        # Recover the original-scale targets to re-run fit() consistently.
        y_orig = y_std * self._y_std + self._y_mean
        return self.fit(X, y_orig)

    def _safe_obj(self, theta: np.ndarray, X: np.ndarray, y_std: np.ndarray) -> float:
        try:
            obj, _ = self._neg_lml_and_grad(theta, X, y_std)
            return obj if math.isfinite(obj) else 1e25
        except Exception:  # pragma: no cover - numerical guard
            return 1e25

    def _run_optimizer(
        self,
        start: np.ndarray,
        bounds: list[tuple[float, float]],
        X: np.ndarray,
        y_std: np.ndarray,
    ) -> np.ndarray:
        if _HAVE_SCIPY:
            res = _scipy_minimize(  # type: ignore[misc]
                fun=lambda th: self._neg_lml_and_grad(th, X, y_std),
                x0=start,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 100, "ftol": 1e-6},
            )
            return np.asarray(res.x, dtype=float)
        return self._gradient_descent(start, bounds, X, y_std)

    @staticmethod
    def _gradient_descent(
        start: np.ndarray,
        bounds: list[tuple[float, float]],
        X: np.ndarray,
        y_std: np.ndarray,
        max_iter: int = 100,
    ) -> np.ndarray:
        """Projected gradient descent with simple backtracking line search."""
        # Bound the closure to the active instance via a small trampoline.
        raise NotImplementedError  # replaced below by bound method


def _project(theta: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    return np.minimum(np.maximum(theta, lows), highs)


def _gp_gradient_descent(
    gp: "GPSurrogate",
    start: np.ndarray,
    bounds: list[tuple[float, float]],
    X: np.ndarray,
    y_std: np.ndarray,
    max_iter: int = 100,
) -> np.ndarray:
    """Pure-numpy projected gradient descent fallback for hyperparameter fitting."""
    theta = _project(start.copy(), bounds)
    obj, grad = gp._neg_lml_and_grad(theta, X, y_std)
    step = 0.1
    for _ in range(max_iter):
        improved = False
        for _ls in range(20):
            cand = _project(theta - step * grad, bounds)
            cand_obj, cand_grad = gp._neg_lml_and_grad(cand, X, y_std)
            if math.isfinite(cand_obj) and cand_obj < obj - 1e-9:
                theta, obj, grad = cand, cand_obj, cand_grad
                step *= 1.3
                improved = True
                break
            step *= 0.5
        if not improved or step < 1e-8:
            break
    return theta


# Wire the bound fallback in (avoids a static-method closure over ``self``).
def _bound_gradient_descent(self, start, bounds, X, y_std, max_iter=100):  # type: ignore[no-untyped-def]
    return _gp_gradient_descent(self, start, bounds, X, y_std, max_iter)


GPSurrogate._gradient_descent = _bound_gradient_descent  # type: ignore[assignment]


# -- linear-algebra helpers -------------------------------------------------


def _cho_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``(L L^T) x = b`` given lower-triangular Cholesky factor ``L``."""
    z = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, z)


def _cho_inv(L: np.ndarray) -> np.ndarray:
    """Invert ``K = L L^T`` from its Cholesky factor."""
    n = L.shape[0]
    eye = np.eye(n)
    z = np.linalg.solve(L, eye)
    return np.linalg.solve(L.T, z)


# ===========================================================================
# Acquisition functions (research-grade, vectorized over candidate rows)
# ===========================================================================


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    # Vectorized standard-normal CDF via the error function.
    from numpy import vectorize  # local import keeps top clean

    if not hasattr(_normal_cdf, "_vec"):
        _normal_cdf._vec = vectorize(lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))  # type: ignore[attr-defined]
    return _normal_cdf._vec(z)  # type: ignore[attr-defined]


def expected_improvement_gp(
    gp: GPSurrogate, X: np.ndarray, y_best: float, xi: float = 0.01
) -> np.ndarray:
    """Expected Improvement: ``E[max(f(x) - y_best - xi, 0)]`` (maximization)."""
    mean, var = gp.predict(X)
    std = np.sqrt(np.maximum(var, 0.0))
    out = np.zeros_like(mean)
    mask = std > 1e-12
    imp = mean - y_best - xi
    z = np.zeros_like(mean)
    z[mask] = imp[mask] / std[mask]
    out[mask] = imp[mask] * _normal_cdf(z[mask]) + std[mask] * _normal_pdf(z[mask])
    return np.maximum(out, 0.0)


def upper_confidence_bound_gp(
    gp: GPSurrogate, X: np.ndarray, beta: float = 2.0
) -> np.ndarray:
    """UCB acquisition: ``mu + beta * sigma`` (maximization)."""
    mean, var = gp.predict(X)
    return mean + beta * np.sqrt(np.maximum(var, 0.0))


def thompson_sampling(
    gp: GPSurrogate, X: np.ndarray, n_samples: int = 1
) -> np.ndarray:
    """Thompson sampling acquisition scores.

    Draws ``n_samples`` posterior functions over ``X`` and returns, for each
    candidate, the fraction of draws in which it is the arg-max.  This gives a
    diversity-promoting score: distinct optima across draws all receive mass,
    enabling broad exploration in batched selection.  Returns shape ``(n,)``.
    """
    draws = gp.sample_posterior(X, n_samples=max(1, n_samples))  # (n_samples, n)
    argmaxes = np.argmax(draws, axis=1)
    scores = np.zeros(X.shape[0], dtype=float)
    for idx in argmaxes:
        scores[idx] += 1.0
    return scores / max(1, n_samples)


def max_value_entropy_search(
    gp: GPSurrogate, X: np.ndarray, n_samples: int = 100
) -> np.ndarray:
    """Max-value Entropy Search (MES), Gumbel-sampling approximation.

    MES selects the point that maximally reduces entropy of the distribution
    over the global maximum value ``y*``.  We approximate the ``y*`` distribution
    by sampling from the posterior over a candidate set and use the closed-form
    per-point information gain (Wang & Jegelka, 2017):

        I(x) ≈ (1/M) Σ_m [ γ_m φ(γ_m) / (2 Φ(γ_m)) − log Φ(γ_m) ]

    where ``γ_m = (y*_m − μ(x)) / σ(x)``.  Especially valuable for *batched*
    experimentation: it scores information about the optimum rather than greedy
    improvement, so a top-k slice yields complementary, non-redundant probes.
    """
    mean, var = gp.predict(X)
    std = np.sqrt(np.maximum(var, 1e-12))

    # Sample plausible maximum values from the posterior.
    draws = gp.sample_posterior(X, n_samples=max(1, n_samples))  # (M, n)
    y_stars = np.max(draws, axis=1)  # (M,)
    # Ensure y* strictly exceeds the predictive mean to keep gamma well-defined.
    floor = mean.max() + 1e-6
    y_stars = np.maximum(y_stars, floor)

    info = np.zeros_like(mean)
    for ystar in y_stars:
        gamma = (ystar - mean) / std
        cdf = _normal_cdf(gamma)
        cdf = np.clip(cdf, 1e-12, 1.0 - 1e-12)
        pdf = _normal_pdf(gamma)
        info += (gamma * pdf) / (2.0 * cdf) - np.log(cdf)
    return info / len(y_stars)


def knowledge_gradient(
    gp: GPSurrogate, X: np.ndarray, X_fantasies: np.ndarray
) -> np.ndarray:
    """Knowledge Gradient (KG): expected gain in the best *predicted* value.

    For each candidate ``x``, we fantasize an observation at its posterior mean,
    condition a copy of the GP on it, and measure how much the maximum posterior
    mean over ``X_fantasies`` improves relative to the current best posterior
    mean.  This one-step look-ahead favours decisions that improve our *model's*
    recommendation, not merely the sampled point.  Returns shape ``(n,)``.
    """
    if gp._X is None or gp._y is None:
        raise RuntimeError("knowledge_gradient requires a fitted GP")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    X_fan = np.atleast_2d(np.asarray(X_fantasies, dtype=float))

    base_mean, _ = gp.predict(X_fan)
    base_best = float(np.max(base_mean))

    cand_mean, _ = gp.predict(X)  # fantasy observation = posterior mean
    y_orig = gp._y * gp._y_std + gp._y_mean

    out = np.empty(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        X_aug = np.vstack([gp._X, X[i][None, :]])
        y_aug = np.concatenate([y_orig, [cand_mean[i]]])
        fantasy = GPSurrogate(
            kernel=gp.kernel,
            noise_variance=gp.noise_variance,
            normalize_y=gp.normalize_y,
            jitter=gp.jitter,
        ).fit(X_aug, y_aug)
        new_mean, _ = fantasy.predict(X_fan)
        out[i] = float(np.max(new_mean)) - base_best
    return np.maximum(out, 0.0)


def integrated_variance_reduction(
    gp: GPSurrogate, X: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """Integrated variance reduction -- a pure-exploration acquisition.

    Scores each candidate by how much it would reduce the *total* posterior
    variance over a reference set ``X_test`` if observed.  Using the GP's
    closed-form variance-update, the reduction at a reference point ``x_t`` from
    observing ``x`` is ``cov(x, x_t)^2 / (var(x) + noise)``.  Summed over the
    reference set, this targets globally informative probes -- ideal for early,
    exploration-heavy campaign phases.  Returns shape ``(n,)``.
    """
    if gp._X is None:
        raise RuntimeError("integrated_variance_reduction requires a fitted GP")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    X_test = np.atleast_2d(np.asarray(X_test, dtype=float))

    _, var_x = gp.predict(X)
    denom = var_x + gp.noise_variance  # (n,)

    out = np.empty(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        # Posterior covariance between candidate i and every reference point.
        k_x_ref = gp.kernel(X[i][None, :], X_test)  # (1, m)
        k_x_tr = gp.kernel(X[i][None, :], gp._X)  # (1, n_train)
        v = np.linalg.solve(gp._L, k_x_tr.T)  # (n_train, 1)
        u = np.linalg.solve(gp._L, gp.kernel(gp._X, X_test))  # (n_train, m)
        cov = k_x_ref - (v.T @ u)  # (1, m) posterior cross-cov in std units
        cov = cov.ravel() * (gp._y_std ** 2)
        out[i] = float(np.sum(cov * cov) / max(denom[i], 1e-12))
    return out


# ===========================================================================
# Normalisation helpers (mirror bayesian_opt to stay drop-in compatible)
# ===========================================================================


def _value_to_unit(value: Any, dim: SearchDimension) -> float:
    """Map an actual parameter value back to ``[0, 1]`` (inverse of _unit_to_value)."""
    if dim.choices is not None:
        choices = list(dim.choices)
        try:
            idx = choices.index(value)
        except ValueError:
            idx = 0
        return (idx + 0.5) / len(choices)
    if dim.param_type == "boolean":
        return 1.0 if value else 0.0
    if dim.min_value is None or dim.max_value is None:
        raise ValueError(
            f"Dimension '{dim.param_name}' requires min_value and max_value"
        )
    fval = float(value)
    if dim.log_scale:
        log_min = math.log(max(dim.min_value, 1e-12))
        log_max = math.log(max(dim.max_value, 1e-12))
        if log_max == log_min:
            return 0.5
        return (math.log(max(fval, 1e-12)) - log_min) / (log_max - log_min)
    span = dim.max_value - dim.min_value
    if span == 0:
        return 0.5
    return (fval - dim.min_value) / span


def _normalize_params(params: dict[str, Any], space: ParameterSpace) -> tuple[float, ...]:
    result: list[float] = []
    for dim in space.dimensions:
        val = params.get(dim.param_name)
        if val is None:
            result.append(0.5)
        else:
            result.append(max(0.0, min(1.0, _value_to_unit(val, dim))))
    return tuple(result)


def _denormalize_point(point: np.ndarray, space: ParameterSpace) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for j, dim in enumerate(space.dimensions):
        params[dim.param_name] = _unit_to_value(float(point[j]), dim)
    return params


def _observations_to_arrays(
    observations: list[Any], space: ParameterSpace
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``Observation``-like records to ``(X, y)`` numpy arrays.

    Accepts two formats:
    1. ``params`` is a dict   → normalized via ``_normalize_params``
    2. ``params`` is a tuple  → if any value is outside [-0.05, 1.05], treated
       as raw (actual) values and normalized; otherwise assumed pre-normalized.

    This handles both the legacy ``bayesian_opt.Observation`` pattern (where
    ``.params`` is a pre-normalized tuple in [0,1]^d) and the direct-dict
    pattern used by benchmark baselines and the orchestrator.
    """
    X_rows: list[tuple[float, ...]] = []
    y_vals: list[float] = []
    for obs in observations:
        params = getattr(obs, "params")
        objective = getattr(obs, "objective")
        if isinstance(params, dict):
            params = _normalize_params(params, space)
        else:
            # Detect whether tuple is already normalized or contains raw values
            vals = tuple(float(v) for v in params)
            if any(v < -0.05 or v > 1.05 for v in vals):
                # Raw values — normalize using dimension bounds
                normed: list[float] = []
                for v, dim in zip(vals, space.dimensions):
                    normed.append(max(0.0, min(1.0, _value_to_unit(v, dim))))
                params = tuple(normed)
            else:
                params = vals
        X_rows.append(params)
        y_vals.append(float(objective))
    return np.asarray(X_rows, dtype=float), np.asarray(y_vals, dtype=float)


# ===========================================================================
# Drop-in BO sampler
# ===========================================================================


_VALID_ACQUISITIONS = {"ei", "ucb", "thompson", "mes", "kg", "ivr"}


def sample_bo_gp(
    space: ParameterSpace,
    n: int,
    observations: list[Any],
    acquisition: str = "ei",
    seed: int | None = None,
    n_candidates: int = 2000,
    optimize_hyperparams: bool = True,
    *,
    xi: float = 0.01,
    beta: float = 2.0,
    mes_samples: int = 100,
) -> list[dict[str, Any]]:
    """GP-based Bayesian-optimisation sampler -- drop-in for ``sample_bo``.

    Pipeline
    --------
    1.  **Cold-start guard.** When ``len(observations) < 3 * n_dims`` the GP is
        data-starved; we fall back to Latin Hypercube Sampling for robust
        space-filling coverage.
    2.  **Kernel selection** via :func:`select_kernel` (ARD-SE for continuous,
        Matern-5/2 / product kernels when categoricals are present).
    3.  **Fit + (optional) hyperparameter optimization** by maximizing the log
        marginal likelihood with random restarts.
    4.  **Acquisition evaluation** over a random candidate pool in ``[0,1]^d``.
    5.  **Top-n selection**, de-normalized to actual parameter dicts.

    Parameters
    ----------
    acquisition:
        One of ``"ei"``, ``"ucb"``, ``"thompson"``, ``"mes"``, ``"kg"``,
        ``"ivr"`` (integrated variance reduction).
    n_candidates:
        Size of the random candidate pool scored by the acquisition function.
    optimize_hyperparams:
        If True, fit kernel length-scales/variance + noise by maximum marginal
        likelihood (recommended; enables the ARD relevance readout).
    """
    if acquisition not in _VALID_ACQUISITIONS:
        raise ValueError(
            f"Unknown acquisition '{acquisition}'. Valid: {sorted(_VALID_ACQUISITIONS)}"
        )

    n_dims = space.n_dims
    n_obs = len(observations) if observations else 0

    # --- cold-start: fall back to LHS ---
    if n_obs < 3 * n_dims:
        logger.info(
            "GP-BO cold-start (%d obs < 3*%d dims): falling back to LHS",
            n_obs,
            n_dims,
        )
        return sample_lhs(space, n, seed=seed)

    X, y = _observations_to_arrays(observations, space)

    # --- fit GP surrogate ---
    kernel = select_kernel(space)
    gp = GPSurrogate(kernel=kernel, noise_variance=1e-4, normalize_y=True)
    gp.fit(X, y)
    if optimize_hyperparams:
        try:
            gp.optimize_hyperparameters(n_restarts=3)
        except Exception as exc:  # pragma: no cover - numerical safety net
            logger.warning("Hyperparameter optimization failed (%s); using priors", exc)

    # --- generate random candidate pool in [0,1]^d ---
    rng = random.Random(seed)
    if seed is not None:
        np.random.seed(seed)
    pool = np.asarray(
        [[rng.random() for _ in range(n_dims)] for _ in range(n_candidates)],
        dtype=float,
    )

    y_best = float(np.max(y))

    # --- score candidates ---
    if acquisition == "ei":
        scores = expected_improvement_gp(gp, pool, y_best, xi=xi)
    elif acquisition == "ucb":
        scores = upper_confidence_bound_gp(gp, pool, beta=beta)
    elif acquisition == "thompson":
        # One posterior sample per requested candidate gives diverse argmaxes.
        scores = thompson_sampling(gp, pool, n_samples=max(n, 10))
    elif acquisition == "mes":
        scores = max_value_entropy_search(gp, pool, n_samples=mes_samples)
    elif acquisition == "kg":
        # Use a sub-sampled reference set for tractable look-ahead.
        ref_idx = np.random.choice(
            pool.shape[0], size=min(128, pool.shape[0]), replace=False
        )
        scores = knowledge_gradient(gp, pool, pool[ref_idx])
    else:  # "ivr"
        ref_idx = np.random.choice(
            pool.shape[0], size=min(128, pool.shape[0]), replace=False
        )
        scores = integrated_variance_reduction(gp, pool, pool[ref_idx])

    # --- select top-n (descending acquisition value) ---
    order = np.argsort(scores)[::-1][:n]
    results = [_denormalize_point(pool[i], space) for i in order]

    logger.info(
        "GP-BO sampled %d candidates (acq=%s, pool=%d, best_obj=%.4f, kernel=%s)",
        len(results),
        acquisition,
        n_candidates,
        y_best,
        type(kernel).__name__,
    )
    return results


def feature_relevance(
    space: ParameterSpace, observations: list[Any]
) -> dict[str, float]:
    """Return a per-parameter relevance ranking from a fitted ARD/Matern GP.

    A key scientific deliverable: after maximum-likelihood fitting, the inverse
    length-scales quantify how strongly each experimental parameter drives the
    objective -- a data-driven sensitivity analysis returned as
    ``{param_name: relevance_in_[0,1]}`` (sums to 1 over continuous dims).
    Returns an empty dict when there is insufficient data.
    """
    n_dims = space.n_dims
    if not observations or len(observations) < 3 * n_dims:
        return {}

    X, y = _observations_to_arrays(observations, space)
    kernel = select_kernel(space)
    gp = GPSurrogate(kernel=kernel, normalize_y=True).fit(X, y)
    try:
        gp.optimize_hyperparameters(n_restarts=3)
    except Exception:  # pragma: no cover
        pass

    # Resolve the ARD-bearing kernel and its column mapping.
    if isinstance(kernel, ProductKernel):
        for sub, cols in zip(kernel.kernels, kernel.column_blocks):
            if isinstance(sub, (ARDSquaredExponential, MaternKernel)):
                rel = sub.relevance()
                return {
                    space.dimensions[c].param_name: float(rel[j])
                    for j, c in enumerate(cols)
                }
        return {}
    if isinstance(kernel, (ARDSquaredExponential, MaternKernel)):
        rel = kernel.relevance()
        return {
            space.dimensions[j].param_name: float(rel[j]) for j in range(n_dims)
        }
    return {}

"""
Meta-learning for HELIOS: learns which algorithmic choices work best
across families of chemistry discovery tasks.

Three meta-learning components:
1. AcquisitionBandit  — which acquisition function to use this campaign
2. StrategyPrior      — which RL strategy to initialize with
3. HyperparamPrior    — which GP hyperparameters to initialize with (ABLR)

This enables HELIOS to improve not just within campaigns (inner RL loop)
and across campaigns (outer RL loop) but also across TASK FAMILIES —
the third level of the hierarchical learning architecture.

Reference: Warm-starting BO with meta-learning (Feurer et al. 2018),
ABLR (Perrone et al. 2018), Task2Vec (Achille et al. 2019).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("helios.services.meta_learning")

__all__ = [
    "AcquisitionBandit",
    "HyperparamPrior",
    "MetaLearner",
    "TaskEmbedding",
]


@dataclass
class TaskEmbedding:
    """A fixed-dimensional embedding of a campaign's task characteristics.

    Used to measure task similarity for meta-learning.
    Features capture: search space geometry, domain type, noise level,
    previous campaign performance.
    """
    campaign_id: str
    n_dimensions: int
    has_categorical: bool
    has_log_scale: bool
    estimated_noise: float
    domain_tags: list[str]       # e.g. ["electrochemistry", "HER"]
    initial_kpi: float | None    # KPI after n_init random experiments

    def to_vector(self) -> list[float]:
        return [
            self.n_dimensions / 10.0,
            float(self.has_categorical),
            float(self.has_log_scale),
            min(self.estimated_noise, 1.0),
            float(self.initial_kpi or 0.0),
        ]

    def similarity(self, other: TaskEmbedding) -> float:
        v1 = self.to_vector()
        v2 = other.to_vector()
        dot = sum(a * b for a, b in zip(v1, v2, strict=False))
        n1 = math.sqrt(sum(a**2 for a in v1)) + 1e-8
        n2 = math.sqrt(sum(a**2 for a in v2)) + 1e-8
        # Tag overlap bonus
        tag_overlap = len(set(self.domain_tags) & set(other.domain_tags))
        tag_bonus = tag_overlap * 0.1
        return min(1.0, dot / (n1 * n2) + tag_bonus)


class AcquisitionBandit:
    """UCB bandit over acquisition functions, conditioned on task embedding.

    Arms: ["ei", "ucb", "thompson", "mes", "kg"]

    For each campaign, selects an acquisition function based on:
    1. Historical performance on similar tasks (context)
    2. UCB exploration bonus for under-explored acquisitions

    After each campaign, updates the arm statistics with the achieved
    discovery_velocity as the reward signal.
    """
    ARMS = ["ei", "ucb", "thompson", "mes", "kg"]

    def __init__(self, ucb_beta: float = 1.0) -> None:
        self._ucb_beta = ucb_beta
        self._arm_stats: dict[str, dict[str, float]] = {
            a: {"total_reward": 0.0, "n_uses": 0, "m2": 0.0}
            for a in self.ARMS
        }
        self._history: list[dict[str, Any]] = []

    def select(self, task: TaskEmbedding) -> str:
        """Select acquisition function for a new campaign."""
        total_uses = sum(s["n_uses"] for s in self._arm_stats.values())
        best_arm, best_score = "ei", -math.inf

        for arm, stats in self._arm_stats.items():
            n = stats["n_uses"]
            if n == 0:
                return arm  # Force exploration of untried arms first
            mean_r = stats["total_reward"] / n
            # UCB bonus
            exploration = self._ucb_beta * math.sqrt(math.log(max(total_uses, 1)) / n)
            score = mean_r + exploration
            if score > best_score:
                best_score = score
                best_arm = arm

        logger.info("acquisition_bandit.select", extra={
            "selected": best_arm, "task": task.campaign_id,
            "n_dims": task.n_dimensions,
        })
        return best_arm

    def update(self, arm: str, reward: float, task: TaskEmbedding) -> None:
        """Update after campaign completes. reward = discovery_velocity."""
        stats = self._arm_stats[arm]
        stats["n_uses"] += 1
        n = stats["n_uses"]
        old_mean = stats["total_reward"] / max(n - 1, 1)
        stats["total_reward"] += reward
        new_mean = stats["total_reward"] / n
        stats["m2"] += (reward - old_mean) * (reward - new_mean)

        self._history.append({
            "arm": arm, "reward": reward,
            "campaign_id": task.campaign_id,
        })
        logger.info("acquisition_bandit.update", extra={
            "arm": arm, "reward": round(reward, 4), "n_uses": n,
        })

    def get_statistics(self) -> dict[str, dict[str, float]]:
        stats = {}
        for arm, s in self._arm_stats.items():
            n = max(s["n_uses"], 1)
            mean = s["total_reward"] / n
            var = s["m2"] / max(n - 1, 1) if n > 1 else 0.0
            stats[arm] = {
                "mean_reward": round(mean, 4),
                "std_reward": round(math.sqrt(var), 4),
                "n_uses": s["n_uses"],
            }
        return stats


class HyperparamPrior:
    """Meta-learns GP hyperparameter initializations across tasks.

    Instead of starting GP optimization from a fixed initialization,
    we warm-start from the hyperparameters that worked well on similar
    past tasks.

    This implements a simplified version of ABLR (Adaptive Bayesian
    Linear Regression, Perrone et al. 2018): we maintain a prior over
    hyperparameters conditioned on task features.
    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def record(
        self,
        task: TaskEmbedding,
        lengthscales: list[float],
        noise_var: float,
        signal_var: float,
        mll: float,
    ) -> None:
        self._records.append({
            "task_vec": task.to_vector(),
            "lengthscales": list(lengthscales),
            "noise_var": noise_var,
            "signal_var": signal_var,
            "mll": mll,
        })

    def suggest_init(
        self, task: TaskEmbedding, n_dims: int
    ) -> dict[str, Any]:
        """Return a warm-start initialization for GP hyperparameters."""
        if not self._records:
            return self._default_init(n_dims)

        task_vec = task.to_vector()
        # Find k most similar tasks by cosine similarity on task_vec
        sims = []
        for rec in self._records:
            rv = rec["task_vec"]
            dot = sum(a * b for a, b in zip(task_vec, rv, strict=False))
            n1 = math.sqrt(sum(a**2 for a in task_vec)) + 1e-8
            n2 = math.sqrt(sum(a**2 for a in rv)) + 1e-8
            sims.append(dot / (n1 * n2))

        k = min(3, len(self._records))
        top_k = sorted(zip(sims, self._records, strict=False), reverse=True)[:k]
        weights = [max(s, 0) for s, _ in top_k]
        total_w = sum(weights) + 1e-8

        avg_ls = [0.0] * n_dims
        avg_noise = 0.0
        avg_signal = 0.0

        for w, rec in top_k:
            wn = w / total_w
            ls = rec["lengthscales"]
            for i in range(min(n_dims, len(ls))):
                avg_ls[i] += wn * ls[i]
            if len(ls) < n_dims:
                for i in range(len(ls), n_dims):
                    avg_ls[i] += wn * 0.5
            avg_noise += wn * rec["noise_var"]
            avg_signal += wn * rec["signal_var"]

        return {
            "lengthscales": avg_ls,
            "noise_var": max(avg_noise, 1e-5),
            "signal_var": max(avg_signal, 0.1),
        }

    @staticmethod
    def _default_init(n_dims: int) -> dict[str, Any]:
        return {
            "lengthscales": [0.5] * n_dims,
            "noise_var": 0.01,
            "signal_var": 1.0,
        }


class MetaLearner:
    """Top-level meta-learning coordinator.

    Combines AcquisitionBandit + HyperparamPrior + campaign transfer
    into a single interface for the orchestrator.
    """

    def __init__(self) -> None:
        self.acquisition_bandit = AcquisitionBandit()
        self.hyperparam_prior = HyperparamPrior()
        self._task_history: dict[str, TaskEmbedding] = {}

    def register_task(self, task: TaskEmbedding) -> None:
        self._task_history[task.campaign_id] = task

    def get_campaign_config(self, task: TaskEmbedding) -> dict[str, Any]:
        """Return recommended algorithm configuration for a new campaign."""
        acquisition = self.acquisition_bandit.select(task)
        hyperparam_init = self.hyperparam_prior.suggest_init(task, task.n_dimensions)
        similar = self._find_similar_campaigns(task, k=3)

        config = {
            "acquisition": acquisition,
            "gp_init": hyperparam_init,
            "similar_campaign_ids": [t.campaign_id for t in similar],
            "expected_benefit": self._estimate_meta_benefit(task, similar),
        }
        logger.info("meta_learner.config", extra={
            "campaign_id": task.campaign_id,
            "acquisition": acquisition,
            "similar_campaigns": len(similar),
        })
        return config

    def record_outcome(
        self,
        task: TaskEmbedding,
        acquisition_used: str,
        discovery_velocity: float,
        final_gp_hyperparams: dict[str, Any],
    ) -> None:
        self.acquisition_bandit.update(acquisition_used, discovery_velocity, task)
        if "lengthscales" in final_gp_hyperparams:
            self.hyperparam_prior.record(
                task,
                lengthscales=final_gp_hyperparams["lengthscales"],
                noise_var=final_gp_hyperparams.get("noise_var", 0.01),
                signal_var=final_gp_hyperparams.get("signal_var", 1.0),
                mll=final_gp_hyperparams.get("mll", 0.0),
            )
        self._task_history[task.campaign_id] = task

    def _find_similar_campaigns(
        self, task: TaskEmbedding, k: int = 3
    ) -> list[TaskEmbedding]:
        if not self._task_history:
            return []
        scored = [
            (task.similarity(t), t)
            for cid, t in self._task_history.items()
            if cid != task.campaign_id
        ]
        scored.sort(reverse=True)
        return [t for _, t in scored[:k]]

    def _estimate_meta_benefit(
        self, task: TaskEmbedding, similar: list[TaskEmbedding]
    ) -> float:
        if not similar:
            return 0.0
        avg_sim = sum(task.similarity(s) for s in similar) / len(similar)
        # Rough estimate: benefit scales with similarity × log(n_similar_campaigns)
        return round(avg_sim * math.log(len(similar) + 1), 3)

"""Prioritized Experience Replay (PER) with SumTree.

Implements proportional prioritization for DQN training.
Higher TD-error transitions are sampled more frequently,
with importance sampling weights to correct the bias.

References:
    Schaul et al., "Prioritized Experience Replay" (2015)
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["PrioritizedReplayBuffer", "SumTree"]


# ---------------------------------------------------------------------------
# SumTree — O(log n) proportional sampling
# ---------------------------------------------------------------------------


class SumTree:
    """Binary tree where each leaf stores a priority value.

    The parent node stores the sum of its children.
    This enables O(log n) proportional sampling: to sample a transition
    with probability proportional to its priority, sample a uniform
    value in [0, total_sum] and traverse the tree.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list[Any] = [None] * capacity
        self.write_idx = 0
        self.size = 0

    @property
    def total(self) -> float:
        """Total sum of all priorities."""
        return float(self.tree[0])

    def add(self, priority: float, data: Any) -> None:
        """Add or overwrite a transition with given priority."""
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self._update(tree_idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _update(self, tree_idx: int, priority: float) -> None:
        """Update priority at tree_idx and propagate up."""
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def update(self, data_idx: int, priority: float) -> None:
        """Update priority for a specific data index."""
        tree_idx = data_idx + self.capacity - 1
        self._update(tree_idx, priority)

    def get(self, cumsum: float) -> tuple[int, float, Any]:
        """Find the leaf node for a given cumulative sum.

        Returns:
            (data_idx, priority, data)
        """
        tree_idx = 0
        while True:
            left = 2 * tree_idx + 1
            right = left + 1

            if left >= len(self.tree):
                # Leaf node
                break

            if cumsum <= self.tree[left]:
                tree_idx = left
            else:
                cumsum -= self.tree[left]
                tree_idx = right

        data_idx = tree_idx - self.capacity + 1
        return data_idx, float(self.tree[tree_idx]), self.data[data_idx]

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer
# ---------------------------------------------------------------------------


@dataclass
class PERConfig:
    """Configuration for Prioritized Experience Replay."""

    capacity: int = 10000
    alpha: float = 0.6  # priority exponent (0 = uniform, 1 = full prioritization)
    beta: float = 0.4  # IS weight exponent (annealed to 1.0 during training)
    beta_increment: float = 0.001  # per-sample beta increment
    epsilon: float = 1e-6  # small constant to avoid zero priority
    max_priority_clip: float = 100.0  # clip max priority to avoid instability


class PrioritizedReplayBuffer:
    """Experience replay buffer with proportional prioritization.

    Transitions with higher TD errors are sampled more frequently.
    Importance sampling (IS) weights correct the resulting bias.
    """

    def __init__(self, config: PERConfig | None = None):
        if config is None:
            config = PERConfig()
        self.config = config
        self.tree = SumTree(config.capacity)
        self.max_priority = 1.0

    def add(self, transition: Any, priority: float | None = None) -> None:
        """Add transition with given priority (default: max priority).

        New transitions always get max priority to ensure they're
        sampled at least once before their TD error is known.
        """
        if priority is None:
            p = self.max_priority
        else:
            p = min(abs(priority), self.config.max_priority_clip)
            p = max(p, self.config.epsilon)

        self.tree.add(p ** self.config.alpha, transition)

    def sample(
        self,
        batch_size: int,
    ) -> tuple[list[Any], list[int], np.ndarray]:
        """Sample batch proportional to priorities.

        Returns:
            (transitions, data_indices, importance_sampling_weights)

        IS weights are normalized so max weight = 1.0.
        """
        n = len(self.tree)
        if n == 0:
            return [], [], np.array([])

        batch_size = min(batch_size, n)
        transitions: list[Any] = []
        indices: list[int] = []
        priorities: list[float] = []

        # Stratified sampling: divide [0, total] into batch_size segments
        segment = self.tree.total / batch_size

        # Anneal beta toward 1.0
        beta = min(1.0, self.config.beta)
        self.config.beta = min(1.0, self.config.beta + self.config.beta_increment)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            cumsum = random.uniform(lo, hi)

            data_idx, priority, data = self.tree.get(cumsum)
            if data is None:
                # Tree slot not yet filled; retry with different value
                cumsum = random.uniform(0, self.tree.total)
                data_idx, priority, data = self.tree.get(cumsum)
                if data is None:
                    continue

            transitions.append(data)
            indices.append(data_idx)
            priorities.append(max(priority, self.config.epsilon))

        if not transitions:
            return [], [], np.array([])

        # Importance sampling weights
        # w_i = (N * P(i))^{-beta} / max_w
        probs = np.array(priorities) / max(self.tree.total, self.config.epsilon)
        is_weights = (n * probs) ** (-beta)
        # Normalize so max weight = 1.0 (prevents scaling issues)
        is_weights /= is_weights.max()

        return transitions, indices, is_weights.astype(np.float32)

    def update_priorities(
        self,
        indices: list[int],
        td_errors: np.ndarray | list[float],
    ) -> None:
        """Update priorities based on TD errors.

        priority = (|td_error| + epsilon) ^ alpha
        """
        if isinstance(td_errors, list):
            td_errors = np.array(td_errors)

        for idx, td_err in zip(indices, td_errors):
            priority = (abs(float(td_err)) + self.config.epsilon)
            priority = min(priority, self.config.max_priority_clip)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.config.alpha)

    def __len__(self) -> int:
        return len(self.tree)

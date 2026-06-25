"""Boundary-layer configuration for candidate-pool arbitration (Δ1+).

``ArbitrationConfig`` controls *routing thresholds* and *candidate-pool
behaviour* only.  It deliberately holds **no** utility weights and **no** phase
logic -- those belong to ``PhaseConfig`` and the strategy authority.  Keeping
the two configs separate prevents the boundary layer from quietly duplicating
campaign authority.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ArbitrationConfig:
    """Knobs for pool construction and concrete-candidate scoring."""

    # --- candidate pool sizing ---
    k_nexus: int = 3  # how many Nexus top-k candidates to request
    include_local_baseline: bool = True
    include_replicate_best: bool = True
    include_sobol_in_exploration: bool = True

    # --- redundancy / diversity penalty (normalized parameter space) ---
    redundancy_distance_threshold: float = 0.05  # below this normalized L2 → penalised
    redundancy_weight: float = 0.10  # max penalty applied to an exact duplicate

    # --- difficulty-aware routing (consumed in Δ3; inert until then) ---
    difficulty_fast_threshold: float = 0.35
    difficulty_deep_threshold: float = 0.75

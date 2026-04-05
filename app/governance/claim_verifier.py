"""Governance Layer — Claim Verifier (single-verifier, multi-domain).

Verifies a claim against three orthogonal rule sets:

  1. Physics feasibility  — is the value within the declared search space?
  2. Safety envelope      — does the value respect hard lab safety limits?
  3. Statistical precedent— is the value an outlier vs. campaign history?

Each check returns a list of PolicyViolation objects.  The verifier does NOT
make the final verdict — that is done by the middleware gate after combining
results from all checks.

Design note: this is intentionally a single verifier (not multi-agent voting)
to keep latency and cost predictable.  The three check domains cover ~80% of
real-world governance failures without the overhead of consensus protocols.
"""
from __future__ import annotations

import logging
from typing import Any

from app.governance.schemas import Claim, PolicyViolation

logger = logging.getLogger(__name__)

# Statistical threshold: claims beyond this many σ from historical mean
# are flagged as outliers.
_SIGMA_SOFT_THRESHOLD = 2.0
_SIGMA_HARD_THRESHOLD = 3.5

# Normalized position thresholds: flag claims near or beyond param bounds.
# A value at 98% of range is unusual but not necessarily wrong.
_NORM_WARN_THRESHOLD = 0.95   # |0 or 1| side
_NORM_HARD_THRESHOLD = 1.0    # exactly at boundary — still legal but flag it


# ---------------------------------------------------------------------------
# ClaimVerifier
# ---------------------------------------------------------------------------

class ClaimVerifier:
    """Domain-specific verifier for a single Claim.

    Usage
    -----
    verifier = ClaimVerifier()
    violations = verifier.verify(
        claim=claim,
        safety_envelope=safety_envelope,
        dim_def=dim_def,        # from ExplorationSpace.dimensions matching claim.param_name
        param_history=(mean, std),
        round_number=round_number,
    )
    """

    def verify(
        self,
        claim: Claim,
        safety_envelope: dict[str, Any],
        dim_def: dict[str, Any] | None,
        param_history: tuple[float | None, float | None],
        round_number: int,
    ) -> list[PolicyViolation]:
        """Run all three domain checks and return all violations found.

        Args:
            claim:           The claim to verify.
            safety_envelope: Dict from SafetyEnvelope.model_dump().
            dim_def:         Matching DimensionDef.model_dump() or None.
            param_history:   (mean, std) from ClaimTracker.get_param_history_stats().
            round_number:    Current campaign round (used to tier the checks).
        """
        violations: list[PolicyViolation] = []

        violations.extend(self._check_physics(claim, dim_def))
        violations.extend(self._check_safety_envelope(claim, safety_envelope))
        violations.extend(
            self._check_statistical(claim, param_history, round_number)
        )

        if violations:
            logger.debug(
                "Claim %s (%s=%s): %d violation(s)",
                claim.id,
                claim.param_name,
                claim.display_value(),
                len(violations),
            )

        return violations

    # ---------------------------------------------------------------- tier 1

    def _check_physics(
        self,
        claim: Claim,
        dim_def: dict[str, Any] | None,
    ) -> list[PolicyViolation]:
        """Verify claim value is within declared search-space bounds."""
        if claim.param_value is None:
            # Categorical claims: check value is in choices list
            if dim_def and claim.param_value_str is not None:
                choices = dim_def.get("choices") or []
                if choices and claim.param_value_str not in choices:
                    return [
                        PolicyViolation(
                            rule_id="physics.categorical_out_of_choices",
                            message=(
                                f"{claim.param_name}={claim.param_value_str!r} "
                                f"is not in allowed choices {choices}"
                            ),
                            severity="hard_block",
                            param_name=claim.param_name,
                        )
                    ]
            return []

        violations: list[PolicyViolation] = []
        value = claim.param_value

        if dim_def is None:
            return []

        min_val: float | None = dim_def.get("min_value")
        max_val: float | None = dim_def.get("max_value")

        if min_val is not None and value < min_val:
            violations.append(
                PolicyViolation(
                    rule_id="physics.below_min",
                    message=(
                        f"{claim.param_name}={value} is below declared "
                        f"minimum {min_val}"
                    ),
                    severity="hard_block",
                    param_name=claim.param_name,
                    observed=value,
                    threshold=min_val,
                )
            )

        if max_val is not None and value > max_val:
            violations.append(
                PolicyViolation(
                    rule_id="physics.above_max",
                    message=(
                        f"{claim.param_name}={value} exceeds declared "
                        f"maximum {max_val}"
                    ),
                    severity="hard_block",
                    param_name=claim.param_name,
                    observed=value,
                    threshold=max_val,
                )
            )

        # Warn when value is suspiciously close to a boundary (round ≥ 1)
        if not violations and min_val is not None and max_val is not None:
            span = max_val - min_val
            if span > 0:
                normalized = (value - min_val) / span
                if normalized > _NORM_WARN_THRESHOLD or normalized < (
                    1 - _NORM_WARN_THRESHOLD
                ):
                    violations.append(
                        PolicyViolation(
                            rule_id="physics.near_boundary",
                            message=(
                                f"{claim.param_name}={value} is near a "
                                f"search-space boundary "
                                f"(normalized={normalized:.3f})"
                            ),
                            severity="warning",
                            param_name=claim.param_name,
                            observed=normalized,
                            threshold=_NORM_WARN_THRESHOLD,
                        )
                    )

        return violations

    # ---------------------------------------------------------------- tier 2

    def _check_safety_envelope(
        self,
        claim: Claim,
        safety_envelope: dict[str, Any],
    ) -> list[PolicyViolation]:
        """Verify claim against hard lab safety limits (SafetyEnvelope)."""
        if claim.param_value is None:
            return []

        violations: list[PolicyViolation] = []
        value = claim.param_value

        # Temperature limit
        max_temp = float(safety_envelope.get("max_temp_c", 95.0))
        if "temp" in claim.param_name.lower() and value > max_temp:
            violations.append(
                PolicyViolation(
                    rule_id="safety.temperature_exceeded",
                    message=(
                        f"{claim.param_name}={value}°C exceeds safety "
                        f"envelope maximum {max_temp}°C"
                    ),
                    severity="hard_block",
                    param_name=claim.param_name,
                    observed=value,
                    threshold=max_temp,
                )
            )

        # Volume limit
        max_vol = float(safety_envelope.get("max_volume_ul", 1000.0))
        if "volume" in claim.param_name.lower() and value > max_vol:
            violations.append(
                PolicyViolation(
                    rule_id="safety.volume_exceeded",
                    message=(
                        f"{claim.param_name}={value}µL exceeds safety "
                        f"envelope maximum {max_vol}µL"
                    ),
                    severity="hard_block",
                    param_name=claim.param_name,
                    observed=value,
                    threshold=max_vol,
                )
            )

        # Low confidence from the emitting agent → soft block
        if claim.confidence < 0.3:
            violations.append(
                PolicyViolation(
                    rule_id="safety.low_confidence",
                    message=(
                        f"Agent {claim.emitting_agent!r} emitted "
                        f"{claim.param_name} with low confidence "
                        f"({claim.confidence:.2f} < 0.30)"
                    ),
                    severity="soft_block",
                    param_name=claim.param_name,
                    observed=claim.confidence,
                    threshold=0.3,
                )
            )

        return violations

    # ---------------------------------------------------------------- tier 3

    def _check_statistical(
        self,
        claim: Claim,
        param_history: tuple[float | None, float | None],
        round_number: int,
    ) -> list[PolicyViolation]:
        """Detect statistical outliers vs. campaign history.

        Tier gating:
          round 0-2 : skip (insufficient history)
          round ≥ 3 : μ ± σ check
        """
        if round_number < 3:
            return []

        if claim.param_value is None:
            return []

        mean, std = param_history
        if mean is None or std is None:
            return []

        sigma_dev = abs(claim.param_value - mean) / std

        if sigma_dev > _SIGMA_HARD_THRESHOLD:
            return [
                PolicyViolation(
                    rule_id="stats.extreme_outlier",
                    message=(
                        f"{claim.param_name}={claim.param_value} deviates "
                        f"{sigma_dev:.1f}σ from campaign history "
                        f"(μ={mean:.3g}, σ={std:.3g})"
                    ),
                    severity="hard_block",
                    param_name=claim.param_name,
                    observed=sigma_dev,
                    threshold=_SIGMA_HARD_THRESHOLD,
                )
            ]

        if sigma_dev > _SIGMA_SOFT_THRESHOLD:
            return [
                PolicyViolation(
                    rule_id="stats.outlier",
                    message=(
                        f"{claim.param_name}={claim.param_value} deviates "
                        f"{sigma_dev:.1f}σ from campaign history "
                        f"(μ={mean:.3g}, σ={std:.3g})"
                    ),
                    severity="soft_block",
                    param_name=claim.param_name,
                    observed=sigma_dev,
                    threshold=_SIGMA_SOFT_THRESHOLD,
                )
            ]

        return []

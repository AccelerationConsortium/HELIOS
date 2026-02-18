"""Nexus Advisor — optional integration with the Nexus optimization platform.

Wraps Nexus REST API calls to provide enhanced diagnostics, causal discovery,
meta-learning advice, and hypothesis tracking.  All calls are optional and
degrade gracefully if Nexus is unreachable.

Uses only ``urllib`` (no extra dependencies).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

NEXUS_URL = os.getenv("NEXUS_URL", "http://localhost:8000")
_TIMEOUT = 5  # seconds


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalEdge:
    """A directed causal relationship discovered by Nexus."""
    source: str
    target: str
    strength: float  # 0–1


@dataclass(frozen=True)
class HypothesisInfo:
    """Status of a single hypothesis tracked by Nexus."""
    hypothesis_id: str
    statement: str
    status: str  # PROPOSED | TESTING | SUPPORTED | REFUTED | INCONCLUSIVE
    evidence_count: int


@dataclass(frozen=True)
class NexusInsights:
    """Enhanced diagnostics returned by Nexus."""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    causal_edges: tuple[CausalEdge, ...] = ()
    hypotheses: tuple[HypothesisInfo, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetaAdvice:
    """Meta-learning transfer advice from Nexus."""
    weight_adjustments: dict[str, float] = field(default_factory=dict)
    recommended_phase: str | None = None
    reasoning: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _api(method: str, path: str, body: dict | None = None) -> dict | None:
    """Call the Nexus REST API.  Returns None on any failure."""
    url = f"{NEXUS_URL}/api{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())
    except (HTTPError, URLError, OSError, ValueError) as exc:
        logger.debug("Nexus API call failed (%s %s): %s", method, path, exc)
        return None


def _get(path: str) -> dict | None:
    return _api("GET", path)


def _post(path: str, body: dict | None = None) -> dict | None:
    return _api("POST", path, body)


# ---------------------------------------------------------------------------
# NexusAdvisor
# ---------------------------------------------------------------------------


class NexusAdvisor:
    """Optional advisor that enriches NeverOT decisions with Nexus insights.

    All public methods return ``None`` when Nexus is unreachable so the
    caller can seamlessly fall back to local-only logic.
    """

    def __init__(self, nexus_url: str | None = None) -> None:
        global NEXUS_URL  # noqa: PLW0603
        if nexus_url is not None:
            NEXUS_URL = nexus_url

    # ---- public API ----

    def get_enhanced_diagnostics(
        self,
        campaign_id: str,
        causal_data: list[list[float]] | None = None,
        var_names: list[str] | None = None,
        tracker_state: dict | None = None,
    ) -> NexusInsights | None:
        """Fetch diagnostics + causal discovery + hypothesis status.

        Returns ``None`` if Nexus is unreachable.
        """
        try:
            # 1. Diagnostics
            diag_raw = _get(f"/campaigns/{campaign_id}/diagnostics")
            diagnostics = diag_raw if isinstance(diag_raw, dict) and "error" not in diag_raw else {}

            # 2. Causal discovery (optional — needs data)
            causal_edges: list[CausalEdge] = []
            if causal_data and var_names:
                causal_raw = _post("/analysis/causal/discover", {
                    "data": causal_data,
                    "var_names": var_names,
                    "alpha": 0.05,
                })
                if causal_raw and "error" not in causal_raw:
                    for edge in causal_raw.get("edges", []):
                        causal_edges.append(CausalEdge(
                            source=edge.get("source", ""),
                            target=edge.get("target", ""),
                            strength=float(edge.get("strength", 0.0)),
                        ))

            # 3. Hypothesis status (optional — needs tracker state)
            hypotheses: list[HypothesisInfo] = []
            if tracker_state:
                hyp_raw = _post("/analysis/hypothesis/status", {
                    "tracker_state": tracker_state,
                })
                if hyp_raw and "error" not in hyp_raw:
                    for h in hyp_raw.get("hypotheses", []):
                        hypotheses.append(HypothesisInfo(
                            hypothesis_id=h.get("id", ""),
                            statement=h.get("statement", ""),
                            status=h.get("status", "INCONCLUSIVE"),
                            evidence_count=int(h.get("evidence_count", 0)),
                        ))

            return NexusInsights(
                diagnostics=diagnostics,
                causal_edges=tuple(causal_edges),
                hypotheses=tuple(hypotheses),
                raw={"diagnostics": diag_raw, "causal": causal_data is not None},
            )
        except Exception as exc:
            logger.warning("Nexus get_enhanced_diagnostics failed: %s", exc)
            return None

    def get_meta_learning_advice(
        self,
        campaign_id: str,
    ) -> MetaAdvice | None:
        """Fetch meta-learning transfer advice for weight adjustment.

        Returns ``None`` if Nexus is unreachable.
        """
        try:
            raw = _post(f"/chat/{campaign_id}", {
                "message": "Based on similar campaigns, what weight adjustments do you recommend for exploration vs exploitation?",
            })
            if raw is None or "error" in raw:
                return None

            # Parse structured advice from Nexus response metadata
            metadata = raw.get("metadata", {})
            recommendations = metadata.get("recommendations", {})

            weight_adj: dict[str, float] = {}
            if "exploration_weight" in recommendations:
                weight_adj["w_info_gain"] = float(recommendations["exploration_weight"])
            if "exploitation_weight" in recommendations:
                weight_adj["w_improvement"] = float(recommendations["exploitation_weight"])

            return MetaAdvice(
                weight_adjustments=weight_adj,
                recommended_phase=metadata.get("recommended_phase"),
                reasoning=raw.get("reply", ""),
                raw=raw,
            )
        except Exception as exc:
            logger.warning("Nexus get_meta_learning_advice failed: %s", exc)
            return None

    def causal_discovery(
        self,
        data: list[list[float]],
        var_names: list[str],
        alpha: float = 0.05,
    ) -> list[CausalEdge] | None:
        """Run causal discovery on provided data.

        Returns ``None`` if Nexus is unreachable.
        """
        try:
            raw = _post("/analysis/causal/discover", {
                "data": data,
                "var_names": var_names,
                "alpha": alpha,
            })
            if raw is None or "error" in raw:
                return None
            edges: list[CausalEdge] = []
            for edge in raw.get("edges", []):
                edges.append(CausalEdge(
                    source=edge.get("source", ""),
                    target=edge.get("target", ""),
                    strength=float(edge.get("strength", 0.0)),
                ))
            return edges
        except Exception as exc:
            logger.warning("Nexus causal_discovery failed: %s", exc)
            return None

    def hypothesis_status(
        self,
        tracker_state: dict,
    ) -> list[HypothesisInfo] | None:
        """Check hypothesis lifecycle status.

        Returns ``None`` if Nexus is unreachable.
        """
        try:
            raw = _post("/analysis/hypothesis/status", {
                "tracker_state": tracker_state,
            })
            if raw is None or "error" in raw:
                return None
            infos: list[HypothesisInfo] = []
            for h in raw.get("hypotheses", []):
                infos.append(HypothesisInfo(
                    hypothesis_id=h.get("id", ""),
                    statement=h.get("statement", ""),
                    status=h.get("status", "INCONCLUSIVE"),
                    evidence_count=int(h.get("evidence_count", 0)),
                ))
            return infos
        except Exception as exc:
            logger.warning("Nexus hypothesis_status failed: %s", exc)
            return None

    def upload_spectral_data(
        self,
        campaign_id: str,
        spectral_matrix: list[list[float]],
        var_names: list[str],
    ) -> dict | None:
        """Send spectral data to Nexus for embedding / SSL processing.

        Returns ``{"embeddings": [[...], ...], "similarity_matrix": [[...], ...]}``
        on success, ``None`` on failure.
        """
        try:
            raw = _post("/analysis/embedding", {
                "campaign_id": campaign_id,
                "data": spectral_matrix,
                "var_names": var_names,
            })
            if raw is None or "error" in raw:
                return None
            return raw
        except Exception as exc:
            logger.warning("Nexus upload_spectral_data failed: %s", exc)
            return None

    def get_similar_experiments(
        self,
        campaign_id: str,
        record_id: str,
        top_k: int = 5,
    ) -> list[dict] | None:
        """Ask Nexus which historical experiments are most similar.

        Uses learned embeddings to find nearest neighbours.
        Returns ``None`` if Nexus is unreachable.
        """
        try:
            raw = _post("/analysis/similarity", {
                "campaign_id": campaign_id,
                "record_id": record_id,
                "top_k": top_k,
            })
            if raw is None or "error" in raw:
                return None
            return raw.get("similar", [])
        except Exception as exc:
            logger.warning("Nexus get_similar_experiments failed: %s", exc)
            return None

    def sync_campaign(
        self,
        campaign_id: str,
        observations: list[dict[str, Any]],
        name: str = "",
        parameters: list[dict] | None = None,
        objectives: list[dict] | None = None,
    ) -> str | None:
        """Create or update a mirror Nexus campaign from NeverOT data.

        Returns the Nexus campaign_id on success, ``None`` on failure.
        """
        try:
            # Try appending first (campaign may already exist)
            append_raw = _post(f"/campaigns/{campaign_id}/append", {
                "data": observations,
            })
            if append_raw and "error" not in append_raw:
                return campaign_id

            # If append fails, try creating a new campaign
            create_raw = _post("/campaigns/from-upload", {
                "name": name or f"neverot-mirror-{campaign_id}",
                "description": "Mirror campaign synced from NeverOT",
                "data": observations,
                "mapping": {
                    "parameters": parameters or [],
                    "objectives": objectives or [],
                    "metadata": [],
                    "ignored": [],
                },
                "batch_size": 5,
                "exploration_weight": 0.5,
            })
            if create_raw and "error" not in create_raw:
                return create_raw.get("campaign_id", campaign_id)

            return None
        except Exception as exc:
            logger.warning("Nexus sync_campaign failed: %s", exc)
            return None

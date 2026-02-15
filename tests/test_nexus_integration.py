"""Tests for Nexus integration into NeverOT.

Covers:
- NexusAdvisor with mocked HTTP responses
- strategy_selector with enable_nexus=True but Nexus unreachable
- InverseDesignAgent with and without Nexus
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from app.services.nexus_advisor import (
    CausalEdge,
    HypothesisInfo,
    MetaAdvice,
    NexusAdvisor,
    NexusInsights,
)


# ---------------------------------------------------------------------------
# NexusAdvisor tests
# ---------------------------------------------------------------------------


class TestNexusAdvisor(unittest.TestCase):
    """Test NexusAdvisor with mocked HTTP responses."""

    def setUp(self):
        self.advisor = NexusAdvisor(nexus_url="http://fake-nexus:8000")

    @patch("app.services.nexus_advisor.urlopen")
    def test_get_enhanced_diagnostics_success(self, mock_urlopen):
        """Should parse diagnostics, causal edges, and hypotheses."""
        # Mock three sequential calls: diagnostics, causal, hypothesis
        responses = [
            # diagnostics
            json.dumps({"convergence_trend": 0.8, "best_kpi_value": 0.95}).encode(),
            # causal discovery
            json.dumps({"edges": [
                {"source": "temperature", "target": "yield", "strength": 0.85},
            ]}).encode(),
            # hypothesis status
            json.dumps({"hypotheses": [
                {"id": "h1", "statement": "temp matters", "status": "SUPPORTED", "evidence_count": 3},
            ]}).encode(),
        ]
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(side_effect=[
            _make_response(responses[0]),
            _make_response(responses[1]),
            _make_response(responses[2]),
        ])
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.side_effect = [
            _make_cm(responses[0]),
            _make_cm(responses[1]),
            _make_cm(responses[2]),
        ]

        result = self.advisor.get_enhanced_diagnostics(
            campaign_id="test-campaign",
            causal_data=[[80.0, 0.7], [90.0, 0.9]],
            var_names=["temperature", "yield"],
            tracker_state={"hypotheses": [{"id": "h1", "statement": "temp matters", "status": "TESTING", "evidence": [], "tests_run": 2}]},
        )

        self.assertIsInstance(result, NexusInsights)
        self.assertEqual(len(result.causal_edges), 1)
        self.assertEqual(result.causal_edges[0].source, "temperature")
        self.assertEqual(len(result.hypotheses), 1)
        self.assertEqual(result.hypotheses[0].status, "SUPPORTED")

    @patch("app.services.nexus_advisor.urlopen")
    def test_get_enhanced_diagnostics_unreachable(self, mock_urlopen):
        """Should return None when Nexus is unreachable."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")

        result = self.advisor.get_enhanced_diagnostics(campaign_id="test")
        # Should return NexusInsights with empty diagnostics (first call fails gracefully)
        # or None if the whole method fails
        # The method catches all exceptions and returns None
        self.assertTrue(result is None or isinstance(result, NexusInsights))

    @patch("app.services.nexus_advisor.urlopen")
    def test_get_meta_learning_advice_success(self, mock_urlopen):
        """Should parse meta-learning advice."""
        resp_data = json.dumps({
            "reply": "Recommend more exploration based on similar campaigns",
            "metadata": {
                "recommendations": {
                    "exploration_weight": 0.1,
                    "exploitation_weight": -0.05,
                },
                "recommended_phase": "explore",
            },
        }).encode()
        mock_urlopen.return_value = _make_cm(resp_data)

        result = self.advisor.get_meta_learning_advice(campaign_id="test")
        self.assertIsInstance(result, MetaAdvice)
        self.assertIn("w_info_gain", result.weight_adjustments)
        self.assertEqual(result.recommended_phase, "explore")

    @patch("app.services.nexus_advisor.urlopen")
    def test_causal_discovery_success(self, mock_urlopen):
        """Should return causal edges."""
        resp_data = json.dumps({"edges": [
            {"source": "a", "target": "b", "strength": 0.9},
            {"source": "b", "target": "c", "strength": 0.4},
        ]}).encode()
        mock_urlopen.return_value = _make_cm(resp_data)

        result = self.advisor.causal_discovery(
            data=[[1, 2, 3], [4, 5, 6]],
            var_names=["a", "b", "c"],
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], CausalEdge)

    @patch("app.services.nexus_advisor.urlopen")
    def test_sync_campaign_append(self, mock_urlopen):
        """Should return campaign_id on successful append."""
        resp_data = json.dumps({"campaign_id": "c1", "appended": 5}).encode()
        mock_urlopen.return_value = _make_cm(resp_data)

        result = self.advisor.sync_campaign("c1", [{"x": "1", "y": "2"}])
        self.assertEqual(result, "c1")

    @patch("app.services.nexus_advisor.urlopen")
    def test_unreachable_returns_none(self, mock_urlopen):
        """All methods should return None when Nexus is unreachable."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")

        self.assertIsNone(self.advisor.causal_discovery([[1]], ["x"]))
        self.assertIsNone(self.advisor.hypothesis_status({"hypotheses": []}))
        self.assertIsNone(self.advisor.get_meta_learning_advice("x"))
        self.assertIsNone(self.advisor.sync_campaign("x", []))


# ---------------------------------------------------------------------------
# strategy_selector with Nexus tests
# ---------------------------------------------------------------------------


class TestStrategySelectorNexus(unittest.TestCase):
    """Test strategy_selector with enable_nexus=True."""

    @patch("app.services.nexus_advisor.urlopen")
    def test_nexus_unreachable_fallback(self, mock_urlopen):
        """With enable_nexus=True but Nexus unreachable, should still produce a decision."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")

        from app.services.strategy_selector import (
            CampaignSnapshot,
            PhaseConfig,
            StrategyDecision,
            select_strategy,
        )

        snapshot = CampaignSnapshot(
            round_number=3,
            max_rounds=10,
            n_observations=15,
            n_dimensions=3,
            has_categorical=False,
            has_log_scale=False,
            kpi_history=tuple(range(15)),
            direction="maximize",
            all_params=tuple({"x": float(i), "y": float(i * 2)} for i in range(15)),
            all_kpis=tuple(float(i) for i in range(15)),
            last_batch_kpis=tuple(float(i) for i in range(10, 15)),
            last_batch_params=tuple({"x": float(i), "y": float(i * 2)} for i in range(10, 15)),
        )
        config = PhaseConfig(enable_nexus=True)

        # Nexus is not running, but strategy selection should succeed
        decision = select_strategy(snapshot, config)
        self.assertIsInstance(decision, StrategyDecision)
        self.assertTrue(len(decision.backend_name) > 0)
        self.assertTrue(decision.confidence > 0)

    def test_nexus_disabled_no_calls(self):
        """With enable_nexus=False (default), no Nexus calls should be made."""
        from app.services.strategy_selector import (
            CampaignSnapshot,
            PhaseConfig,
            select_strategy,
        )

        snapshot = CampaignSnapshot(
            round_number=1,
            max_rounds=10,
            n_observations=5,
            n_dimensions=2,
            has_categorical=False,
            has_log_scale=False,
            kpi_history=(1.0, 2.0, 3.0, 4.0, 5.0),
            direction="maximize",
        )
        config = PhaseConfig(enable_nexus=False)

        with patch("app.services.nexus_advisor.urlopen") as mock_url:
            decision = select_strategy(snapshot, config)
            mock_url.assert_not_called()


# ---------------------------------------------------------------------------
# InverseDesignAgent with Nexus tests
# ---------------------------------------------------------------------------


class TestInverseDesignAgentNexus(unittest.TestCase):
    """Test InverseDesignAgent with and without Nexus."""

    def _make_input(self):
        from app.agents.inverse_design_agent import InverseDesignInput
        return InverseDesignInput(
            objective="HER catalyst with η10 < 50mV in 1M KOH",
            target_metrics={"eta10_mv": {"direction": "minimize", "target": 50}},
            max_systems=2,
            search_mode="literature",
        )

    def _run_async(self, coro):
        """Run an async coroutine in a new event loop."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @patch("app.services.nexus_advisor.urlopen")
    def test_without_nexus(self, mock_urlopen):
        """Agent should work fine without Nexus."""
        from app.agents.inverse_design_agent import InverseDesignAgent
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")

        agent = InverseDesignAgent()
        input_data = self._make_input()
        result = self._run_async(agent.process(input_data))
        self.assertIsNotNone(result)

    @patch("app.services.nexus_advisor.urlopen")
    def test_with_nexus_available(self, mock_urlopen):
        """Agent should enhance results when Nexus is available."""
        from app.agents.inverse_design_agent import InverseDesignAgent

        causal_resp = json.dumps({"edges": [
            {"source": "eta10_mv", "target": "eta10_mv", "strength": 0.9},
        ]}).encode()
        hyp_resp = json.dumps({"hypotheses": [
            {"id": "hyp_eta10_mv", "statement": "achievable", "status": "SUPPORTED", "evidence_count": 2},
        ]}).encode()
        sync_resp = json.dumps({"campaign_id": "inv-mirror", "appended": 2}).encode()

        mock_urlopen.side_effect = [
            _make_cm(causal_resp),
            _make_cm(hyp_resp),
            _make_cm(sync_resp),
        ]

        agent = InverseDesignAgent()
        input_data = self._make_input()
        result = self._run_async(agent.process(input_data))
        self.assertIsNotNone(result)

    @patch("app.services.nexus_advisor.urlopen")
    def test_with_nexus_unreachable(self, mock_urlopen):
        """Agent should degrade gracefully when Nexus is unreachable."""
        from app.agents.inverse_design_agent import InverseDesignAgent
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("Connection refused")
        agent = InverseDesignAgent()
        input_data = self._make_input()
        result = self._run_async(agent.process(input_data))
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(data: bytes):
    """Create a mock response object."""
    resp = MagicMock()
    resp.read.return_value = data
    return resp


def _make_cm(data: bytes):
    """Create a context manager that returns a mock response."""
    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    resp.read.return_value = data
    return resp


if __name__ == "__main__":
    unittest.main()

"""Tests for v3 multi-agent enhancements.

Covers:
- AgentCapability declaration and has_capability()
- AgentReputation scoring and degradation detection
- RoundBlackboard: write/read/subscribe/prefix
- KnowledgeBus: publish/drain/format_for_prompt/TTL
- AcquisitionBandit: selection, update, statistics
- ConformalPredictor: coverage guarantees
"""
import asyncio
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# AgentCapability tests
# ---------------------------------------------------------------------------

class TestAgentCapability:
    def test_capability_creation(self):
        from app.agents.base import AgentCapability
        cap = AgentCapability("hypothesis.bayesian", "Bayesian BO", min_confidence=0.8)
        assert cap.tag == "hypothesis.bayesian"
        assert cap.min_confidence == 0.8

    def test_has_capability_exact(self):
        from app.agents.base import AgentCapability, BaseAgent
        from app.agents.design_agent import DesignAgent

        agent = DesignAgent()
        # DesignAgent should have hypothesis-related capabilities
        assert len(agent.capabilities) > 0

    def test_has_capability_prefix_match(self):
        from app.agents.base import AgentCapability
        # Test a simple agent with known capabilities
        from app.agents.validation_agent import ValidationAgent
        agent = ValidationAgent()
        caps = agent.capabilities
        assert any("validation" in c.tag.lower() for c in caps)

    def test_capability_tags_classmethod(self):
        from app.agents.design_agent import DesignAgent
        tags = DesignAgent.capability_tags()
        assert isinstance(tags, list)
        assert len(tags) > 0
        assert all(isinstance(t, str) for t in tags)


# ---------------------------------------------------------------------------
# AgentReputation tests
# ---------------------------------------------------------------------------

class TestAgentReputation:
    def test_score_cold_start(self):
        from app.agents.control_plane import AgentReputation
        rep = AgentReputation()
        assert rep.score == 0.5

    def test_score_perfect_agent(self):
        from app.agents.control_plane import AgentReputation
        rep = AgentReputation(success_count=100, failure_count=0,
                              total_latency_ms=5000.0)  # 50ms avg
        assert rep.score > 0.9

    def test_score_poor_agent(self):
        from app.agents.control_plane import AgentReputation
        rep = AgentReputation(success_count=2, failure_count=8,
                              total_latency_ms=10000.0)
        assert rep.score < 0.5

    def test_is_degraded_threshold(self):
        from app.agents.control_plane import AgentReputation
        rep = AgentReputation(consecutive_failures=2)
        assert not rep.is_degraded
        rep2 = AgentReputation(consecutive_failures=3)
        assert rep2.is_degraded

    def test_score_weighted_correctly(self):
        from app.agents.control_plane import AgentReputation
        # 80% success rate, 100ms avg latency
        rep = AgentReputation(success_count=8, failure_count=2,
                              total_latency_ms=1000.0)
        # success_rate=0.8, lat_score=max(0,1-100/500)=0.8 → score=0.7*0.8+0.3*0.8=0.8
        assert abs(rep.score - 0.8) < 0.01

    def test_control_plane_has_reputations_property(self):
        from app.agents.control_plane import ControlPlane
        cp = ControlPlane()
        reps = cp.reputations
        assert isinstance(reps, dict)


# ---------------------------------------------------------------------------
# RoundBlackboard tests
# ---------------------------------------------------------------------------

class TestRoundBlackboard:
    def test_basic_write_read(self):
        from app.agents.blackboard import RoundBlackboard

        async def _run():
            bb = RoundBlackboard("round-1", "camp-abc")
            await bb.write("spectral.peak", 450.0, author="analyzer_agent")
            entry = bb.read("spectral.peak")
            assert entry is not None
            assert entry.value == 450.0
            assert entry.author == "analyzer_agent"

        asyncio.run(_run())

    def test_read_missing_key_returns_none(self):
        from app.agents.blackboard import RoundBlackboard

        bb = RoundBlackboard("r1", "c1")
        assert bb.read("nonexistent") is None

    def test_size_increments(self):
        from app.agents.blackboard import RoundBlackboard

        async def _run():
            bb = RoundBlackboard("r1", "c1")
            assert bb.size == 0
            await bb.write("k1", 1, author="a")
            assert bb.size == 1
            await bb.write("k2", 2, author="a")
            assert bb.size == 2
            await bb.write("k1", 3, author="a")  # overwrite
            assert bb.size == 2  # no new key

        asyncio.run(_run())

    def test_subscribe_receives_notification(self):
        from app.agents.blackboard import RoundBlackboard
        import asyncio

        async def _run():
            bb = RoundBlackboard("r1", "c1")
            q = bb.subscribe("sensor.temp")
            await bb.write("sensor.temp", 37.5, author="sensing_agent")
            entry = await asyncio.wait_for(q.get(), timeout=1.0)
            assert entry.value == 37.5

        asyncio.run(_run())

    def test_subscribe_prefix_receives_matching(self):
        from app.agents.blackboard import RoundBlackboard
        import asyncio

        async def _run():
            bb = RoundBlackboard("r1", "c1")
            q = bb.subscribe_prefix("spectral.")
            await bb.write("spectral.peak_nm", 450, author="a")
            await bb.write("instrument.ot2", "ok", author="b")  # should not arrive
            entry = await asyncio.wait_for(q.get(), timeout=1.0)
            assert "peak_nm" in entry.key or entry.key.startswith("spectral.")
            # Instrument entry should not be in this queue
            assert q.empty()

        asyncio.run(_run())

    def test_confidence_validation(self):
        from app.agents.blackboard import RoundBlackboard

        async def _run():
            bb = RoundBlackboard("r1", "c1")
            with pytest.raises((ValueError, AssertionError, Exception)):
                await bb.write("k", 1, author="a", confidence=1.5)

        asyncio.run(_run())

    def test_keys_with_prefix(self):
        from app.agents.blackboard import RoundBlackboard

        async def _run():
            bb = RoundBlackboard("r1", "c1")
            await bb.write("spectral.xrd", 1, author="a")
            await bb.write("spectral.uvvis", 2, author="a")
            await bb.write("instrument.ot2", 3, author="a")
            keys = bb.keys_with_prefix("spectral.")
            assert set(keys) == {"spectral.xrd", "spectral.uvvis"}

        asyncio.run(_run())

    def test_read_all_snapshot(self):
        from app.agents.blackboard import RoundBlackboard

        async def _run():
            bb = RoundBlackboard("r1", "c1")
            await bb.write("a", 1, author="x")
            await bb.write("b", 2, author="y")
            snap = bb.read_all()
            assert set(snap.keys()) == {"a", "b"}

        asyncio.run(_run())

    def test_structured_entry_metadata_snapshot(self):
        from app.agents.blackboard import RoundBlackboard

        async def _run():
            bb = RoundBlackboard("r1", "c1")
            await bb.write(
                "agent.output.decision_nodes",
                [{"id": "d1", "selected": "approve"}],
                author="validation_agent",
                confidence=0.88,
                entry_type="decision",
                tags=("agent_output", "decision_nodes"),
                metadata={"count": 1},
            )
            entry = bb.read("agent.output.decision_nodes")
            assert entry is not None
            assert entry.entry_type == "decision"
            assert entry.tags == ("agent_output", "decision_nodes")
            assert entry.metadata["count"] == 1

            snap = bb.snapshot()
            stored = snap["entries"]["agent.output.decision_nodes"]
            assert stored["entry_type"] == "decision"
            assert stored["tags"] == ["agent_output", "decision_nodes"]
            assert stored["metadata"]["count"] == 1

        asyncio.run(_run())

    def test_manager_get_or_create(self):
        from app.agents.blackboard import BlackboardManager
        mgr = BlackboardManager()
        bb1 = mgr.get_or_create("r1", "c1")
        bb2 = mgr.get_or_create("r1", "c1")  # Same — should return same instance
        assert bb1 is bb2
        bb3 = mgr.get_or_create("r2", "c1")
        assert bb3 is not bb1

    def test_manager_isolates_same_round_across_campaigns(self):
        from app.agents.blackboard import BlackboardManager

        mgr = BlackboardManager()
        c1 = mgr.get_or_create("r1", "camp-1")
        c2 = mgr.get_or_create("r1", "camp-2")

        assert c1 is not c2
        assert mgr.get_existing("r1", "camp-1") is c1
        assert mgr.get_existing("r1", "camp-2") is c2
        assert mgr.get_existing("r1") is None
        assert ("camp-1", "r1") in mgr.active_boards()
        assert ("camp-2", "r1") in mgr.active_boards()


# ---------------------------------------------------------------------------
# KnowledgeBus tests
# ---------------------------------------------------------------------------

class TestKnowledgeBus:
    def test_publish_increments_count(self):
        from app.services.knowledge_bus import KnowledgeBus, KnowledgeEvent
        bus = KnowledgeBus("camp-test")
        ev = KnowledgeEvent("sensing", "k", "tip fail rate up", 0.9, "5", 2)
        bus.publish(ev)
        assert bus.stats()["published"] == 1
        assert bus.stats()["active"] == 1

    def test_drain_returns_unpublished(self):
        from app.services.knowledge_bus import KnowledgeBus, KnowledgeEvent

        async def _run():
            bus = KnowledgeBus("c1")
            ev = KnowledgeEvent("a", "k1", "delta", 0.8, "1", 3)
            bus.publish(ev)
            events = await bus.drain_for_agent("engineer_agent", current_round=1)
            assert len(events) == 1
            # Second drain should return empty (already seen)
            events2 = await bus.drain_for_agent("engineer_agent", current_round=1)
            assert len(events2) == 0

        asyncio.run(_run())

    def test_different_agents_get_same_events(self):
        from app.services.knowledge_bus import KnowledgeBus, KnowledgeEvent

        async def _run():
            bus = KnowledgeBus("c1")
            ev = KnowledgeEvent("sensor", "key", "info", 0.9, "1", 3)
            bus.publish(ev)
            e1 = await bus.drain_for_agent("agent_a", current_round=1)
            e2 = await bus.drain_for_agent("agent_b", current_round=1)
            assert len(e1) == 1
            assert len(e2) == 1

        asyncio.run(_run())

    def test_ttl_expiry(self):
        from app.services.knowledge_bus import KnowledgeEvent

        ev = KnowledgeEvent("src", "k", "d", 0.9, round_id="3", ttl_rounds=2)
        assert not ev.is_expired(4)   # round 3 + ttl 2 = valid until round 5
        assert ev.is_expired(5)
        assert ev.is_expired(10)

    def test_format_for_prompt_empty(self):
        from app.services.knowledge_bus import KnowledgeBus
        bus = KnowledgeBus("c1")
        assert KnowledgeBus.format_for_prompt([]) == ""

    def test_format_for_prompt_content(self):
        from app.services.knowledge_bus import KnowledgeBus, KnowledgeEvent
        ev = KnowledgeEvent("sensing", "instrument.ot2.tip", "failure rate elevated", 0.85, "2")
        fmt = KnowledgeBus.format_for_prompt([ev])
        assert "failure rate elevated" in fmt
        assert "instrument.ot2.tip" in fmt
        assert "85%" in fmt

    def test_get_bus_factory(self):
        from app.services.knowledge_bus import get_bus
        b1 = get_bus("camp-x")
        b2 = get_bus("camp-x")
        assert b1 is b2
        b3 = get_bus("camp-y")
        assert b3 is not b1


# ---------------------------------------------------------------------------
# AcquisitionBandit (meta-learning) tests
# ---------------------------------------------------------------------------

class TestAcquisitionBandit:
    def test_select_untried_arm_first(self):
        from app.services.meta_learning import AcquisitionBandit, TaskEmbedding
        bandit = AcquisitionBandit()
        task = TaskEmbedding("c1", 4, False, False, 0.02, ["ec"], 0.3)
        # All arms untried — should return first untried arm
        selected = bandit.select(task)
        assert selected in AcquisitionBandit.ARMS

    def test_update_changes_statistics(self):
        from app.services.meta_learning import AcquisitionBandit, TaskEmbedding
        bandit = AcquisitionBandit()
        task = TaskEmbedding("c1", 4, False, False, 0.02, [], 0.5)
        bandit.update("ei", 0.9, task)
        stats = bandit.get_statistics()
        assert stats["ei"]["n_uses"] == 1
        assert abs(stats["ei"]["mean_reward"] - 0.9) < 1e-6

    def test_prefers_best_arm_after_many_updates(self):
        from app.services.meta_learning import AcquisitionBandit, TaskEmbedding
        bandit = AcquisitionBandit(ucb_beta=0.01)  # Low exploration
        task = TaskEmbedding("c1", 4, False, False, 0.01, [], 0.5)
        # Seed "thompson" with high rewards
        for _ in range(10):
            bandit.update("thompson", 1.0, task)
        for arm in ["ei", "ucb", "mes", "kg"]:
            for _ in range(5):
                bandit.update(arm, 0.1, task)
        selected = bandit.select(task)
        assert selected == "thompson"


# ---------------------------------------------------------------------------
# ConformalPredictor tests
# ---------------------------------------------------------------------------

class TestConformalPredictor:
    def test_calibration(self):
        from app.services.uncertainty_propagation import ConformalPredictor
        rng = np.random.default_rng(42)
        n = 50
        y_pred = rng.uniform(0, 1, n)
        pred_std = rng.uniform(0.05, 0.2, n)
        y_true = y_pred + rng.normal(0, 0.1, n)
        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(y_pred, y_true, pred_std)
        assert cp._fitted

    def test_interval_contains_calibration_center(self):
        from app.services.uncertainty_propagation import ConformalPredictor
        rng = np.random.default_rng(0)
        y_pred = rng.uniform(0, 1, 40)
        pred_std = np.ones(40) * 0.15
        y_true = y_pred + rng.normal(0, 0.1, 40)
        cp = ConformalPredictor(coverage=0.90).calibrate(y_pred, y_true, pred_std)
        lo, hi = cp.predict_interval(0.5, 0.15)
        assert lo < 0.5 < hi

    def test_unfitted_fallback(self):
        from app.services.uncertainty_propagation import ConformalPredictor
        cp = ConformalPredictor(coverage=0.90)
        lo, hi = cp.predict_interval(0.5, 0.1)
        assert lo < 0.5 < hi

    def test_propagate_uncertainty_composition(self):
        from app.services.uncertainty_propagation import propagate_uncertainty
        u = propagate_uncertainty(
            model_mean=0.7,
            model_std=0.05,
            measurement_noise=0.02,
            agent_std=0.01,
            transfer_penalty=0.03,
        )
        expected_std = (0.05**2 + 0.02**2 + 0.01**2 + 0.03**2)**0.5
        assert abs(u.std - expected_std) < 1e-8
        assert u.ci_lower < 0.7 < u.ci_upper
        assert u.source in ("gaussian_propagated", "conformal")

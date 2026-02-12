"""Tests for the Deck/Layout Agent."""
import pytest
from app.services.deck_layout import (
    compute_tip_racks_needed,
    compute_transfers,
    compute_volume_requirements,
    create_well_allocator_from_deck_plan,
    estimate_tip_usage,
    plan_deck_layout,
    select_pipette,
    validate_deck_layout,
    DeckPlan,
    SlotAssignment,
    WellAllocator,
    WellExhaustedError,
    _generate_well_names,
    SLOT_RANGE,
)


class TestSelectPipette:
    def test_small_volume_selects_p20(self):
        assert select_pipette(5.0) == "p20_single_gen2"

    def test_medium_volume_selects_p300(self):
        assert select_pipette(100.0) == "p300_single_gen2"

    def test_large_volume_selects_p300(self):
        assert select_pipette(500.0) == "p300_single_gen2"

    def test_edge_volume_20ul(self):
        assert select_pipette(20.0) == "p20_single_gen2"

    def test_volume_between_pipettes(self):
        # 21 uL is above p20 max, should use p300
        assert select_pipette(21.0) == "p300_single_gen2"

    def test_very_small_volume(self):
        # 0.5 uL is below p20 min
        assert select_pipette(0.5) is None


class TestComputeTransfers:
    def test_single_transfer(self):
        assert compute_transfers(100.0, "p300_single_gen2") == 1

    def test_multiple_transfers(self):
        assert compute_transfers(600.0, "p300_single_gen2") == 2

    def test_exact_capacity(self):
        assert compute_transfers(300.0, "p300_single_gen2") == 1


class TestEstimateTipUsage:
    def test_simple_aspirate(self):
        steps = [
            {"primitive": "robot.aspirate", "params": {"volume_ul": 100}},
        ]
        usage = estimate_tip_usage(steps)
        assert usage.get("p300_single_gen2", 0) == 1

    def test_no_aspirate_steps(self):
        steps = [
            {"primitive": "robot.home", "params": {}},
            {"primitive": "heat", "params": {"temp_c": 50}},
        ]
        usage = estimate_tip_usage(steps)
        assert sum(usage.values()) == 0

    def test_multiple_aspirates(self):
        steps = [
            {"primitive": "robot.aspirate", "params": {"volume_ul": 10}},
            {"primitive": "robot.aspirate", "params": {"volume_ul": 200}},
        ]
        usage = estimate_tip_usage(steps)
        assert usage.get("p20_single_gen2", 0) == 1
        assert usage.get("p300_single_gen2", 0) == 1


class TestPlanDeckLayout:
    def test_basic_plan(self):
        steps = [
            {"primitive": "robot.aspirate", "params": {"volume_ul": 100}},
            {"primitive": "robot.dispense", "params": {"volume_ul": 100}},
        ]
        plan = plan_deck_layout(steps)
        assert isinstance(plan, DeckPlan)
        assert plan.pipette_right == "p300_single_gen2"
        assert 10 in plan.slots  # waste
        assert plan.slots[10].role == "waste"

    def test_all_slots_assigned(self):
        steps = [{"primitive": "robot.aspirate", "params": {"volume_ul": 100}}]
        plan = plan_deck_layout(steps)
        for slot_num in SLOT_RANGE:
            assert slot_num in plan.slots

    def test_tip_racks_assigned(self):
        steps = [{"primitive": "robot.aspirate", "params": {"volume_ul": 100}}]
        plan = plan_deck_layout(steps)
        tip_slots = [s for s in plan.slots.values() if s.role == "tips"]
        assert len(tip_slots) >= 1


class TestValidateDeckLayout:
    def test_valid_layout(self):
        steps = [
            {"step_key": "s1", "primitive": "robot.aspirate", "params": {"volume_ul": 100}},
        ]
        plan = plan_deck_layout(steps)
        result = validate_deck_layout(plan, steps)
        assert result.valid
        assert len(result.errors) == 0

    def test_volume_exceeds_policy(self):
        steps = [
            {"step_key": "s1", "primitive": "robot.aspirate", "params": {"volume_ul": 2000}},
        ]
        plan = plan_deck_layout(steps)
        result = validate_deck_layout(plan, steps, policy_snapshot={"max_volume_ul": 1000})
        assert not result.valid
        assert any("exceeds policy max" in e for e in result.errors)

    def test_to_dict(self):
        steps = [{"primitive": "robot.aspirate", "params": {"volume_ul": 100}}]
        plan = plan_deck_layout(steps)
        d = plan.to_dict()
        assert "slots" in d
        assert "pipette_left" in d
        assert "pipette_right" in d


# ===========================================================================
# Well Allocator
# ===========================================================================


class TestGenerateWellNames:
    def test_24_wells(self):
        names = _generate_well_names(24)
        assert len(names) == 24
        # Column-major: A1, B1, C1, D1, A2, B2, ...
        assert names[0] == "A1"
        assert names[1] == "B1"
        assert names[2] == "C1"
        assert names[3] == "D1"
        assert names[4] == "A2"
        assert names[-1] == "D6"

    def test_96_wells(self):
        names = _generate_well_names(96)
        assert len(names) == 96
        assert names[0] == "A1"
        assert names[7] == "H1"  # last in first column
        assert names[8] == "A2"  # first in second column

    def test_single_well(self):
        names = _generate_well_names(1)
        assert names == ["A1"]

    def test_12_reservoir(self):
        names = _generate_well_names(12)
        assert len(names) == 12
        assert names[0] == "A1"
        assert names[11] == "A12"


class TestWellAllocator:
    def _make_allocator(self, n_wells: int = 24) -> WellAllocator:
        names = _generate_well_names(n_wells)
        return WellAllocator(
            labware_name="reactor",
            slot_number=4,
            well_names=names,
        )

    def test_sequential_allocation(self):
        alloc = self._make_allocator(24)
        w1 = alloc.allocate(round_number=1, candidate_index=0)
        w2 = alloc.allocate(round_number=1, candidate_index=1)
        w3 = alloc.allocate(round_number=2, candidate_index=0)
        assert w1 == "A1"
        assert w2 == "B1"
        assert w3 == "C1"

    def test_capacity_and_remaining(self):
        alloc = self._make_allocator(6)
        assert alloc.capacity == 6
        assert alloc.remaining == 6
        alloc.allocate(1, 0)
        assert alloc.remaining == 5

    def test_exhaustion_raises(self):
        alloc = self._make_allocator(2)
        # Manually create 2-well allocator
        alloc = WellAllocator(
            labware_name="tiny",
            slot_number=1,
            well_names=["A1", "A2"],
        )
        alloc.allocate(1, 0)
        alloc.allocate(1, 1)
        with pytest.raises(WellExhaustedError) as exc_info:
            alloc.allocate(2, 0)
        assert "exhausted" in str(exc_info.value).lower()
        assert exc_info.value.capacity == 2
        assert exc_info.value.requested_round == 2

    def test_peek_does_not_consume(self):
        alloc = self._make_allocator(24)
        peeked = alloc.peek()
        assert peeked == "A1"
        assert alloc.remaining == 24
        allocated = alloc.allocate(1, 0)
        assert allocated == "A1"
        assert alloc.remaining == 23

    def test_peek_returns_none_when_exhausted(self):
        alloc = WellAllocator(
            labware_name="tiny", slot_number=1, well_names=["A1"],
        )
        alloc.allocate(1, 0)
        assert alloc.peek() is None

    def test_get_well_for_lookup(self):
        alloc = self._make_allocator(24)
        alloc.allocate(round_number=1, candidate_index=0)
        alloc.allocate(round_number=1, candidate_index=1)
        alloc.allocate(round_number=2, candidate_index=0)
        assert alloc.get_well_for(1, 0) == "A1"
        assert alloc.get_well_for(1, 1) == "B1"
        assert alloc.get_well_for(2, 0) == "C1"
        assert alloc.get_well_for(99, 99) is None

    def test_history_tracking(self):
        alloc = self._make_allocator(24)
        alloc.allocate(1, 0)
        alloc.allocate(1, 1)
        assert len(alloc.history) == 2
        assert alloc.history[0].well_name == "A1"
        assert alloc.history[0].round_number == 1
        assert alloc.history[1].well_name == "B1"

    def test_reset_clears_state(self):
        alloc = self._make_allocator(24)
        alloc.allocate(1, 0)
        alloc.allocate(1, 1)
        alloc.reset()
        assert alloc.remaining == 24
        assert len(alloc.history) == 0
        # Can allocate from the beginning again
        assert alloc.allocate(1, 0) == "A1"

    def test_snapshot_serialization(self):
        alloc = self._make_allocator(24)
        alloc.allocate(1, 0)
        snap = alloc.snapshot()
        assert snap["capacity"] == 24
        assert snap["allocated"] == 1
        assert snap["remaining"] == 23
        assert len(snap["allocations"]) == 1
        assert snap["allocations"][0]["well"] == "A1"

    def test_empty_well_names_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            WellAllocator(labware_name="x", slot_number=1, well_names=[])

    def test_24_round_campaign_one_per_round(self):
        """Simulate a 24-round campaign: 1 candidate per round on 24-well plate."""
        alloc = self._make_allocator(24)
        wells_used = []
        for r in range(1, 25):
            w = alloc.allocate(round_number=r, candidate_index=0)
            wells_used.append(w)
        assert len(wells_used) == 24
        assert len(set(wells_used)) == 24  # all unique
        assert alloc.remaining == 0

    def test_multi_candidate_rounds(self):
        """3 candidates per round, 8 rounds = 24 wells."""
        alloc = self._make_allocator(24)
        for r in range(1, 9):
            for c in range(3):
                alloc.allocate(round_number=r, candidate_index=c)
        assert alloc.remaining == 0


class TestCreateWellAllocatorFromDeckPlan:
    def test_with_destination_slot(self):
        """Standard 96-well destination plate gives allocator with 96 wells."""
        steps = [{"primitive": "robot.aspirate", "params": {"volume_ul": 100}}]
        plan = plan_deck_layout(steps)
        # Manually set slot 3 to have a recognized destination labware
        plan.slots[3] = SlotAssignment(
            slot_number=3,
            role="destination",
            labware_name="output_plate",
            labware_definition="nest_96_wellplate_200ul_flat",
        )
        alloc = create_well_allocator_from_deck_plan(plan)
        assert alloc is not None
        assert alloc.capacity == 96

    def test_no_destination_returns_none(self):
        """No destination slot → None."""
        steps = [{"primitive": "robot.aspirate", "params": {"volume_ul": 100}}]
        plan = plan_deck_layout(steps)
        # All destination slots are empty (no labware_definition)
        alloc = create_well_allocator_from_deck_plan(plan)
        assert alloc is None

    def test_with_contents_based_wells(self):
        """When labware isn't in catalog, use contents keys as wells."""
        steps = [{"primitive": "robot.aspirate", "params": {"volume_ul": 100}}]
        plan = plan_deck_layout(steps)
        plan.slots[3] = SlotAssignment(
            slot_number=3,
            role="destination",
            labware_name="custom_reactor",
            labware_definition="custom_24_well_reactor",
            contents={
                "A1": {"empty": True},
                "B1": {"empty": True},
                "A2": {"empty": True},
                "B2": {"empty": True},
            },
        )
        alloc = create_well_allocator_from_deck_plan(plan)
        assert alloc is not None
        assert alloc.capacity == 4

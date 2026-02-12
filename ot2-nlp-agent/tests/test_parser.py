"""Tests for the NL Parser."""

import pytest
from ot2_agent.parser import NLParser
from ot2_agent.operations import OperationType


@pytest.fixture
def parser():
    return NLParser()


class TestLanguageDetection:
    def test_detect_chinese(self, parser):
        assert parser.detect_language("从A1孔吸取100微升") == "zh"
        assert parser.detect_language("取枪头") == "zh"

    def test_detect_english(self, parser):
        assert parser.detect_language("aspirate 100ul from A1") == "en"
        assert parser.detect_language("pick up tip") == "en"

    def test_detect_mixed(self, parser):
        # Mixed content with more Chinese
        assert parser.detect_language("从slot 1吸取liquid") == "zh"


class TestVolumeExtraction:
    def test_chinese_microliter(self, parser):
        intent = parser.parse("吸取100微升")
        assert intent.params.get("volume") == 100
        assert intent.params.get("volume_unit") == "ul"

    def test_chinese_milliliter(self, parser):
        intent = parser.parse("吸取2毫升")
        assert intent.params.get("volume") == 2
        assert intent.params.get("volume_unit") == "ml"

    def test_english_ul(self, parser):
        intent = parser.parse("aspirate 50ul")
        assert intent.params.get("volume") == 50

    def test_decimal_volume(self, parser):
        intent = parser.parse("吸取2.5微升")
        assert intent.params.get("volume") == 2.5


class TestWellExtraction:
    def test_single_well(self, parser):
        intent = parser.parse("从A1孔吸取")
        assert "A1" in intent.params.get("location", "") or "A1" in intent.params.values()

    def test_multiple_wells(self, parser):
        intent = parser.parse("从A1吸取分配到B1")
        assert intent.params.get("source") == "A1"
        assert intent.params.get("destination") == "B1"

    def test_well_range(self, parser):
        intent = parser.parse("从A1到A12")
        well_range = intent.params.get("well_range")
        assert well_range is not None
        assert well_range["start"] == "A1"
        assert well_range["end"] == "A12"


class TestSlotExtraction:
    def test_chinese_slot(self, parser):
        intent = parser.parse("位置3的96孔板")
        assert intent.params.get("slot") == 3

    def test_english_slot(self, parser):
        intent = parser.parse("slot 5")
        assert intent.params.get("slot") == 5


class TestOperationDetection:
    def test_aspirate_chinese(self, parser):
        intent = parser.parse("吸取100微升")
        assert intent.operation_type == OperationType.ASPIRATE

    def test_dispense_chinese(self, parser):
        intent = parser.parse("分配50微升")
        assert intent.operation_type == OperationType.DISPENSE

    def test_transfer_chinese(self, parser):
        intent = parser.parse("转移液体")
        assert intent.operation_type == OperationType.TRANSFER

    def test_pick_up_tip_chinese(self, parser):
        intent = parser.parse("取枪头")
        assert intent.operation_type == OperationType.PICK_UP_TIP

    def test_drop_tip_chinese(self, parser):
        intent = parser.parse("丢弃枪头")
        assert intent.operation_type == OperationType.DROP_TIP

    def test_mix_chinese(self, parser):
        intent = parser.parse("混匀3次")
        assert intent.operation_type == OperationType.MIX

    def test_aspirate_english(self, parser):
        intent = parser.parse("aspirate 100ul")
        assert intent.operation_type == OperationType.ASPIRATE

    def test_dispense_english(self, parser):
        intent = parser.parse("dispense 50ul")
        assert intent.operation_type == OperationType.DISPENSE


class TestMultiStepParsing:
    def test_numbered_steps_chinese(self, parser):
        text = "第一步：取枪头。第二步：吸取100微升。第三步：分配。"
        intents = parser.parse_multi_step(text)
        assert len(intents) == 3

    def test_sequence_words_chinese(self, parser):
        text = "取枪头，然后吸取100微升，最后丢弃枪头"
        intents = parser.parse_multi_step(text)
        assert len(intents) >= 2

    def test_english_steps(self, parser):
        text = "Step 1: pick up tip. Step 2: aspirate 100ul."
        intents = parser.parse_multi_step(text)
        assert len(intents) == 2


class TestTimeExtraction:
    def test_seconds_chinese(self, parser):
        intent = parser.parse("等待30秒")
        assert intent.params.get("seconds") == 30

    def test_minutes_chinese(self, parser):
        intent = parser.parse("等待2分钟")
        assert intent.params.get("seconds") == 120


class TestRepetitionExtraction:
    def test_chinese_repetitions(self, parser):
        intent = parser.parse("混匀3次")
        assert intent.params.get("repetitions") == 3

    def test_english_repetitions(self, parser):
        intent = parser.parse("mix 5 times")
        assert intent.params.get("repetitions") == 5

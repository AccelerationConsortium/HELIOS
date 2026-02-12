"""Tests for the NL intent parsing module."""
from __future__ import annotations

import pytest
from app.api.v1.endpoints.nl_parse import parse_nl_text


# ---------------------------------------------------------------------------
# Direction extraction
# ---------------------------------------------------------------------------

class TestDirectionExtraction:
    def test_minimize_chinese(self):
        result = parse_nl_text("我想最小化CV")
        assert result.direction == "minimize"

    def test_maximize_chinese(self):
        result = parse_nl_text("目标是最大化absorbance")
        assert result.direction == "maximize"

    def test_minimize_english(self):
        result = parse_nl_text("I want to minimize the coefficient of variation")
        assert result.direction == "minimize"

    def test_maximize_english(self):
        result = parse_nl_text("Goal: maximize yield per batch")
        assert result.direction == "maximize"

    def test_less_than_implies_minimize(self):
        result = parse_nl_text("CV小于5%")
        assert result.direction == "minimize"

    def test_greater_than_implies_maximize(self):
        result = parse_nl_text("absorbance大于0.8")
        assert result.direction == "maximize"


# ---------------------------------------------------------------------------
# KPI extraction
# ---------------------------------------------------------------------------

class TestKPIExtraction:
    def test_cv_english(self):
        result = parse_nl_text("minimize CV over 10 rounds")
        assert result.objective_kpi == "cv"

    def test_cv_chinese(self):
        result = parse_nl_text("目标是变异系数小于3%")
        assert result.objective_kpi == "cv"

    def test_absorbance(self):
        result = parse_nl_text("maximize absorbance at 450nm")
        assert result.objective_kpi == "absorbance"

    def test_yield(self):
        result = parse_nl_text("提高产率")
        assert result.objective_kpi == "yield"

    def test_fluorescence(self):
        result = parse_nl_text("maximize fluorescence intensity")
        assert result.objective_kpi == "fluorescence"


# ---------------------------------------------------------------------------
# Numeric extraction
# ---------------------------------------------------------------------------

class TestNumericExtraction:
    def test_max_rounds_chinese(self):
        result = parse_nl_text("最多跑10轮")
        assert result.max_rounds == 10

    def test_max_rounds_english(self):
        result = parse_nl_text("run for max 8 rounds")
        assert result.max_rounds == 8

    def test_target_less_than(self):
        result = parse_nl_text("CV小于5.0")
        assert result.target_value == 5.0

    def test_target_english(self):
        result = parse_nl_text("target < 3.5%")
        assert result.target_value == 3.5

    def test_batch_size_chinese(self):
        result = parse_nl_text("每轮4个样品")
        assert result.batch_size == 4

    def test_batch_size_english(self):
        result = parse_nl_text("batch_size=6")
        assert result.batch_size == 6


# ---------------------------------------------------------------------------
# Strategy extraction
# ---------------------------------------------------------------------------

class TestStrategyExtraction:
    def test_bayesian(self):
        result = parse_nl_text("use Bayesian optimization strategy")
        assert result.strategy == "bayesian"

    def test_bayesian_chinese(self):
        result = parse_nl_text("使用贝叶斯策略")
        assert result.strategy == "bayesian"

    def test_lhs(self):
        result = parse_nl_text("use LHS sampling")
        assert result.strategy == "lhs"

    def test_random(self):
        result = parse_nl_text("随机采样")
        assert result.strategy == "random"


# ---------------------------------------------------------------------------
# Protocol pattern extraction
# ---------------------------------------------------------------------------

class TestProtocolExtraction:
    def test_serial_dilution(self):
        result = parse_nl_text("做一个serial dilution实验")
        assert result.protocol_pattern_id == "serial_dilution"

    def test_serial_dilution_chinese(self):
        result = parse_nl_text("连续稀释实验")
        assert result.protocol_pattern_id == "serial_dilution"

    def test_mixing(self):
        result = parse_nl_text("mixing experiment")
        assert result.protocol_pattern_id == "mixing"


# ---------------------------------------------------------------------------
# Slot extraction
# ---------------------------------------------------------------------------

class TestSlotExtraction:
    def test_slot_assignment(self):
        result = parse_nl_text("slot 1放tip_rack，slot 2放reservoir")
        assert "slot_1" in result.slot_assignments
        assert "slot_2" in result.slot_assignments

    def test_slot_english(self):
        result = parse_nl_text("slot 3: plate_96")
        assert "slot_3" in result.slot_assignments


# ---------------------------------------------------------------------------
# Dimension extraction
# ---------------------------------------------------------------------------

class TestDimensionExtraction:
    def test_volume_range(self):
        result = parse_nl_text("volume 1-50uL")
        assert len(result.dimensions) >= 1
        dim = result.dimensions[0]
        assert dim["name"] == "volume"
        assert dim["low"] == 1.0
        assert dim["high"] == 50.0

    def test_range_with_unit(self):
        result = parse_nl_text("flow_rate 10-100uL/s")
        assert len(result.dimensions) >= 1

    def test_multiple_dimensions(self):
        result = parse_nl_text("volume 1-50uL, flow_rate 10-100uL/s")
        assert len(result.dimensions) >= 2


# ---------------------------------------------------------------------------
# Full integration: mixed Chinese + English
# ---------------------------------------------------------------------------

class TestFullParsing:
    def test_mixed_chinese_english(self):
        text = (
            "我要做一个serial dilution实验，用96孔板，"
            "slot 1放tip_rack，slot 2放reservoir，slot 3放plate。"
            "volume从200到12.5uL，8个梯度。"
            "目标是CV小于5，最多跑10轮，每轮4个样品。"
        )
        result = parse_nl_text(text)
        assert result.protocol_pattern_id == "serial_dilution"
        assert result.objective_kpi == "cv"
        assert result.direction == "minimize"
        assert result.max_rounds == 10
        assert result.target_value == 5.0
        assert result.batch_size == 4
        assert len(result.slot_assignments) >= 2

    def test_english_only(self):
        text = (
            "I want to optimize a dispensing protocol. "
            "Parameter: volume 1-50uL, flow_rate 10-100uL/s. "
            "Minimize CV, target < 3, max 8 rounds, batch size 6. "
            "Use Bayesian strategy."
        )
        result = parse_nl_text(text)
        assert result.objective_kpi == "cv"
        assert result.direction == "minimize"
        assert result.max_rounds == 8
        assert result.target_value == 3.0
        assert result.batch_size == 6
        assert result.strategy == "bayesian"
        assert len(result.dimensions) >= 1

    def test_empty_text(self):
        result = parse_nl_text("")
        assert result.objective_kpi is None
        assert result.direction is None
        assert result.max_rounds is None

    def test_original_text_preserved(self):
        result = parse_nl_text("hello world")
        assert result.original_text == "hello world"

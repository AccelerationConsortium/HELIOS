"""
Liquid Handler Natural Language Parser

Parses natural language instructions for liquid handling operations.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from ...core.plugin_base import ParserBase
from .operations import LiquidOperation, LIQUID_OPERATIONS


class LiquidHandlerParser(ParserBase):
    """
    Natural language parser for liquid handling instructions.

    Supports both English and Chinese instructions for operations like:
    - aspirate, dispense, transfer
    - mix, pick up tip, drop tip
    - wait, pause, etc.
    """

    # Volume patterns
    VOLUME_PATTERNS = [
        r'(\d+(?:\.\d+)?)\s*(?:ul|µl|microliter|microliters)',
        r'(\d+(?:\.\d+)?)\s*(?:ml|milliliter|milliliters)',
        r'(\d+(?:\.\d+)?)\s*微升',
        r'(\d+(?:\.\d+)?)\s*毫升',
    ]

    # Time patterns
    TIME_PATTERNS = [
        r'(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)',
        r'(\d+(?:\.\d+)?)\s*(?:m|min|minute|minutes)',
        r'(\d+(?:\.\d+)?)\s*秒',
        r'(\d+(?:\.\d+)?)\s*分钟',
    ]

    # Repetition patterns
    REP_PATTERNS = [
        r'(\d+)\s*(?:times|x|repetitions|reps)',
        r'(\d+)\s*次',
        r'repeat\s*(\d+)',
        r'重复\s*(\d+)',
    ]

    def __init__(self):
        self._operations = LIQUID_OPERATIONS

    def parse(self, instruction: str) -> Dict[str, Any]:
        """
        Parse a natural language instruction.

        Args:
            instruction: Natural language text

        Returns:
            Dict with operation, action, params, confidence, language
        """
        language = self.detect_language(instruction)
        text_lower = instruction.lower()

        # Find matching operation
        best_match = None
        best_confidence = 0.0

        for op_type, op_def in self._operations.items():
            matches, confidence = op_def.matches(instruction, language)
            if matches and confidence > best_confidence:
                best_match = op_type
                best_confidence = confidence

        if not best_match:
            return {
                "operation": None,
                "action": None,
                "params": {},
                "confidence": 0.0,
                "language": language,
                "description": instruction,
            }

        # Extract parameters based on operation type
        params = self._extract_params(instruction, best_match, language)

        # Build result
        op_def = self._operations[best_match]
        return {
            "operation": best_match.value,
            "action": op_def.action,
            "params": params,
            "confidence": best_confidence,
            "language": language,
            "description": instruction,
        }

    def _extract_params(
        self,
        instruction: str,
        op_type: LiquidOperation,
        language: str
    ) -> Dict[str, Any]:
        """Extract parameters from instruction based on operation type."""
        params = {}

        # Volume extraction
        if op_type in [
            LiquidOperation.ASPIRATE,
            LiquidOperation.DISPENSE,
            LiquidOperation.TRANSFER,
            LiquidOperation.DISTRIBUTE,
            LiquidOperation.CONSOLIDATE,
            LiquidOperation.MIX,
            LiquidOperation.AIR_GAP,
        ]:
            volume = self._extract_volume(instruction)
            if volume:
                params['volume'] = volume

        # Well/location extraction
        if op_type in [
            LiquidOperation.ASPIRATE,
            LiquidOperation.DISPENSE,
            LiquidOperation.MIX,
            LiquidOperation.PICK_UP_TIP,
            LiquidOperation.DROP_TIP,
            LiquidOperation.TOUCH_TIP,
            LiquidOperation.BLOW_OUT,
            LiquidOperation.MOVE_TO,
        ]:
            well = self.extract_well(instruction)
            if well:
                params['location'] = well

        # Source/destination for transfer
        if op_type == LiquidOperation.TRANSFER:
            source, dest = self._extract_source_dest(instruction, language)
            if source:
                params['source'] = source
            if dest:
                params['destination'] = dest

        # Well range for distribute/consolidate
        if op_type in [LiquidOperation.DISTRIBUTE, LiquidOperation.CONSOLIDATE]:
            wells = self.extract_well_range(instruction)
            if wells:
                if op_type == LiquidOperation.DISTRIBUTE:
                    params['destinations'] = wells
                else:
                    params['sources'] = wells

        # Repetitions for mix
        if op_type == LiquidOperation.MIX:
            reps = self._extract_repetitions(instruction)
            if reps:
                params['repetitions'] = reps

        # Time for wait
        if op_type == LiquidOperation.WAIT:
            duration = self._extract_time(instruction)
            if duration:
                params['duration_seconds'] = duration

        return params

    def _extract_volume(self, text: str) -> Optional[float]:
        """Extract volume in microliters."""
        for pattern in self.VOLUME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                # Convert ml to ul
                if 'ml' in pattern.lower() or '毫升' in pattern:
                    value *= 1000
                return value
        return None

    def _extract_time(self, text: str) -> Optional[float]:
        """Extract time in seconds."""
        for pattern in self.TIME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                # Convert minutes to seconds
                if 'min' in pattern.lower() or '分钟' in pattern:
                    value *= 60
                return value
        return None

    def _extract_repetitions(self, text: str) -> Optional[int]:
        """Extract number of repetitions."""
        for pattern in self.REP_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _extract_source_dest(
        self,
        text: str,
        language: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract source and destination wells for transfer."""
        source = None
        dest = None

        # English patterns
        en_patterns = [
            r'from\s+([A-Ha-h][1-9]|[A-Ha-h]1[0-2])\s+to\s+([A-Ha-h][1-9]|[A-Ha-h]1[0-2])',
            r'([A-Ha-h][1-9]|[A-Ha-h]1[0-2])\s+to\s+([A-Ha-h][1-9]|[A-Ha-h]1[0-2])',
        ]

        # Chinese patterns
        zh_patterns = [
            r'从\s*([A-Ha-h][1-9]|[A-Ha-h]1[0-2])\s*到\s*([A-Ha-h][1-9]|[A-Ha-h]1[0-2])',
            r'([A-Ha-h][1-9]|[A-Ha-h]1[0-2])\s*[到至]\s*([A-Ha-h][1-9]|[A-Ha-h]1[0-2])',
        ]

        patterns = zh_patterns if language == "zh" else en_patterns

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                source = match.group(1).upper()
                dest = match.group(2).upper()
                break

        # Fallback: just find wells in order
        if not source or not dest:
            wells = re.findall(r'\b([A-Ha-h][1-9]|[A-Ha-h]1[0-2])\b', text, re.IGNORECASE)
            if len(wells) >= 2:
                source = wells[0].upper()
                dest = wells[1].upper()
            elif len(wells) == 1:
                # Ambiguous - could be source or dest
                if 'from' in text.lower() or '从' in text:
                    source = wells[0].upper()
                else:
                    dest = wells[0].upper()

        return source, dest

"""
Natural Language Parser for OT-2 commands.
Supports English and Chinese instructions.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .operations import Operation, OperationMapper, OperationType


@dataclass
class ParsedIntent:
    """Parsed user intent from natural language."""
    operation_type: Optional[OperationType]
    params: Dict[str, Any]
    original_text: str
    confidence: float
    language: str  # 'en' or 'zh'


class NLParser:
    """
    Natural Language Parser for OT-2 instructions.

    Parses natural language commands into structured operations.
    Supports both English and Chinese.

    Example:
        parser = NLParser()
        intent = parser.parse("从A1孔吸取100微升")
        # intent.operation_type = OperationType.ASPIRATE
        # intent.params = {'volume': 100, 'location': 'A1', 'unit': 'ul'}
    """

    # Volume patterns
    VOLUME_PATTERNS = [
        # Chinese patterns
        (r'(\d+(?:\.\d+)?)\s*微升', 'ul'),
        (r'(\d+(?:\.\d+)?)\s*毫升', 'ml'),
        (r'(\d+(?:\.\d+)?)\s*ul', 'ul'),
        (r'(\d+(?:\.\d+)?)\s*μl', 'ul'),
        (r'(\d+(?:\.\d+)?)\s*ml', 'ml'),
        # English patterns
        (r'(\d+(?:\.\d+)?)\s*microliter', 'ul'),
        (r'(\d+(?:\.\d+)?)\s*milliliter', 'ml'),
    ]

    # Well patterns (A1, B2, etc.)
    WELL_PATTERNS = [
        r'([A-H])(\d{1,2})',  # Standard: A1, B12
        r'([A-H])\s*-\s*(\d{1,2})',  # With dash: A-1
        r'孔\s*([A-H])(\d{1,2})',  # Chinese: 孔A1
        r'([A-H])(\d{1,2})\s*孔',  # Chinese: A1孔
    ]

    # Well range patterns (A1-A12, B1:B8)
    WELL_RANGE_PATTERNS = [
        r'([A-H])(\d{1,2})\s*[-到至:]\s*([A-H])(\d{1,2})',  # A1-A12, A1到A12
        r'从\s*([A-H])(\d{1,2})\s*到\s*([A-H])(\d{1,2})',  # 从A1到A12
    ]

    # Slot patterns
    SLOT_PATTERNS = [
        r'slot\s*(\d{1,2})',  # slot 1, slot1
        r'位置\s*(\d{1,2})',  # 位置1
        r'槽位\s*(\d{1,2})',  # 槽位1
        r'(\d{1,2})\s*号位',  # 1号位
    ]

    # Time patterns
    TIME_PATTERNS = [
        (r'(\d+(?:\.\d+)?)\s*秒', 'seconds'),
        (r'(\d+(?:\.\d+)?)\s*分钟', 'minutes'),
        (r'(\d+(?:\.\d+)?)\s*seconds?', 'seconds'),
        (r'(\d+(?:\.\d+)?)\s*minutes?', 'minutes'),
        (r'(\d+(?:\.\d+)?)\s*s\b', 'seconds'),
        (r'(\d+(?:\.\d+)?)\s*min', 'minutes'),
    ]

    # Repetition patterns
    REPETITION_PATTERNS = [
        r'(\d+)\s*次',  # 3次
        r'(\d+)\s*times?',  # 3 times
        r'重复\s*(\d+)',  # 重复3
        r'repeat\s*(\d+)',  # repeat 3
    ]

    # Temperature patterns
    TEMPERATURE_PATTERNS = [
        r'(\d+(?:\.\d+)?)\s*[°度]?\s*[Cc摄氏]',  # 37°C, 37度, 37摄氏度
        r'(\d+(?:\.\d+)?)\s*celsius',
    ]

    def __init__(self):
        self.mapper = OperationMapper()

    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Chinese or English."""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return 'en'
        return 'zh' if chinese_chars / total_chars > 0.3 else 'en'

    def parse(self, text: str) -> ParsedIntent:
        """
        Parse natural language text into a structured intent.

        Args:
            text: Natural language instruction

        Returns:
            ParsedIntent with operation type and parameters
        """
        language = self.detect_language(text)
        operation_type = self.mapper.detect_operation_type(text)

        params = {}
        confidence = 0.0

        # Extract volume
        volume_info = self._extract_volume(text)
        if volume_info:
            params['volume'] = volume_info[0]
            params['volume_unit'] = volume_info[1]
            confidence += 0.2

        # Extract well locations
        wells = self._extract_wells(text)
        if wells:
            if len(wells) == 1:
                params['location'] = wells[0]
            elif len(wells) >= 2:
                params['source'] = wells[0]
                params['destination'] = wells[1] if len(wells) == 2 else wells[1:]
            confidence += 0.2

        # Extract well ranges
        well_range = self._extract_well_range(text)
        if well_range:
            params['well_range'] = well_range
            confidence += 0.1

        # Extract slot
        slot = self._extract_slot(text)
        if slot:
            params['slot'] = slot
            confidence += 0.1

        # Extract time
        time_info = self._extract_time(text)
        if time_info:
            params['seconds'] = time_info[0] if time_info[1] == 'seconds' else time_info[0] * 60
            confidence += 0.1

        # Extract repetitions
        reps = self._extract_repetitions(text)
        if reps:
            params['repetitions'] = reps
            confidence += 0.1

        # Extract temperature
        temp = self._extract_temperature(text)
        if temp:
            params['temperature'] = temp
            confidence += 0.1

        # Adjust confidence based on operation type detection
        if operation_type:
            confidence += 0.3

        return ParsedIntent(
            operation_type=operation_type,
            params=params,
            original_text=text,
            confidence=min(confidence, 1.0),
            language=language
        )

    def parse_multi_step(self, text: str) -> List[ParsedIntent]:
        """
        Parse multi-step instructions.

        Handles instructions like:
        - "第一步...第二步..."
        - "Step 1...Step 2..."
        - "首先...然后...最后..."
        """
        # Split by step markers
        step_patterns = [
            r'第[一二三四五六七八九十\d]+步[：:]?\s*',  # 第一步：
            r'步骤\s*\d+[：:]?\s*',  # 步骤1：
            r'Step\s*\d+[：:]?\s*',  # Step 1:
            r'\d+[\.、\)]\s*',  # 1. or 1、 or 1)
        ]

        # Try splitting by explicit steps
        for pattern in step_patterns:
            parts = re.split(pattern, text, flags=re.IGNORECASE)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                return [self.parse(part) for part in parts]

        # Try splitting by sequence words
        sequence_patterns = [
            r'[，,]\s*然后\s*',  # 然后
            r'[，,]\s*接着\s*',  # 接着
            r'[，,]\s*最后\s*',  # 最后
            r'[，,]?\s*then\s+',  # then
            r'[，,]?\s*next\s+',  # next
            r'[，,]?\s*finally\s+',  # finally
        ]

        for pattern in sequence_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                parts = re.split(pattern, text, flags=re.IGNORECASE)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) > 1:
                    return [self.parse(part) for part in parts]

        # No multi-step markers found, treat as single step
        return [self.parse(text)]

    def _extract_volume(self, text: str) -> Optional[Tuple[float, str]]:
        """Extract volume and unit from text."""
        for pattern, unit in self.VOLUME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return (float(match.group(1)), unit)
        return None

    def _extract_wells(self, text: str) -> List[str]:
        """Extract well positions from text."""
        wells = []
        for pattern in self.WELL_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    well = f"{match[0].upper()}{match[1]}"
                else:
                    well = match.upper()
                if well not in wells:
                    wells.append(well)
        return wells

    def _extract_well_range(self, text: str) -> Optional[Dict[str, str]]:
        """Extract well range (e.g., A1-A12) from text."""
        for pattern in self.WELL_RANGE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {
                    'start': f"{match.group(1).upper()}{match.group(2)}",
                    'end': f"{match.group(3).upper()}{match.group(4)}"
                }
        return None

    def _extract_slot(self, text: str) -> Optional[int]:
        """Extract slot number from text."""
        for pattern in self.SLOT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                slot = int(match.group(1))
                if 1 <= slot <= 11:  # OT-2 has slots 1-11
                    return slot
        return None

    def _extract_time(self, text: str) -> Optional[Tuple[float, str]]:
        """Extract time duration from text."""
        for pattern, unit in self.TIME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return (float(match.group(1)), unit)
        return None

    def _extract_repetitions(self, text: str) -> Optional[int]:
        """Extract repetition count from text."""
        for pattern in self.REPETITION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _extract_temperature(self, text: str) -> Optional[float]:
        """Extract temperature from text."""
        for pattern in self.TEMPERATURE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

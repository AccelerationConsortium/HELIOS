"""
Potentiostat Natural Language Parser

Parses natural language instructions for electrochemistry operations.
"""

import re
from typing import Any, Dict, Optional
from ...core.plugin_base import ParserBase
from .operations import ElectrochemOperation, ELECTROCHEM_OPERATIONS


class PotentiostatParser(ParserBase):
    """
    Natural language parser for electrochemistry instructions.

    Supports operations like:
    - EIS, CV, OCV, CP, CA
    - Deposition/dissolution
    - Data saving and plotting
    """

    # Frequency patterns
    FREQ_PATTERNS = [
        r'(\d+(?:\.\d+)?)\s*(?:khz|kHz)',
        r'(\d+(?:\.\d+)?)\s*(?:hz|Hz)',
        r'(\d+(?:\.\d+)?)\s*(?:mhz|mHz)',
    ]

    # Voltage patterns
    VOLTAGE_PATTERNS = [
        r'(\-?\d+(?:\.\d+)?)\s*(?:v|V|volt|volts)',
        r'(\-?\d+(?:\.\d+)?)\s*(?:mv|mV)',
    ]

    # Current patterns
    CURRENT_PATTERNS = [
        r'(\-?\d+(?:\.\d+)?)\s*(?:a|A|amp|amps)',
        r'(\-?\d+(?:\.\d+)?)\s*(?:ma|mA)',
        r'(\-?\d+(?:\.\d+)?)\s*(?:ua|µa|µA|uA)',
    ]

    # Time patterns
    TIME_PATTERNS = [
        r'(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)',
        r'(\d+(?:\.\d+)?)\s*(?:m|min|minute|minutes)',
        r'(\d+(?:\.\d+)?)\s*秒',
    ]

    def __init__(self):
        self._operations = ELECTROCHEM_OPERATIONS

    def parse(self, instruction: str) -> Dict[str, Any]:
        """Parse electrochemistry instruction."""
        language = self.detect_language(instruction)

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

        # Extract parameters
        params = self._extract_params(instruction, best_match, language)

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
        op_type: ElectrochemOperation,
        language: str
    ) -> Dict[str, Any]:
        """Extract parameters based on operation type."""
        params = {}

        if op_type == ElectrochemOperation.EIS:
            params.update(self._extract_eis_params(instruction))

        elif op_type == ElectrochemOperation.CV:
            params.update(self._extract_cv_params(instruction))

        elif op_type == ElectrochemOperation.CP:
            params.update(self._extract_cp_params(instruction))

        elif op_type == ElectrochemOperation.CA:
            params.update(self._extract_ca_params(instruction))

        elif op_type == ElectrochemOperation.OCV:
            params.update(self._extract_ocv_params(instruction))

        elif op_type in [ElectrochemOperation.LSV, ElectrochemOperation.DPV, ElectrochemOperation.SWV]:
            params.update(self._extract_sweep_params(instruction))

        return params

    def _extract_eis_params(self, text: str) -> Dict[str, Any]:
        """Extract EIS parameters."""
        params = {}

        # Frequency range: "from 10kHz to 0.1Hz"
        freq_match = re.search(
            r'from\s*(\d+(?:\.\d+)?)\s*(k?hz)\s*to\s*(\d+(?:\.\d+)?)\s*(m?hz)',
            text, re.IGNORECASE
        )
        if freq_match:
            start_val, start_unit, stop_val, stop_unit = freq_match.groups()
            start_hz = float(start_val) * (1000 if 'k' in start_unit.lower() else 1)
            stop_hz = float(stop_val) * (0.001 if 'm' in stop_unit.lower() else 1)
            params['freq_start_hz'] = start_hz
            params['freq_stop_hz'] = stop_hz

        # Amplitude
        amp_match = re.search(r'(\d+(?:\.\d+)?)\s*mv\s*amplitude', text, re.IGNORECASE)
        if amp_match:
            params['amplitude_v'] = float(amp_match.group(1)) / 1000

        return params

    def _extract_cv_params(self, text: str) -> Dict[str, Any]:
        """Extract CV parameters."""
        params = {}

        # Scan rate
        rate_match = re.search(r'(\d+(?:\.\d+)?)\s*mv/s', text, re.IGNORECASE)
        if rate_match:
            params['scan_rate_v_s'] = float(rate_match.group(1)) / 1000

        # Cycles
        cycle_match = re.search(r'(\d+)\s*(?:cycles?|scans?)', text, re.IGNORECASE)
        if cycle_match:
            params['cycles'] = int(cycle_match.group(1))

        # Voltage range
        voltages = self._extract_voltages(text)
        if len(voltages) >= 2:
            params['start_v'] = min(voltages)
            params['vertex1_v'] = max(voltages)
            params['vertex2_v'] = min(voltages)
            params['end_v'] = min(voltages)

        return params

    def _extract_cp_params(self, text: str) -> Dict[str, Any]:
        """Extract CP (chronopotentiometry) parameters."""
        params = {}

        # Current
        current = self._extract_current(text)
        if current is not None:
            params['current_a'] = current

        # Duration
        duration = self._extract_time(text)
        if duration is not None:
            params['duration_s'] = duration

        return params

    def _extract_ca_params(self, text: str) -> Dict[str, Any]:
        """Extract CA (chronoamperometry) parameters."""
        params = {}

        # Voltage
        voltages = self._extract_voltages(text)
        if voltages:
            params['voltage_v'] = voltages[0]

        # Duration
        duration = self._extract_time(text)
        if duration is not None:
            params['duration_s'] = duration

        return params

    def _extract_ocv_params(self, text: str) -> Dict[str, Any]:
        """Extract OCV parameters."""
        params = {}

        # Duration
        duration = self._extract_time(text)
        if duration is not None:
            params['duration_s'] = duration

        return params

    def _extract_sweep_params(self, text: str) -> Dict[str, Any]:
        """Extract linear sweep parameters."""
        params = {}

        voltages = self._extract_voltages(text)
        if len(voltages) >= 2:
            params['start_v'] = voltages[0]
            params['end_v'] = voltages[1]

        # Scan rate
        rate_match = re.search(r'(\d+(?:\.\d+)?)\s*mv/s', text, re.IGNORECASE)
        if rate_match:
            params['scan_rate_v_s'] = float(rate_match.group(1)) / 1000

        return params

    def _extract_voltages(self, text: str) -> list:
        """Extract all voltage values from text."""
        voltages = []
        for pattern in self.VOLTAGE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                val = float(match.group(1))
                if 'mv' in pattern.lower():
                    val /= 1000
                voltages.append(val)
        return voltages

    def _extract_current(self, text: str) -> Optional[float]:
        """Extract current value in Amps."""
        for pattern in self.CURRENT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if 'ma' in pattern.lower():
                    val /= 1000
                elif 'ua' in pattern.lower() or 'µa' in pattern.lower():
                    val /= 1000000
                return val
        return None

    def _extract_time(self, text: str) -> Optional[float]:
        """Extract time in seconds."""
        for pattern in self.TIME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if 'min' in pattern.lower() or '分' in pattern:
                    val *= 60
                return val
        return None

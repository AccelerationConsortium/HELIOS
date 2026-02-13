"""
pH Sensor Controller for colorimetric strip-based pH measurement.

Wraps the pHAnalyzer from pH_measure.pizerocam for integration with
the NeverOT ActionDispatcher. Provides dispense_strip, read_ph, and
calibrate operations.
"""
import logging
import time
from typing import Any, Optional

# The pHAnalyzer import is only available when the real hardware SDK is installed.
# In simulated / dry-run mode the module is never imported.
try:
    from pH_measure.pizerocam.src.image_req_client.ph_analyzer_new_0_6range import (
        pHAnalyzer,
    )
except ImportError:
    pHAnalyzer = None  # type: ignore[assignment,misc]

LOGGER = logging.getLogger(__name__)


class PhSensorController:
    """
    High-level controller for colorimetric pH strip measurement.

    Uses the pHAnalyzer hardware (Raspberry Pi Zero camera + strip dispenser)
    to measure pH via image analysis of indicator strips.

    Typical workflow per well:
        1. dispense_strip()  — advance a fresh indicator strip
        2. [robot dispenses sample onto strip]
        3. read_ph(well=...)  — capture image and compute pH
        4. dispense_strip()  — eject used strip
    """

    def __init__(
        self,
        repeat_reads: int = 3,
        settle_time_s: float = 2.0,
        dry_run: bool = False,
    ):
        """
        Initialize pH sensor controller.

        Args:
            repeat_reads: Number of repeated readings per measurement (averaged).
            settle_time_s: Seconds to wait after dispensing before reading.
            dry_run: If True, simulate all operations without real hardware.
        """
        self.repeat_reads = repeat_reads
        self.settle_time_s = settle_time_s
        self.dry_run = dry_run
        self._analyzer: Any = None

        if dry_run:
            LOGGER.info("[pH] Initialized in DRY-RUN mode (no hardware)")
        else:
            self._connect()

    def _connect(self):
        """Establish connection to the pHAnalyzer hardware."""
        if pHAnalyzer is None:
            LOGGER.warning(
                "[pH] pHAnalyzer SDK not installed — falling back to dry-run mode"
            )
            self.dry_run = True
            return

        try:
            self._analyzer = pHAnalyzer()
            LOGGER.info("[pH] Connected to pHAnalyzer hardware")
        except Exception as e:
            LOGGER.error(f"[pH] Failed to connect to pHAnalyzer: {e}")
            LOGGER.warning("[pH] Falling back to dry-run mode")
            self.dry_run = True

    def is_connected(self) -> bool:
        """Check if pH analyzer hardware is available."""
        return self._analyzer is not None and not self.dry_run

    def dispense_strip(self) -> bool:
        """
        Advance/eject a pH indicator strip.

        Call before dispensing sample (to present fresh strip) and
        after reading (to eject used strip).

        Returns:
            True if successful.
        """
        if self.dry_run:
            LOGGER.info("[pH][DRY-RUN] Strip dispensed (simulated)")
            return True

        try:
            self._analyzer.dispense_strip()
            LOGGER.info("[pH] Strip dispensed")
            return True
        except Exception as e:
            LOGGER.error(f"[pH] Failed to dispense strip: {e}")
            return False

    def read_ph(
        self,
        well: str = "A1",
        repeat: Optional[int] = None,
        settle_time: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Read pH value from the indicator strip.

        Captures multiple images and returns statistics.

        Args:
            well: Well identifier (for metadata tracking).
            repeat: Override default repeat_reads count.
            settle_time: Override default settle_time_s.

        Returns:
            Dict with keys:
                - ph_mean: Average pH across readings
                - ph_std: Standard deviation
                - ph_readings: List of individual readings
                - well: Well identifier
                - n_readings: Number of valid readings
        """
        n = repeat if repeat is not None else self.repeat_reads
        wait = settle_time if settle_time is not None else self.settle_time_s

        if self.dry_run:
            # Deterministic simulation: pH ~ 4.5 for "pure acid" wells
            import random
            simulated = [4.5 + random.gauss(0, 0.1) for _ in range(n)]
            mean_ph = sum(simulated) / len(simulated)
            std_ph = (sum((x - mean_ph) ** 2 for x in simulated) / len(simulated)) ** 0.5
            LOGGER.info(f"[pH][DRY-RUN] Well {well}: pH={mean_ph:.2f} ± {std_ph:.2f}")
            return {
                "ph_mean": round(mean_ph, 3),
                "ph_std": round(std_ph, 4),
                "ph_readings": [round(v, 3) for v in simulated],
                "well": well,
                "n_readings": n,
            }

        # Wait for dye to develop on strip
        LOGGER.info(f"[pH] Waiting {wait}s for strip color development...")
        time.sleep(wait)

        readings: list[float] = []
        for i in range(n):
            try:
                result = self._analyzer.read_ph(well=well)
                # pHAnalyzer may return float or dict{'ph': float}
                if isinstance(result, dict) and "ph" in result:
                    ph_val = float(result["ph"])
                else:
                    ph_val = float(result)
                readings.append(ph_val)
                LOGGER.debug(f"[pH] Reading {i+1}/{n} for {well}: {ph_val:.3f}")
            except Exception as e:
                LOGGER.warning(f"[pH] Reading {i+1}/{n} failed: {e}")

        if not readings:
            LOGGER.error(f"[pH] All readings failed for well {well}")
            return {
                "ph_mean": None,
                "ph_std": None,
                "ph_readings": [],
                "well": well,
                "n_readings": 0,
            }

        mean_ph = sum(readings) / len(readings)
        std_ph = (
            (sum((x - mean_ph) ** 2 for x in readings) / len(readings)) ** 0.5
            if len(readings) > 1
            else 0.0
        )

        LOGGER.info(
            f"[pH] Well {well}: pH={mean_ph:.3f} ± {std_ph:.4f} ({len(readings)} readings)"
        )
        return {
            "ph_mean": round(mean_ph, 3),
            "ph_std": round(std_ph, 4),
            "ph_readings": [round(v, 3) for v in readings],
            "well": well,
            "n_readings": len(readings),
        }

    def calibrate(
        self,
        known_ph: float,
        well: str = "A1",
    ) -> dict[str, Any]:
        """
        Run a calibration check against a known pH standard.

        Args:
            known_ph: Expected pH value of calibration buffer.
            well: Well identifier for logging.

        Returns:
            Dict with measured vs expected pH and offset.
        """
        result = self.read_ph(well=well, repeat=5)
        measured = result["ph_mean"]

        if measured is None:
            return {
                "success": False,
                "known_ph": known_ph,
                "measured_ph": None,
                "offset": None,
                "well": well,
            }

        offset = measured - known_ph
        success = abs(offset) < 0.5  # acceptable calibration tolerance

        LOGGER.info(
            f"[pH] Calibration: known={known_ph:.2f}, "
            f"measured={measured:.3f}, offset={offset:+.3f}, "
            f"{'PASS' if success else 'FAIL'}"
        )

        return {
            "success": success,
            "known_ph": known_ph,
            "measured_ph": measured,
            "offset": round(offset, 4),
            "well": well,
        }

    def close(self):
        """Release hardware resources."""
        if self._analyzer is not None:
            LOGGER.info("[pH] Sensor controller closed")
        self._analyzer = None

"""
Flex SSH Bridge — wraps matterlab_opentrons.OpenTrons for Flex robot control.

The Opentrons Flex uses SSH-based communication via the matterlab_opentrons
library, unlike the OT-2 which uses the HTTP API.  This bridge provides a
unified interface compatible with NeverOT's ActionDispatcher.

Typical usage:
    bridge = FlexBridge(host_alias="otflex", simulation=True)
    bridge.load_labware({
        "nickname": "plate_96_1",
        "loadname": "corning_96_wellplate_360ul_flat",
        "location": "C2",
        "ot_default": True,
        "config": {},
    })
    bridge.pick_up_tip(pip_name="p1000")
    bridge.aspirate(pip_name="p1000", volume=100)
    bridge.dispense(pip_name="p1000", volume=100)
    bridge.blow_out(pip_name="p1000")
    bridge.drop_tip(pip_name="p1000")
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

# matterlab_opentrons is only available in the Flex lab environment.
try:
    from matterlab_opentrons import OpenTrons
except ImportError:
    OpenTrons = None  # type: ignore[assignment,misc]

LOGGER = logging.getLogger(__name__)


class FlexBridge:
    """
    SSH-based bridge for Opentrons Flex robot.

    Wraps the matterlab_opentrons.OpenTrons class to provide a consistent
    interface for the NeverOT dispatcher.  All methods mirror the real
    OpenTronsControl.py API signatures so that protocol scripts can be
    translated 1-to-1.
    """

    def __init__(
        self,
        host_alias: str = "otflex",
        password: Optional[str] = None,
        simulation: bool = True,
        api_version: str = "2.21",
        dry_run: bool = False,
    ):
        """
        Initialize the Flex bridge.

        Args:
            host_alias: SSH alias for the Flex robot.
            password: SSH password (reads OPENTRONS_PASSWORD env var if None).
            simulation: If True, run in Flex simulation mode.
            api_version: Opentrons API version string.
            dry_run: If True, skip all SSH communication entirely.
        """
        self.host_alias = host_alias
        self.simulation = simulation
        self.api_version = api_version
        self.dry_run = dry_run
        self._ot: Any = None

        # Labware tracking: nickname -> location mapping
        self.labware_nicknames: dict[str, str] = {}

        # Tip tracking: pip_name -> current tip index  (mirrors OpenTronsControl)
        self.tip_racks: dict[str, list[str]] = {}
        self.tip_index: dict[str, int] = {}
        self._tip_index_file = str(
            Path(__file__).parent / "flex_tip_index.json"
        )
        self._load_tip_index()

        if dry_run:
            LOGGER.info("[Flex] Initialized in DRY-RUN mode (no SSH)")
            return

        if OpenTrons is None:
            LOGGER.warning(
                "[Flex] matterlab_opentrons not installed — falling back to dry-run"
            )
            self.dry_run = True
            return

        # Resolve password
        pwd = password or os.environ.get("OPENTRONS_PASSWORD", "")
        if not pwd:
            LOGGER.warning(
                "[Flex] No password provided and OPENTRONS_PASSWORD not set "
                "— falling back to dry-run"
            )
            self.dry_run = True
            return

        self._connect(pwd)

    # ------------------------------------------------------------------
    # Tip index persistence  (matches OpenTronsControl.tip_index.json)
    # ------------------------------------------------------------------

    def _load_tip_index(self):
        """Load persistent tip index from JSON file."""
        try:
            with open(self._tip_index_file, "r") as f:
                self.tip_index = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.tip_index = {}

    def _save_tip_index(self):
        """Persist tip index to JSON file."""
        try:
            with open(self._tip_index_file, "w") as f:
                json.dump(self.tip_index, f)
        except OSError as exc:
            LOGGER.warning(f"[Flex] Could not save tip index: {exc}")

    def reset_tip_index(self, pip_name: Optional[str] = None):
        """Reset tip counter (all pipettes or a specific one).

        Args:
            pip_name: If given, reset only this pipette's index.
        """
        if pip_name:
            self.tip_index[pip_name] = 0
        else:
            self.tip_index = {k: 0 for k in self.tip_index}
        self._save_tip_index()
        LOGGER.info(
            f"[Flex] Tip index reset: "
            f"{pip_name if pip_name else 'all pipettes'}"
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self, password: str):
        """Establish SSH connection and initialize protocol API."""
        try:
            self._ot = OpenTrons(
                host_alias=self.host_alias,
                password=password,
                simulation=self.simulation,
            )
            # The OpenTrons constructor already calls _get_protocol()
            # which sets up `protocol` on the remote side.
            LOGGER.info(
                f"[Flex] Connected to {self.host_alias} "
                f"(simulation={self.simulation}, API={self.api_version})"
            )
        except Exception as e:
            LOGGER.error(f"[Flex] SSH connection failed: {e}")
            LOGGER.warning("[Flex] Falling back to dry-run mode")
            self.dry_run = True
            self._ot = None

    def is_connected(self) -> bool:
        """Check if the Flex bridge is connected."""
        return self._ot is not None and not self.dry_run

    # ------------------------------------------------------------------
    # Raw SSH  (mirrors OpenTronsControl.invoke)
    # ------------------------------------------------------------------

    def invoke(self, code: str) -> Optional[str]:
        """Execute arbitrary Python code on the Flex robot via SSH.

        This is the low-level escape hatch that maps directly to
        ``OpenTrons.invoke(code)``.

        Args:
            code: Python code string to execute on the remote robot.

        Returns:
            Raw stdout from the remote execution, or None on error / dry-run.
        """
        if self.dry_run:
            LOGGER.debug(f"[Flex][DRY-RUN] invoke: {code}")
            return None
        try:
            return self._ot.invoke(code)
        except Exception as e:
            LOGGER.error(f"[Flex] invoke failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Setup operations
    # ------------------------------------------------------------------

    def home(self) -> bool:
        """Home all robot axes."""
        if self.dry_run:
            LOGGER.info("[Flex][DRY-RUN] Homed")
            return True
        try:
            self._ot.home()
            LOGGER.info("[Flex] Homed")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] Home failed: {e}")
            return False

    def load_labware(self, labware_spec: dict[str, Any]) -> bool:
        """
        Load labware onto the Flex deck.

        Args:
            labware_spec: Dict matching matterlab_opentrons format:
                - nickname: str — unique labware nickname
                - loadname: str — labware definition name
                - location: str — Flex slot (e.g. "C2")
                - ot_default: bool — True if standard Opentrons labware
                - config: dict — custom labware JSON definition (if ot_default=False)

        Returns:
            True if loaded successfully.
        """
        nickname = labware_spec.get("nickname", "unknown")
        location = labware_spec.get("location", "?")

        if self.dry_run:
            LOGGER.info(
                f"[Flex][DRY-RUN] Loaded labware '{nickname}' at {location}"
            )
            self.labware_nicknames[nickname] = location
            return True

        try:
            self._ot.load_labware(labware_spec)
            self.labware_nicknames[nickname] = location
            LOGGER.info(f"[Flex] Loaded labware '{nickname}' at {location}")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] Failed to load labware '{nickname}': {e}")
            return False

    def load_labware_from_json(
        self,
        nickname: str,
        json_path: str,
        location: str,
    ) -> bool:
        """
        Load custom labware from a JSON definition file.

        Args:
            nickname: Unique labware nickname.
            json_path: Path to labware JSON file.
            location: Flex deck slot (e.g. "D1").

        Returns:
            True if loaded successfully.
        """
        try:
            config = json.loads(Path(json_path).read_text())
        except Exception as e:
            LOGGER.error(f"[Flex] Cannot read labware JSON '{json_path}': {e}")
            return False

        return self.load_labware({
            "nickname": nickname,
            "loadname": config.get("parameters", {}).get("loadName", nickname),
            "location": location,
            "ot_default": False,
            "config": config,
        })

    def load_instrument(self, instrument_spec: dict[str, Any]) -> bool:
        """
        Load a pipette instrument.

        Args:
            instrument_spec: Dict matching matterlab_opentrons format:
                - nickname: str — pipette nickname (e.g. "p1000")
                - instrument_name: str — e.g. "flex_1channel_1000"
                - mount: str — "left" or "right"
                - tip_racks: list[str] — list of tip rack nicknames
                - ot_default: bool

        Returns:
            True if loaded successfully.
        """
        nickname = instrument_spec.get("nickname", "unknown")
        tip_rack_list = instrument_spec.get("tip_racks", [])

        if self.dry_run:
            LOGGER.info(f"[Flex][DRY-RUN] Loaded instrument '{nickname}'")
            # Track tip racks even in dry-run
            if tip_rack_list:
                self.tip_racks[nickname] = tip_rack_list
                if nickname not in self.tip_index:
                    self.tip_index[nickname] = 0
                    self._save_tip_index()
            return True

        try:
            self._ot.load_instrument(instrument_spec)
            # Mirror OpenTronsControl: track tip_racks per pipette
            if tip_rack_list:
                self.tip_racks[nickname] = tip_rack_list
                if nickname not in self.tip_index:
                    self.tip_index[nickname] = 0
                    self._save_tip_index()
            LOGGER.info(f"[Flex] Loaded instrument '{nickname}'")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] Failed to load instrument '{nickname}': {e}")
            return False

    def load_module(self, module_spec: dict[str, Any]) -> bool:
        """
        Load a hardware module (e.g. heater-shaker).

        Args:
            module_spec: Dict with:
                - nickname: str — e.g. "hs"
                - module_name: str — e.g. "heaterShakerModuleV1"
                - location: str — Flex slot
                - adapter: str — adapter load name

        Returns:
            True if loaded successfully.
        """
        nickname = module_spec.get("nickname", "unknown")

        if self.dry_run:
            LOGGER.info(f"[Flex][DRY-RUN] Loaded module '{nickname}'")
            return True
        try:
            self._ot.load_module(module_spec)
            LOGGER.info(f"[Flex] Loaded module '{nickname}'")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] Failed to load module '{nickname}': {e}")
            return False

    def load_trash_bin(
        self,
        nickname: str = "default_trash",
        location: str = "A3",
    ) -> bool:
        """Load the Flex trash bin (default position A3).

        Args:
            nickname: Variable name for the trash bin on the remote side.
            location: Deck slot for the trash bin.
        """
        if self.dry_run:
            LOGGER.info("[Flex][DRY-RUN] Trash bin loaded")
            return True
        try:
            self._ot.load_trash_bin(nickname=nickname, location=location)
            LOGGER.info(f"[Flex] Trash bin loaded at {location}")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] Failed to load trash bin: {e}")
            return False

    # ------------------------------------------------------------------
    # Pipetting operations
    # ------------------------------------------------------------------

    def pick_up_tip(
        self,
        pip_name: str = "p1000",
        location: Optional[str] = None,
    ) -> bool:
        """Pick up a tip from the assigned tip rack.

        Mirrors OpenTronsControl: uses persistent ``tip_index`` to track
        which tip to pick next across sessions.

        Args:
            pip_name: Pipette nickname.
            location: Optional explicit tip location (e.g. "rack_name['A1']").
                      If *None*, the next tip is selected automatically from
                      the registered tip racks using ``tip_index``.
        """
        if self.dry_run:
            # Still advance tip index in dry-run for realistic simulation
            tip_count = self.tip_index.get(pip_name, 0)
            self.tip_index[pip_name] = tip_count + 1
            self._save_tip_index()
            LOGGER.info(
                f"[Flex][DRY-RUN] Picked up tip ({pip_name}, index={tip_count})"
            )
            return True
        try:
            self._ot.pick_up_tip(pip_name=pip_name, location=location)
            LOGGER.info(f"[Flex] Picked up tip ({pip_name})")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] pick_up_tip failed: {e}")
            return False

    def return_tip(self, pip_name: str = "p1000") -> bool:
        """Return tip to original position (instead of dropping to trash).

        Args:
            pip_name: Pipette nickname.
        """
        if self.dry_run:
            LOGGER.info(f"[Flex][DRY-RUN] Returned tip ({pip_name})")
            return True
        try:
            self._ot.return_tip(pip_name=pip_name)
            LOGGER.info(f"[Flex] Returned tip ({pip_name})")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] return_tip failed: {e}")
            return False

    def drop_tip(self, pip_name: str = "p1000") -> bool:
        """Drop tip into trash."""
        if self.dry_run:
            LOGGER.info(f"[Flex][DRY-RUN] Dropped tip ({pip_name})")
            return True
        try:
            self._ot.drop_tip(pip_name=pip_name)
            LOGGER.info(f"[Flex] Dropped tip ({pip_name})")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] drop_tip failed: {e}")
            return False

    def get_location(
        self,
        labware_nickname: str,
        position: str = "A1",
        top: float = 0,
        bottom: float = 0,
        center: float = 0,
        x_offset: float = 0,
        y_offset: float = 0,
    ) -> bool:
        """
        Set the target location for the next move/aspirate/dispense.

        Mirrors ``OpenTronsControl.get_location_from_labware`` exactly:
        only one of *top*, *bottom*, *center* should be non-zero.

        Args:
            labware_nickname: Nickname of the labware.
            position: Well position (e.g. "A1").
            top: Z offset from top of well in mm (negative = deeper).
            bottom: Z offset from bottom of well in mm.
            center: If non-zero, use center of well.
            x_offset: X offset in mm (for fine centering).
            y_offset: Y offset in mm.

        Returns:
            True if successful.
        """
        if self.dry_run:
            ref = "top" if top else ("bottom" if bottom else ("center" if center else "top"))
            LOGGER.debug(
                f"[Flex][DRY-RUN] Location set: {labware_nickname}:{position} "
                f"({ref}={top or bottom or center}, x={x_offset}, y={y_offset})"
            )
            return True
        try:
            self._ot.get_location_from_labware(
                labware_nickname,
                position=position,
                top=top,
                bottom=bottom,
                center=center,
                x_offset=x_offset,
                y_offset=y_offset,
            )
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] get_location failed: {e}")
            return False

    def move_to_pip(self, pip_name: str = "p1000") -> bool:
        """Move pipette to the previously set location."""
        if self.dry_run:
            LOGGER.debug(f"[Flex][DRY-RUN] Moved {pip_name} to location")
            return True
        try:
            self._ot.move_to_pip(pip_name=pip_name)
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] move_to_pip failed: {e}")
            return False

    def prepare_aspirate(self, pip_name: str = "p1000") -> bool:
        """Prepare pipette for aspiration (pre-wet / prime).

        Args:
            pip_name: Pipette nickname.
        """
        if self.dry_run:
            LOGGER.debug(f"[Flex][DRY-RUN] Prepared aspirate ({pip_name})")
            return True
        try:
            self._ot.prepare_aspirate(pip_name=pip_name)
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] prepare_aspirate failed: {e}")
            return False

    def aspirate(
        self,
        pip_name: str = "p1000",
        volume: float = 0,
    ) -> bool:
        """
        Aspirate liquid at the current location.

        Args:
            pip_name: Pipette nickname.
            volume: Volume in µL.

        Returns:
            True if successful.
        """
        if self.dry_run:
            LOGGER.info(f"[Flex][DRY-RUN] Aspirated {volume}µL ({pip_name})")
            return True
        try:
            self._ot.aspirate(pip_name=pip_name, volume=volume)
            LOGGER.debug(f"[Flex] Aspirated {volume}µL ({pip_name})")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] aspirate failed: {e}")
            return False

    def dispense(
        self,
        pip_name: str = "p1000",
        volume: float = 0,
        push_out: Optional[float] = None,
    ) -> bool:
        """
        Dispense liquid at the current location.

        Args:
            pip_name: Pipette nickname.
            volume: Volume in µL.
            push_out: Extra plunger push (µL) to ensure full dispensing.
                      Maps to ``push_out`` in ``OpenTronsControl.dispense``.

        Returns:
            True if successful.
        """
        if self.dry_run:
            LOGGER.info(
                f"[Flex][DRY-RUN] Dispensed {volume}µL ({pip_name})"
                + (f" push_out={push_out}" if push_out else "")
            )
            return True
        try:
            self._ot.dispense(pip_name=pip_name, volume=volume, push_out=push_out)
            LOGGER.debug(f"[Flex] Dispensed {volume}µL ({pip_name})")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] dispense failed: {e}")
            return False

    def blow_out(self, pip_name: str = "p1000") -> bool:
        """Blow out remaining liquid at the current location.

        Args:
            pip_name: Pipette nickname.
        """
        if self.dry_run:
            LOGGER.info(f"[Flex][DRY-RUN] Blow-out ({pip_name})")
            return True
        try:
            self._ot.blow_out(pip_name=pip_name)
            LOGGER.debug(f"[Flex] Blow-out ({pip_name})")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] blow_out failed: {e}")
            return False

    def touch_tip(
        self,
        pip_name: str = "p1000",
        labware_nickname: str = "",
        position: str = "A1",
        radius: float = 1.0,
        v_offset: float = -1.0,
    ) -> bool:
        """Touch the tip to the sides of the well to remove droplets.

        Args:
            pip_name: Pipette nickname.
            labware_nickname: Labware where the well is.
            position: Well position.
            radius: Fraction of well radius to touch (0-1).
            v_offset: Vertical offset from top of well (mm).
        """
        if self.dry_run:
            LOGGER.debug(
                f"[Flex][DRY-RUN] touch_tip {pip_name} at "
                f"{labware_nickname}:{position}"
            )
            return True
        try:
            self._ot.touch_tip(
                pip_name=pip_name,
                labware_nickname=labware_nickname,
                position=position,
                radius=radius,
                v_offset=v_offset,
            )
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] touch_tip failed: {e}")
            return False

    # ------------------------------------------------------------------
    # High-level compound operations
    # ------------------------------------------------------------------

    def transfer(
        self,
        pip_name: str,
        volume: float,
        source_labware: str,
        source_well: str,
        dest_labware: str,
        dest_well: str,
        source_top: float = 0,
        source_bottom: float = 0,
        dest_top: float = 0,
        dest_bottom: float = 0,
        source_x_offset: float = 0,
        source_y_offset: float = 0,
        dest_x_offset: float = 0,
        dest_y_offset: float = 1,
        push_out: Optional[float] = None,
    ) -> bool:
        """
        High-level transfer: aspirate from source, dispense to destination.

        Args:
            pip_name: Pipette nickname.
            volume: Volume in µL.
            source_labware: Source labware nickname.
            source_well: Source well position.
            dest_labware: Destination labware nickname.
            dest_well: Destination well position.
            source_top: Z offset from top at source.
            source_bottom: Z offset from bottom at source.
            dest_top: Z offset from top at destination.
            dest_bottom: Z offset from bottom at destination.
            source_x_offset: X offset at source.
            source_y_offset: Y offset at source.
            dest_x_offset: X offset at destination.
            dest_y_offset: Y offset at destination.
            push_out: Extra plunger push on dispense.

        Returns:
            True if successful.
        """
        ok = self.get_location(
            source_labware, source_well,
            top=source_top, bottom=source_bottom,
            x_offset=source_x_offset, y_offset=source_y_offset,
        )
        ok = ok and self.aspirate(pip_name=pip_name, volume=volume)
        ok = ok and self.get_location(
            dest_labware, dest_well,
            top=dest_top, bottom=dest_bottom,
            x_offset=dest_x_offset, y_offset=dest_y_offset,
        )
        ok = ok and self.dispense(pip_name=pip_name, volume=volume, push_out=push_out)
        return ok

    def mix(
        self,
        pip_name: str,
        labware: str,
        well: str,
        volume: float = 300,
        cycles: int = 3,
        top: float = 0,
        bottom: float = 0,
        x_offset: float = 0,
        y_offset: float = 1,
    ) -> bool:
        """
        Mix by repeated aspirate/dispense at a single well.

        Args:
            pip_name: Pipette nickname.
            labware: Labware nickname.
            well: Well position.
            volume: Mix volume in µL.
            cycles: Number of mix cycles.
            top: Z offset from top.
            bottom: Z offset from bottom.
            x_offset: X offset.
            y_offset: Y offset.

        Returns:
            True if all cycles successful.
        """
        for i in range(cycles):
            ok = self.get_location(
                labware, well,
                top=top, bottom=bottom,
                x_offset=x_offset, y_offset=y_offset,
            )
            ok = ok and self.aspirate(pip_name=pip_name, volume=volume)
            ok = ok and self.dispense(pip_name=pip_name, volume=volume)
            if not ok:
                LOGGER.error(f"[Flex] Mix cycle {i+1}/{cycles} failed")
                return False
        return True

    # ------------------------------------------------------------------
    # Movement & speed
    # ------------------------------------------------------------------

    def set_speed(self, pip_name: str, speed: float) -> bool:
        """Set default movement speed for a pipette (mm/s).

        Args:
            pip_name: Pipette nickname.
            speed: Speed in mm/s.
        """
        if self.dry_run:
            LOGGER.debug(f"[Flex][DRY-RUN] Set speed {pip_name} → {speed}")
            return True
        try:
            self._ot.set_speed(pip_name=pip_name, speed=speed)
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] set_speed failed: {e}")
            return False

    def delay(self, seconds: float = 0, minutes: float = 0) -> bool:
        """Insert a timed delay on the robot.

        Args:
            seconds: Delay in seconds.
            minutes: Delay in minutes (additive with seconds).
        """
        if self.dry_run:
            total = seconds + minutes * 60
            LOGGER.info(f"[Flex][DRY-RUN] Delay {total}s")
            return True
        try:
            self._ot.delay(seconds=seconds, minutes=minutes)
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] delay failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Gripper operations
    # ------------------------------------------------------------------

    def move_labware_w_gripper(
        self,
        labware_nickname: str,
        new_location: str,
    ) -> bool:
        """Move labware using the Flex integrated gripper.

        Args:
            labware_nickname: Nickname of the labware to move.
            new_location: Target slot, module adapter, or "OFF_DECK".
        """
        if self.dry_run:
            LOGGER.info(
                f"[Flex][DRY-RUN] Gripper move '{labware_nickname}' → {new_location}"
            )
            # Update tracking
            if labware_nickname in self.labware_nicknames:
                self.labware_nicknames[labware_nickname] = new_location
            return True
        try:
            self._ot.move_labware_w_gripper(
                labware_nickname=labware_nickname,
                new_location=new_location,
            )
            if labware_nickname in self.labware_nicknames:
                self.labware_nicknames[labware_nickname] = new_location
            LOGGER.info(
                f"[Flex] Gripper moved '{labware_nickname}' → {new_location}"
            )
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] Gripper move failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Heater-Shaker module operations
    # ------------------------------------------------------------------

    def hs_latch_open(self, nickname: str) -> bool:
        """Open heater-shaker labware latch."""
        if self.dry_run:
            LOGGER.info(f"[Flex][DRY-RUN] HS latch opened ({nickname})")
            return True
        try:
            self._ot.hs_latch_open(nickname=nickname)
            LOGGER.info(f"[Flex] HS latch opened ({nickname})")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] hs_latch_open failed: {e}")
            return False

    def hs_latch_close(self, nickname: str) -> bool:
        """Close heater-shaker labware latch."""
        if self.dry_run:
            LOGGER.info(f"[Flex][DRY-RUN] HS latch closed ({nickname})")
            return True
        try:
            self._ot.hs_latch_close(nickname=nickname)
            LOGGER.info(f"[Flex] HS latch closed ({nickname})")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] hs_latch_close failed: {e}")
            return False

    def set_rpm(self, nickname: str, rpm: int) -> bool:
        """Set heater-shaker shake speed (200-3000 RPM, 0 to stop)."""
        if self.dry_run:
            LOGGER.info(f"[Flex][DRY-RUN] HS RPM → {rpm} ({nickname})")
            return True
        try:
            self._ot.set_rpm(nickname=nickname, rpm=rpm)
            LOGGER.info(f"[Flex] HS RPM → {rpm} ({nickname})")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] set_rpm failed: {e}")
            return False

    def set_temp(self, nickname: str, temp: float) -> bool:
        """Set heater-shaker temperature (27-95°C, 0 to deactivate)."""
        if self.dry_run:
            LOGGER.info(f"[Flex][DRY-RUN] HS temp → {temp}°C ({nickname})")
            return True
        try:
            self._ot.set_temp(nickname=nickname, temp=temp)
            LOGGER.info(f"[Flex] HS temp → {temp}°C ({nickname})")
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] set_temp failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Protocol control
    # ------------------------------------------------------------------

    def pause(self) -> bool:
        """Pause protocol execution on the robot."""
        if self.dry_run:
            LOGGER.info("[Flex][DRY-RUN] Protocol paused")
            return True
        try:
            self._ot.pause()
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] pause failed: {e}")
            return False

    def resume(self) -> bool:
        """Resume protocol execution on the robot."""
        if self.dry_run:
            LOGGER.info("[Flex][DRY-RUN] Protocol resumed")
            return True
        try:
            self._ot.resume()
            return True
        except Exception as e:
            LOGGER.error(f"[Flex] resume failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Close the SSH connection."""
        if self._ot is not None:
            try:
                self._ot.close_session()
            except Exception:
                pass  # best-effort
            LOGGER.info("[Flex] Bridge connection closed")
        self._ot = None

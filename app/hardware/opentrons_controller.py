"""
Opentrons Robot Controller
Wrapper for Opentrons API to provide simplified interface for liquid handling operations
"""
import json
import logging
from typing import Dict, Optional

# These imports are only available when the real hardware SDK is installed.
# In simulated / dry-run mode the module is never imported.
try:
    from opentrons import opentronsClient
except ImportError:
    opentronsClient = None  # type: ignore[assignment,misc]

LOGGER = logging.getLogger(__name__)


class OpentronsController:
    """
    High-level controller for Opentrons OT-2 robot
    Manages pipettes, labware, and liquid handling operations
    """

    def __init__(
        self,
        robot_ip: str,
        pipette_small: str = "p300_single",
        pipette_large: str = "p1000_single_gen2",
        move_speed: int = 200
    ):
        """
        Initialize Opentrons controller

        Args:
            robot_ip: IP address of the Opentrons robot
            pipette_small: Small pipette model name
            pipette_large: Large pipette model name
            move_speed: Default movement speed in mm/s
        """
        self.robot_ip = robot_ip
        self.pipette_small = pipette_small
        self.pipette_large = pipette_large
        self.move_speed = move_speed

        # Initialize client
        self.client = opentronsClient(strRobotIP=robot_ip)

        # Labware tracking
        self.labware = {}  # slot -> labware_id mapping

        # Tip tracking
        self.next_tip_small = 0
        self.next_tip_large = 30

        LOGGER.info(f"OpentronsController initialized for robot at {robot_ip}", extra={"device": "robot"})

    def initialize_pipettes(
        self,
        small_mount: str = 'left',
        large_mount: str = 'right'
    ):
        """
        Load pipettes onto robot mounts

        Args:
            small_mount: Mount for small pipette ('left' or 'right')
            large_mount: Mount for large pipette ('left' or 'right')
        """
        self.client.loadPipette(strPipetteName=self.pipette_small, strMount=small_mount)
        self.client.loadPipette(strPipetteName=self.pipette_large, strMount=large_mount)
        LOGGER.info(f"Loaded pipettes: {self.pipette_small} on {small_mount}, {self.pipette_large} on {large_mount}", extra={"device": "robot"})

    def load_standard_labware(
        self,
        slot: int,
        labware_name: str
    ) -> str:
        """
        Load standard Opentrons labware

        Args:
            slot: Deck slot number (1-11)
            labware_name: Standard labware name (e.g., 'opentrons_96_tiprack_1000ul')

        Returns:
            Labware ID
        """
        labware_id = self.client.loadLabware(strSlot=slot, strLabwareName=labware_name)
        self.labware[slot] = labware_id
        logging.info(f"Loaded labware '{labware_name}' in slot {slot}")
        return labware_id

    def load_custom_labware(
        self,
        slot: int,
        labware_json_path: str
    ) -> str:
        """
        Load custom labware from JSON definition

        Args:
            slot: Deck slot number (1-11)
            labware_json_path: Path to labware JSON file

        Returns:
            Labware ID
        """
        with open(labware_json_path) as f:
            labware_definition = json.load(f)

        labware_id = self.client.loadCustomLabware(dicLabware=labware_definition, strSlot=slot)
        self.labware[slot] = labware_id
        logging.info(f"Loaded custom labware from '{labware_json_path}' in slot {slot}")
        return labware_id

    def home_robot(self):
        """Home all robot axes"""
        self.client.homeRobot()
        logging.info("Robot homed")

    def set_lights(self, on: bool):
        """
        Control deck lights

        Args:
            on: True to turn lights on, False to turn off
        """
        self.client.lights(on)

    def pick_up_tip(
        self,
        labware_id: str,
        well_name: str,
        pipette_name: str,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0
    ):
        """
        Pick up a tip from specified location

        Args:
            labware_id: ID of the labware containing tips
            well_name: Well position (e.g., 'A1')
            pipette_name: Name of pipette to use
            offset_x, offset_y, offset_z: Position offsets in mm
        """
        self.client.pickUpTip(
            strLabwareName=labware_id,
            strWellName=well_name,
            strPipetteName=pipette_name,
            fltOffsetX=offset_x,
            fltOffsetY=offset_y,
            fltOffsetZ=offset_z
        )
        logging.debug(f"Picked up tip from {labware_id}:{well_name}")

    def has_tip(self, pipette_name: str) -> bool:
        """
        Check if pipette currently has a tip attached

        Args:
            pipette_name: Name of pipette to check

        Returns:
            True if pipette has a tip, False otherwise
        """
        try:
            status = self.client.pipetteHasTip(strPipetteName=pipette_name)
            return status
        except Exception as e:
            logging.warning(f"Could not check tip status for {pipette_name}: {e}")
            return False

    def drop_tip(
        self,
        pipette_name: str,
        labware_id: str = None,
        well_name: str = None,
        drop_in_trash: bool = False,
        trash: bool = False,
        offset_start: str = 'top',
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0
    ):
        """
        Drop tip at specified location or in trash (EXPLICIT trash parameter required for safety)

        Args:
            pipette_name: Name of pipette to use
            labware_id: ID of the labware for tip disposal (optional if trash=True)
            well_name: Well position (e.g., 'A1') (optional if trash=True)
            drop_in_trash: DEPRECATED - use 'trash' parameter instead
            trash: If True, drop in trash/disposal; if False, drop at labware location (REQUIRED)
            offset_start: Starting point for offset ('top', 'bottom', 'center')
            offset_x, offset_y, offset_z: Position offsets in mm
        """
        # Support both old and new parameter names
        should_drop_in_trash = trash or drop_in_trash

        if should_drop_in_trash:
            # Drop in trash/disposal
            self.client.dropTip(
                strPipetteName=pipette_name,
                boolDropInDisposal=True
            )
            logging.info(f"Dropped tip in trash")
        else:
            # Drop in specific labware location
            if not labware_id or not well_name:
                raise ValueError(
                    "drop_tip requires explicit labware_id and well_name when trash=False. "
                    "For safety, specify trash=True to drop in trash."
                )
            self.client.dropTip(
                strLabwareName=labware_id,
                strWellName=well_name,
                strPipetteName=pipette_name,
                boolDropInDisposal=False,
                strOffsetStart=offset_start,
                fltOffsetX=offset_x,
                fltOffsetY=offset_y,
                fltOffsetZ=offset_z
            )
            logging.info(f"Dropped tip at {labware_id}:{well_name}")

    def aspirate(
        self,
        volume: float,
        labware_id: str,
        well_name: str,
        pipette_name: str,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0,
        offset_start: str = 'top'
    ):
        """
        Aspirate liquid from specified location

        Args:
            volume: Volume to aspirate in µL
            labware_id: ID of the labware
            well_name: Well position (e.g., 'A1')
            pipette_name: Name of pipette to use
            offset_x, offset_y, offset_z: Position offsets in mm
            offset_start: Reference point ('top' or 'bottom')
        """
        self.client.aspirate(
            intVolume=int(volume),
            strLabwareName=labware_id,
            strWellName=well_name,
            strPipetteName=pipette_name,
            strOffsetStart=offset_start,
            fltOffsetX=offset_x,
            fltOffsetY=offset_y,
            fltOffsetZ=offset_z
        )
        logging.debug(f"Aspirated {volume}µL from {labware_id}:{well_name}")

    def dispense(
        self,
        volume: float,
        labware_id: str,
        well_name: str,
        pipette_name: str,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0,
        offset_start: str = 'top'
    ):
        """
        Dispense liquid to specified location

        Args:
            volume: Volume to dispense in µL
            labware_id: ID of the labware
            well_name: Well position (e.g., 'A1')
            pipette_name: Name of pipette to use
            offset_x, offset_y, offset_z: Position offsets in mm
            offset_start: Reference point ('top' or 'bottom')
        """
        self.client.dispense(
            intVolume=int(volume),
            strLabwareName=labware_id,
            strWellName=well_name,
            strPipetteName=pipette_name,
            strOffsetStart=offset_start,
            fltOffsetX=offset_x,
            fltOffsetY=offset_y,
            fltOffsetZ=offset_z
        )
        logging.debug(f"Dispensed {volume}µL to {labware_id}:{well_name}")

    def blowout(
        self,
        labware_id: str,
        well_name: str,
        pipette_name: str,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0,
        offset_start: str = 'top'
    ):
        """
        Blowout remaining liquid from pipette

        Args:
            labware_id: ID of the labware
            well_name: Well position (e.g., 'A1')
            pipette_name: Name of pipette to use
            offset_x, offset_y, offset_z: Position offsets in mm
            offset_start: Reference point ('top' or 'bottom')
        """
        self.client.blowout(
            strLabwareName=labware_id,
            strWellName=well_name,
            strPipetteName=pipette_name,
            strOffsetStart=offset_start,
            fltOffsetX=offset_x,
            fltOffsetY=offset_y,
            fltOffsetZ=offset_z
        )
        logging.info(f"Blowout successful.")
        logging.debug(f"Blowout at {labware_id}:{well_name}")

    def move_to_well(
        self,
        labware_id: str,
        well_name: str,
        pipette_name: str,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0,
        offset_start: str = 'top',
        speed: Optional[int] = None
    ):
        """
        Move pipette to specified well position

        Args:
            labware_id: ID of the labware
            well_name: Well position (e.g., 'A1')
            pipette_name: Name of pipette to use
            offset_x, offset_y, offset_z: Position offsets in mm
            offset_start: Reference point ('top' or 'bottom')
            speed: Movement speed in mm/s (uses default if None)
        """
        if speed is None:
            speed = self.move_speed

        self.client.moveToWell(
            strLabwareName=labware_id,
            strWellName=well_name,
            strPipetteName=pipette_name,
            strOffsetStart=offset_start,
            fltOffsetX=offset_x,
            fltOffsetY=offset_y,
            fltOffsetZ=offset_z,
            intSpeed=speed
        )

    def transfer_liquid(
        self,
        volume: float,
        source_labware: str,
        source_well: str,
        dest_labware: str,
        dest_well: str,
        pipette_name: str,
        mix_after: bool = False,
        mix_volume: float = None,
        mix_repetitions: int = 3
    ):
        """
        High-level liquid transfer operation

        Args:
            volume: Volume to transfer in µL
            source_labware: Source labware ID
            source_well: Source well name
            dest_labware: Destination labware ID
            dest_well: Destination well name
            pipette_name: Name of pipette to use
            mix_after: Whether to mix after dispensing
            mix_volume: Volume for mixing (uses transfer volume if None)
            mix_repetitions: Number of mix cycles
        """
        # Aspirate from source
        self.aspirate(
            volume=volume,
            labware_id=source_labware,
            well_name=source_well,
            pipette_name=pipette_name
        )

        # Dispense to destination
        self.dispense(
            volume=volume,
            labware_id=dest_labware,
            well_name=dest_well,
            pipette_name=pipette_name
        )

        # Mix if requested
        if mix_after:
            if mix_volume is None:
                mix_volume = volume * 0.8  # Use 80% of transfer volume

            for _ in range(mix_repetitions):
                self.aspirate(
                    volume=mix_volume,
                    labware_id=dest_labware,
                    well_name=dest_well,
                    pipette_name=pipette_name
                )
                self.dispense(
                    volume=mix_volume,
                    labware_id=dest_labware,
                    well_name=dest_well,
                    pipette_name=pipette_name
                )

        logging.info(f"Transferred {volume}µL from {source_labware}:{source_well} to {dest_labware}:{dest_well}")

    def get_labware_id(self, slot: int) -> Optional[str]:
        """
        Get labware ID for a given slot

        Args:
            slot: Deck slot number

        Returns:
            Labware ID or None if no labware in slot
        """
        return self.labware.get(slot)

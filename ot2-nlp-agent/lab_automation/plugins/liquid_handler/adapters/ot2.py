"""
OT-2 Hardware Adapter

Translates generic liquid handling operations to Opentrons OT-2 specific commands.
"""

from typing import Any, Dict, List, Optional


class OT2Adapter:
    """
    Adapter for Opentrons OT-2 liquid handling robot.

    Translates generic liquid handling operations to OT-2
    Python API calls.
    """

    name = "ot2"
    manufacturer = "Opentrons"
    model = "OT-2"

    # Standard OT-2 labware mappings
    LABWARE_ALIASES = {
        # Plates
        "96 well plate": "corning_96_wellplate_360ul_flat",
        "96孔板": "corning_96_wellplate_360ul_flat",
        "384 well plate": "corning_384_wellplate_112ul_flat",
        "384孔板": "corning_384_wellplate_112ul_flat",

        # Tip racks
        "tip rack": "opentrons_96_tiprack_300ul",
        "tiprack": "opentrons_96_tiprack_300ul",
        "枪头架": "opentrons_96_tiprack_300ul",
        "tip rack 20": "opentrons_96_tiprack_20ul",
        "tip rack 300": "opentrons_96_tiprack_300ul",
        "tip rack 1000": "opentrons_96_tiprack_1000ul",

        # Reservoirs
        "reservoir": "nest_12_reservoir_15ml",
        "储液槽": "nest_12_reservoir_15ml",

        # Tubes
        "tube rack": "opentrons_24_tuberack_nest_1.5ml_snapcap",
        "试管架": "opentrons_24_tuberack_nest_1.5ml_snapcap",
    }

    # Pipette mappings
    PIPETTE_ALIASES = {
        "p20": "p20_single_gen2",
        "p300": "p300_single_gen2",
        "p1000": "p1000_single_gen2",
        "p20_multi": "p20_multi_gen2",
        "p300_multi": "p300_multi_gen2",
        "单道": "p300_single_gen2",
        "八道": "p300_multi_gen2",
    }

    def resolve_labware(self, alias: str) -> str:
        """Resolve labware alias to OT-2 labware name."""
        return self.LABWARE_ALIASES.get(alias.lower(), alias)

    def resolve_pipette(self, alias: str) -> str:
        """Resolve pipette alias to OT-2 pipette name."""
        return self.PIPETTE_ALIASES.get(alias.lower(), alias)

    def translate_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate generic action to OT-2 specific command.

        Args:
            action: Generic action string (e.g., "liquid_handler.transfer")
            params: Action parameters

        Returns:
            OT-2 specific command dictionary
        """
        # Map generic actions to OT-2 API methods
        action_map = {
            "liquid_handler.aspirate": "aspirate",
            "liquid_handler.dispense": "dispense",
            "liquid_handler.transfer": "transfer",
            "liquid_handler.distribute": "distribute",
            "liquid_handler.consolidate": "consolidate",
            "liquid_handler.mix": "mix",
            "liquid_handler.pick_up_tip": "pick_up_tip",
            "liquid_handler.drop_tip": "drop_tip",
            "liquid_handler.touch_tip": "touch_tip",
            "liquid_handler.blow_out": "blow_out",
            "liquid_handler.air_gap": "air_gap",
            "liquid_handler.move_to": "move_to",
            "liquid_handler.home": "home",
            "liquid_handler.pause": "pause",
        }

        ot2_action = action_map.get(action, action.split(".")[-1])

        return {
            "action": f"robot.{ot2_action}",
            "params": self._translate_params(ot2_action, params),
        }

    def _translate_params(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Translate generic params to OT-2 specific params."""
        result = params.copy()

        # Rename 'location' to appropriate OT-2 param name
        if 'location' in result:
            if action in ['aspirate', 'dispense']:
                # Keep as location
                pass
            elif action == 'move_to':
                result['well'] = result.pop('location')

        return result

    def generate_python_code(
        self,
        action: str,
        params: Dict[str, Any],
        pipette_var: str = "pipette",
        labware_var: str = "plate"
    ) -> str:
        """
        Generate OT-2 Python API code for an action.

        Args:
            action: Action name
            params: Action parameters
            pipette_var: Variable name for the pipette
            labware_var: Variable name for the labware

        Returns:
            Python code string
        """
        location = params.get('location', 'A1')
        volume = params.get('volume', 100)

        code_templates = {
            "aspirate": f"{pipette_var}.aspirate({volume}, {labware_var}['{location}'])",
            "dispense": f"{pipette_var}.dispense({volume}, {labware_var}['{location}'])",
            "transfer": f"{pipette_var}.transfer({volume}, {labware_var}['{params.get('source', 'A1')}'], {labware_var}['{params.get('destination', 'B1')}'])",
            "mix": f"{pipette_var}.mix({params.get('repetitions', 3)}, {volume}, {labware_var}['{location}'])",
            "pick_up_tip": f"{pipette_var}.pick_up_tip()",
            "drop_tip": f"{pipette_var}.drop_tip()",
            "touch_tip": f"{pipette_var}.touch_tip()",
            "blow_out": f"{pipette_var}.blow_out()",
            "home": "protocol.home()",
            "pause": f"protocol.pause('{params.get('message', 'Paused')}')",
        }

        return code_templates.get(action, f"# Unknown action: {action}")

    def get_capabilities(self) -> Dict[str, Any]:
        """Get OT-2 adapter capabilities."""
        return {
            "name": self.name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "deck_slots": 11,
            "pipette_mounts": ["left", "right"],
            "supported_pipettes": list(self.PIPETTE_ALIASES.values()),
            "supported_labware": list(self.LABWARE_ALIASES.values()),
            "api_level": "2.13",
        }

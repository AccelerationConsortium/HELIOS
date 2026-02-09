---
name: ot2-robot
description: "OT-2 liquid handling robot — pipette operations, labware management, and deck movement"
version: "1.0.0"
instrument: ot2-robot
resource_id: ot2-robot
primitives:
  - name: robot.home
    error_class: BYPASS
    safety_class: INFORMATIONAL
    params: {}
    description: "Home all axes to known positions"
    contract:
      preconditions: []
      effects:
        - "set:robot_homed:true"
    timeout:
      seconds: 60
      retries: 1
  - name: robot.load_pipettes
    error_class: BYPASS
    safety_class: INFORMATIONAL
    params:
      small_mount: {type: string, optional: true, description: "Pipette on left mount (e.g. p20_single_gen2)"}
      large_mount: {type: string, optional: true, description: "Pipette on right mount (e.g. p300_single_gen2)"}
    description: "Load pipettes onto the robot mounts"
    contract:
      preconditions: []
      effects:
        - "set:pipettes_loaded:true"
    timeout:
      seconds: 30
      retries: 1
  - name: robot.set_lights
    error_class: BYPASS
    safety_class: INFORMATIONAL
    params:
      "on": {type: boolean, description: "Turn deck lights on or off"}
    description: "Control the robot deck lights"
    contract:
      preconditions: []
      effects: []
    timeout:
      seconds: 5
      retries: 0
  - name: robot.load_labware
    error_class: CRITICAL
    safety_class: CAREFUL
    params:
      slot: {type: integer, description: "Deck slot number (1-11)"}
      labware: {type: string, description: "Labware definition name (e.g. opentrons_96_tiprack_300ul)"}
      name: {type: string, description: "Human-readable label for this labware instance"}
    description: "Load a standard labware definition onto a deck slot"
    contract:
      preconditions: []
      effects:
        - "set:labware_loaded:{name}:true"
    timeout:
      seconds: 30
      retries: 2
  - name: robot.load_custom_labware
    error_class: CRITICAL
    safety_class: CAREFUL
    params:
      slot: {type: integer, description: "Deck slot number (1-11)"}
      labware_file: {type: string, description: "Path to custom labware JSON definition"}
      name: {type: string, description: "Human-readable label"}
    description: "Load a custom labware JSON definition onto a deck slot"
    contract:
      preconditions: []
      effects:
        - "set:labware_loaded:{name}:true"
    timeout:
      seconds: 30
      retries: 2
  - name: robot.move_to_well
    error_class: CRITICAL
    safety_class: CAREFUL
    params:
      labware: {type: string, description: "Labware name (as loaded)"}
      well: {type: string, description: "Well identifier (e.g. A1, B3)"}
      pipette: {type: string, description: "Which pipette to move (small or large)"}
      offset_x: {type: number, optional: true, default: 0, description: "X offset in mm"}
      offset_y: {type: number, optional: true, default: 0, description: "Y offset in mm"}
      offset_z: {type: number, optional: true, default: 0, description: "Z offset in mm"}
      speed: {type: number, optional: true, description: "Movement speed override"}
    description: "Move pipette to a specific well position"
    contract:
      preconditions:
        - "labware_loaded:{labware}"
        - "pipettes_loaded"
      effects: []
    timeout:
      seconds: 30
      retries: 1
  - name: robot.pick_up_tip
    error_class: CRITICAL
    safety_class: CAREFUL
    params:
      labware: {type: string, description: "Tip rack labware name"}
      well: {type: string, description: "Tip position (e.g. A1)"}
      pipette: {type: string, description: "Which pipette (small or large)"}
      offset_x: {type: number, optional: true, default: 0}
      offset_y: {type: number, optional: true, default: 0}
      offset_z: {type: number, optional: true, default: 0}
    description: "Pick up a pipette tip from a tip rack"
    contract:
      preconditions:
        - "labware_loaded:{labware}"
        - "tip_off:{pipette}"
        - "pipettes_loaded"
      effects:
        - "set:tip_state:{pipette}:on"
    timeout:
      seconds: 30
      retries: 2
  - name: robot.drop_tip
    error_class: CRITICAL
    safety_class: CAREFUL
    params:
      pipette: {type: string, description: "Which pipette (small or large)"}
      labware: {type: string, optional: true, description: "Target labware (if not trash)"}
      well: {type: string, optional: true, description: "Target well"}
      drop_in_trash: {type: boolean, optional: true, default: true, description: "Drop in trash instead of labware"}
      offset_x: {type: number, optional: true, default: 0}
      offset_y: {type: number, optional: true, default: 0}
      offset_z: {type: number, optional: true, default: 0}
    description: "Drop the current pipette tip"
    contract:
      preconditions:
        - "tip_on:{pipette}"
      effects:
        - "set:tip_state:{pipette}:off"
        - "set:pipette_volume:{pipette}:0"
    timeout:
      seconds: 30
      retries: 1
  - name: robot.aspirate
    error_class: CRITICAL
    safety_class: HAZARDOUS
    params:
      labware: {type: string, description: "Source labware name"}
      well: {type: string, description: "Source well"}
      pipette: {type: string, description: "Which pipette (small or large)"}
      volume: {type: number, description: "Volume in uL (must not exceed pipette capacity)"}
      offset_x: {type: number, optional: true, default: 0}
      offset_y: {type: number, optional: true, default: 0}
      offset_z: {type: number, optional: true, default: 0}
    description: "Aspirate liquid from a well"
    contract:
      preconditions:
        - "labware_loaded:{labware}"
        - "tip_on:{pipette}"
        - "pipettes_loaded"
      effects:
        - "increase:pipette_volume:{pipette}:{volume}"
    timeout:
      seconds: 30
      retries: 0
  - name: robot.dispense
    error_class: CRITICAL
    safety_class: HAZARDOUS
    params:
      labware: {type: string, description: "Destination labware name"}
      well: {type: string, description: "Destination well"}
      pipette: {type: string, description: "Which pipette (small or large)"}
      volume: {type: number, description: "Volume in uL"}
      offset_x: {type: number, optional: true, default: 0}
      offset_y: {type: number, optional: true, default: 0}
      offset_z: {type: number, optional: true, default: 0}
    description: "Dispense liquid into a well"
    contract:
      preconditions:
        - "labware_loaded:{labware}"
        - "tip_on:{pipette}"
        - "pipettes_loaded"
      effects:
        - "decrease:pipette_volume:{pipette}:{volume}"
    timeout:
      seconds: 30
      retries: 0
  - name: robot.blowout
    error_class: BYPASS
    safety_class: REVERSIBLE
    params:
      labware: {type: string, description: "Target labware name"}
      well: {type: string, description: "Target well"}
      pipette: {type: string, description: "Which pipette (small or large)"}
      offset_x: {type: number, optional: true, default: 0}
      offset_y: {type: number, optional: true, default: 0}
      offset_z: {type: number, optional: true, default: 0}
    description: "Blow out remaining liquid from the pipette tip"
    contract:
      preconditions:
        - "labware_loaded:{labware}"
        - "tip_on:{pipette}"
      effects:
        - "set:pipette_volume:{pipette}:0"
    timeout:
      seconds: 15
      retries: 1
---

# OT-2 Liquid Handling Robot

The OT-2 is the primary liquid handling instrument. It moves pipettes across a
deck with 11 slots, each holding labware (tip racks, well plates, reservoirs).

## When to Use

Use robot primitives whenever the protocol requires:
- Transferring liquid between containers
- Preparing samples by mixing reagents
- Serial dilutions or multi-well plate operations
- Any physical pipetting operation

## Workflow Pattern

A typical liquid transfer follows this sequence:

```
robot.home
  -> robot.load_pipettes
  -> robot.load_labware (source, destination, tips)
  -> robot.pick_up_tip
  -> robot.aspirate (from source)
  -> robot.dispense (to destination)
  -> robot.blowout (optional, clear residual)
  -> robot.drop_tip
```

## Safety Constraints

- **Volume limits**: p20 max 20 uL, p300 max 300 uL. The safety gate rejects
  volumes exceeding pipette capacity at compile time.
- **Tip tracking**: The dispatcher tracks which tips have been used. Never
  aspirate without a tip — the system will raise CRITICAL.
- **Labware registration**: All labware must be loaded before any pipetting
  operation references it. The dispatcher resolves labware names to deck
  positions internally.
- **Offset limits**: Z offsets below -5 mm risk crashing into labware.
  Use conservative offsets unless calibration data confirms clearance.

## Resource Locking

All robot primitives require the `ot2-robot` resource lock. Only one step
can use the robot at a time. Parallel steps using other instruments
(PLC, relay, squidstat) can run concurrently.

## Error Behavior

| Primitive | Error Class | Safety Class | On Failure |
|-----------|------------|-------------|------------|
| robot.home | BYPASS | INFORMATIONAL | Log and continue |
| robot.load_pipettes | BYPASS | INFORMATIONAL | Log and continue |
| robot.set_lights | BYPASS | INFORMATIONAL | Log and continue |
| robot.load_labware | CRITICAL | CAREFUL | Retry (2x), then abort |
| robot.load_custom_labware | CRITICAL | CAREFUL | Retry (2x), then abort |
| robot.move_to_well | CRITICAL | CAREFUL | Retry (1x), then abort |
| robot.pick_up_tip | CRITICAL | CAREFUL | Retry (2x), then abort |
| robot.drop_tip | CRITICAL | CAREFUL | Retry (1x), then abort |
| robot.aspirate | CRITICAL | HAZARDOUS | Abort immediately |
| robot.dispense | CRITICAL | HAZARDOUS | Abort immediately |
| robot.blowout | BYPASS | REVERSIBLE | Log and continue |

*11 primitives. 3 INFORMATIONAL, 1 REVERSIBLE, 5 CAREFUL, 2 HAZARDOUS.*

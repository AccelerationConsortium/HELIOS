"""Hardware controllers copied from refactored_battery project.

These modules wrap real lab instruments:
- OpentronsController: OT-2 liquid handling robot
- PLCController: Modbus TCP pump/stirrer control
- RelayController: SainSmart USB 16-channel relay
- ActionDispatcher: Maps action strings to hardware handler methods
- RunContext / PhaseResult: Execution state tracking
"""

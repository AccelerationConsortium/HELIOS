"""Hardware controllers for NeverOT self-driving lab.

These modules wrap real lab instruments:
- OpentronsController: OT-2 liquid handling robot (HTTP API)
- FlexBridge: Opentrons Flex robot (SSH via matterlab_opentrons)
- PhSensorController: Colorimetric pH strip measurement (pHAnalyzer)
- PLCController: Modbus TCP pump/stirrer control
- RelayController: SainSmart USB 16-channel relay
- ActionDispatcher: Maps action strings to hardware handler methods
- RunContext / PhaseResult: Execution state tracking
"""

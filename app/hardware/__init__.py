"""Hardware controllers for the self-driving lab.

These modules wrap real lab instruments:
- OpentronsController: OT-2 liquid handling robot (HTTP API)
- FlexBridge: Opentrons Flex robot (SSH via matterlab_opentrons)
- PhSensorController: Colorimetric pH strip measurement (pHAnalyzer)
- RunContext / PhaseResult: Execution state tracking

Lab-internal device drivers (pump/relay control, potentiostat dispatch)
are kept out of the public distribution; the execution layer falls back
to the simulated adapter when they are absent.
"""

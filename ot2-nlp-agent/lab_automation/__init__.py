"""
Lab Automation Agent - Plugin-based Multi-Instrument Control

A modular framework for creating lab automation workflows using
natural language instructions across multiple instrument types.

Supported device types:
- liquid_handler: Liquid handling robots (OT-2, Hamilton, Tecan, etc.)
- potentiostat: Electrochemistry instruments (SquidStat, Gamry, BioLogic)
- pump_controller: Pump systems (PLC, peristaltic, syringe)
- camera: Imaging systems (USB, SSH, IP cameras)

Example:
    from lab_automation import LabAutomationAgent
    from lab_automation.plugins import (
        LiquidHandlerPlugin,
        PotentiostatPlugin,
        PumpControllerPlugin,
    )

    # Create agent and register plugins
    agent = LabAutomationAgent()
    agent.register_plugin(LiquidHandlerPlugin())
    agent.register_plugin(PotentiostatPlugin())
    agent.register_plugin(PumpControllerPlugin())

    # Create workflow from natural language
    workflow = agent.create_workflow(
        name="Battery Test Workflow",
        instructions=[
            "transfer 100ul from A1 to B1",
            "run EIS from 10kHz to 0.1Hz",
            "wait 30 seconds",
            "pump 5ml water",
        ]
    )

    # Export to JSON
    workflow.save_json("battery_test.json")
"""

from .core import (
    LabAutomationAgent,
    PluginBase,
    OperationDef,
    ParserBase,
    Workflow,
    Phase,
    Step,
    StepParams,
)

__version__ = "2.0.0"
__all__ = [
    'LabAutomationAgent',
    'PluginBase',
    'OperationDef',
    'ParserBase',
    'Workflow',
    'Phase',
    'Step',
    'StepParams',
]

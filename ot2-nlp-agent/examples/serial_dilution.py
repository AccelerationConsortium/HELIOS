"""
Serial Dilution Example - OT-2 NLP Agent

Demonstrates creating a serial dilution protocol using natural language.
"""

import sys
sys.path.insert(0, '..')

from ot2_agent import OT2Agent


def main():
    # Create the agent
    agent = OT2Agent()

    # Create protocol
    protocol = agent.create_protocol(
        name="Serial Dilution Protocol",
        description="Perform 2x serial dilution across row A"
    )

    # Add labware
    agent.add_labware(protocol, "96孔板", slot=1, name="plate")
    agent.add_labware(protocol, "tip rack", slot=2, name="tips")
    agent.add_labware(protocol, "reservoir", slot=3, name="diluent")

    # Add pipette
    agent.add_pipette(protocol, "p300", mount="left", tip_rack="tips")

    # Method 1: Using convenience method
    print("Creating serial dilution using convenience method...")
    wells = [f"A{i}" for i in range(1, 9)]  # A1-A8
    agent.quick_serial_dilution(
        protocol,
        wells=wells,
        initial_volume=200,  # µL
        dilution_factor=2.0,
        mix_reps=3
    )

    # Validate
    print("\n" + "="*50)
    print("VALIDATION RESULT")
    print("="*50)
    result = agent.validate(protocol)
    print(result)

    # Preview
    print("\n" + "="*50)
    print("PROTOCOL PREVIEW")
    print("="*50)
    print(agent.preview(protocol))

    # =========================================================
    # Alternative: Using natural language
    # =========================================================
    print("\n" + "="*50)
    print("NATURAL LANGUAGE VERSION")
    print("="*50)

    protocol2 = agent.create_protocol(
        name="Serial Dilution (NL)",
        description="Same protocol created with natural language"
    )

    agent.add_labware(protocol2, "96孔板", slot=1, name="plate")
    agent.add_labware(protocol2, "tip rack", slot=2, name="tips")
    agent.add_pipette(protocol2, "p300", mount="left", tip_rack="tips")

    # Using natural language with multi-step
    agent.add_instruction(protocol2, """
    第一步：取枪头
    第二步：在A1孔混匀3次，100微升
    第三步：从A1孔吸取100微升，分配到A2孔
    第四步：在A2孔混匀3次
    第五步：从A2孔吸取100微升，分配到A3孔
    第六步：在A3孔混匀3次
    第七步：丢弃枪头
    """)

    print(agent.preview(protocol2))


if __name__ == "__main__":
    main()

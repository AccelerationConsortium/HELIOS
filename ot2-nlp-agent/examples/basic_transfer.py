"""
Basic Transfer Example - OT-2 NLP Agent

Demonstrates creating a simple liquid transfer protocol using natural language.
"""

import sys
sys.path.insert(0, '..')

from ot2_agent import OT2Agent


def main():
    # Create the agent
    agent = OT2Agent()

    # Create a new protocol
    protocol = agent.create_protocol(
        name="Basic Sample Transfer",
        description="Transfer samples from column 1 to column 2"
    )

    # Add labware
    agent.add_labware(protocol, "96孔板", slot=1, name="plate")
    agent.add_labware(protocol, "300ul tips", slot=2, name="tips")
    agent.add_labware(protocol, "reservoir", slot=3, name="reservoir")

    # Add pipette
    agent.add_pipette(protocol, "p300", mount="left", tip_rack="tips")

    # Add instructions using natural language (Chinese)
    print("Adding instructions...")
    agent.add_instructions(protocol, [
        "取枪头",
        "从reservoir的A1孔吸取200微升",
        "分配到plate的A1孔",
        "混匀3次，每次100微升",
        "丢弃枪头",
    ])

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

    # Generate code
    print("\n" + "="*50)
    print("GENERATED CODE")
    print("="*50)
    code = agent.generate(protocol)
    print(code)

    # Save to file
    agent.save(protocol, "basic_transfer_protocol.py")
    print("\n✅ Protocol saved to basic_transfer_protocol.py")


if __name__ == "__main__":
    main()

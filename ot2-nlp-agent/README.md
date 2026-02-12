# OT-2 NLP Agent

Natural language interface for Opentrons OT-2 liquid handling robot. Allows non-programmers to design workflows using natural language in English or Chinese.

## Features

- **Natural Language Parsing**: Understands instructions like "从A1孔吸取100微升" or "aspirate 100ul from A1"
- **Multi-language Support**: English and Chinese
- **Protocol Validation**: Catches errors before execution
- **Code Generation**: Generates valid Opentrons Python protocols
- **Reusable**: Can be integrated into any lab automation system

## Quick Start

```python
from ot2_agent import OT2Agent

# Create agent
agent = OT2Agent()

# Create protocol
protocol = agent.create_protocol("Sample Transfer")

# Add labware
agent.add_labware(protocol, "96孔板", slot=1, name="plate")
agent.add_labware(protocol, "tip rack", slot=2, name="tips")

# Add pipette
agent.add_pipette(protocol, "p300", mount="left", tip_rack="tips")

# Add instructions using natural language
agent.add_instructions(protocol, [
    "取枪头",
    "从A1孔吸取100微升",
    "分配到B1孔",
    "丢弃枪头"
])

# Validate
result = agent.validate(protocol)
print(result)

# Generate Python code
code = agent.generate(protocol)
print(code)

# Save to file
agent.save(protocol, "my_protocol.py")
```

## Supported Operations

| Operation | English Keywords | Chinese Keywords |
|-----------|-----------------|------------------|
| Aspirate | aspirate, suck, draw | 吸取, 吸, 抽取 |
| Dispense | dispense, release, add | 分配, 释放, 注入 |
| Transfer | transfer, move liquid | 转移, 移液 |
| Pick up tip | pick up tip, get tip | 取枪头, 拿枪头 |
| Drop tip | drop tip, eject tip | 丢弃枪头, 弃枪头 |
| Mix | mix, stir, blend | 混合, 混匀, 搅拌 |
| Move to | move to, go to | 移动到, 前往 |
| Pause | pause, stop | 暂停, 停止 |
| Wait | wait, delay | 等待, 延迟 |

## Labware Aliases

| Alias | Opentrons Name |
|-------|----------------|
| 96孔板 / 96-well plate | corning_96_wellplate_360ul_flat |
| 24孔板 / 24-well plate | corning_24_wellplate_3.4ml_flat |
| 枪头盒 / tip rack | opentrons_96_tiprack_300ul |
| 试剂槽 / reservoir | nest_12_reservoir_15ml |

## Custom Labware (3D Printed)

Support for custom/3D-printed labware:

### Built-in Templates

```python
# List available templates
print(agent.list_custom_labware_templates())
# ['battery_holder_1x4', 'battery_holder_2x4', 'battery_holder_3x4',
#  'battery_holder_4x6', 'vial_holder_3x4', 'vial_holder_4x6',
#  'custom_reservoir_4', 'custom_reservoir_8', 'electrode_holder_1x8']

# Use a template
agent.add_custom_labware(protocol, "battery_holder_4x6", slot=1, name="batteries")
```

### Create Custom Labware

```python
# Create a custom coin cell holder
coin_holder = agent.create_custom_labware(
    name="coin_cell_holder_3x5",
    rows=3,
    columns=5,
    well_depth=5,          # mm
    well_diameter=20,      # mm (for circular wells)
    well_volume=500,       # µL
    row_spacing=25,        # mm between rows
    column_spacing=25,     # mm between columns
    description="Holder for 15 CR2032 coin cells"
)
agent.add_custom_labware(protocol, coin_holder, slot=1, name="coin_cells")
```

### Load from Opentrons JSON

```python
from ot2_agent import get_labware_manager

manager = get_labware_manager()

# Load from Opentrons Labware Creator JSON
labware = manager.load_from_file("my_custom_labware.json")
agent.add_custom_labware(protocol, labware, slot=1)

# Save to JSON for sharing
labware.save_json("exported_labware.json")
```

The generated protocol automatically embeds custom labware definitions and uses `load_labware_from_definition()` to load them.

## Architecture

```
ot2-nlp-agent/
├── ot2_agent/
│   ├── __init__.py      # Package exports
│   ├── agent.py         # Main OT2Agent class
│   ├── parser.py        # Natural language parser
│   ├── operations.py    # Operation definitions
│   ├── protocol.py      # Protocol generator
│   └── validator.py     # Protocol validator
├── examples/            # Example protocols
├── tests/              # Unit tests
└── docs/               # Documentation
```

## Multi-step Instructions

The agent can parse complex multi-step instructions:

```python
# Using step markers
agent.add_instruction(protocol, """
第一步：从slot1的A1孔吸取50微升
第二步：分配到slot2的B1-B4孔
第三步：丢弃枪头
""")

# Using sequence words
agent.add_instruction(protocol,
    "从A1吸取100微升，然后分配到B1，最后混匀3次"
)
```

## Validation

The validator checks for common issues:

```python
result = agent.validate(protocol)

if not result.is_valid:
    print(f"Found {result.errors_count} errors")
    for issue in result.issues:
        print(issue)
```

Validation checks include:
- Missing labware or pipette configuration
- Invalid slot numbers (1-11)
- Volume limits
- Well coordinate format (A1-H12)
- Tip management (pick up before use, drop when done)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ot2-nlp-agent.git
cd ot2-nlp-agent

# Install in development mode
pip install -e .

# Or just copy the ot2_agent folder to your project
```

## Integration

### With Gradio UI

```python
import gradio as gr
from ot2_agent import OT2Agent

agent = OT2Agent()

def process_instructions(instructions: str):
    protocol = agent.create_protocol("User Protocol")
    # Add default labware...
    agent.add_instruction(protocol, instructions)

    validation = agent.validate(protocol)
    preview = agent.preview(protocol)
    code = agent.generate(protocol)

    return str(validation), preview, code

iface = gr.Interface(
    fn=process_instructions,
    inputs=gr.Textbox(label="Enter instructions"),
    outputs=[
        gr.Textbox(label="Validation"),
        gr.Textbox(label="Preview"),
        gr.Code(label="Generated Code", language="python")
    ]
)
iface.launch()
```

### With FastAPI

```python
from fastapi import FastAPI
from ot2_agent import OT2Agent

app = FastAPI()
agent = OT2Agent()

@app.post("/parse")
def parse_instruction(instruction: str):
    intent = agent.parse(instruction)
    return {
        "operation": intent.operation_type.value if intent.operation_type else None,
        "params": intent.params,
        "confidence": intent.confidence
    }
```

## License

MIT License

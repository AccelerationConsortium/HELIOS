"""
Lab Automation Agent - Gradio Frontend
Battery Lab Dashboard Design System

Supports multiple instrument types through plugin architecture:
- Liquid Handler (OT-2, Hamilton, etc.)
- Potentiostat (SquidStat, Gamry, etc.)
- Pump Controller (PLC, peristaltic, etc.)
- Camera (USB, SSH, IP)
"""

import gradio as gr
import json

# Try to import new plugin system, fall back to legacy
try:
    from lab_automation import LabAutomationAgent
    from lab_automation.plugins import (
        LiquidHandlerPlugin,
        PotentiostatPlugin,
        PumpControllerPlugin,
        CameraPlugin,
    )

    # Create agent with all plugins
    agent = LabAutomationAgent()
    agent.register_plugin(LiquidHandlerPlugin())
    agent.register_plugin(PotentiostatPlugin())
    agent.register_plugin(PumpControllerPlugin())
    agent.register_plugin(CameraPlugin())

    USE_PLUGIN_SYSTEM = True
except ImportError:
    # Fall back to legacy OT2Agent
    from ot2_agent import OT2Agent
    agent = OT2Agent()
    USE_PLUGIN_SYSTEM = False

# Import Planner/Compiler types for experiment planning
try:
    from ot2_agent.planner import Planner, PlannerOutput, WorkflowDraft, ConfirmedWorkflow
    from ot2_agent.compiler import Compiler, CompilerOutput
    from ot2_agent.planner.domain_knowledge import OERDomainKnowledge
    from ot2_agent.compiler.device_mapper import DeviceRegistry
    HAS_PLANNER = True
except ImportError:
    HAS_PLANNER = False

# Legacy mapper for reference tab
try:
    from ot2_agent.operations import OperationMapper
    mapper = OperationMapper()
except ImportError:
    mapper = None

# ============ Battery Lab Dashboard CSS ============

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

:root {
    /* Primary Colors */
    --color-navy: #213C51;
    --color-navy-light: #2d4d66;
    --color-navy-dark: #192d3d;
    --color-steel: #6594B1;
    --color-steel-light: #7da8c4;
    --color-steel-dim: rgba(101, 148, 177, 0.12);
    --color-pink: #DDAED3;
    --color-pink-dim: rgba(221, 174, 211, 0.2);
    --color-light: #EEEEEE;

    /* Backgrounds */
    --bg-primary: #f8f9fa;
    --bg-secondary: #ffffff;
    --bg-tertiary: #f0f2f5;
    --bg-card: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);

    /* Text */
    --text-primary: #213C51;
    --text-secondary: #3d5a6f;
    --text-tertiary: #6594B1;

    /* Status */
    --color-success: #22c55e;
    --color-success-dim: rgba(34, 197, 94, 0.12);
    --color-warning: #f59e0b;
    --color-error: #ef4444;

    /* Borders & Shadows */
    --border-subtle: rgba(33, 60, 81, 0.08);
    --border-default: rgba(33, 60, 81, 0.12);
    --shadow-sm: 0 1px 2px rgba(33, 60, 81, 0.06);
    --shadow-md: 0 4px 12px rgba(33, 60, 81, 0.08);
    --shadow-lg: 0 8px 32px rgba(33, 60, 81, 0.12);

    /* Spacing */
    --space-2: 8px;
    --space-3: 12px;
    --space-4: 16px;
    --space-5: 20px;
    --space-6: 24px;
    --space-8: 32px;

    /* Radius */
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
}

/* Base styles */
body, .gradio-container {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Gradient overlay */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: radial-gradient(ellipse at top right, rgba(101, 148, 177, 0.05) 0%, transparent 50%),
                radial-gradient(ellipse at bottom left, rgba(221, 174, 211, 0.03) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* Main header */
.main-header {
    text-align: center;
    padding: var(--space-8) var(--space-4);
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    margin-bottom: var(--space-6);
    box-shadow: var(--shadow-md);
}

.main-header h1 {
    color: var(--color-navy) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    margin-bottom: var(--space-2) !important;
}

.main-header h3 {
    color: var(--text-tertiary) !important;
    font-weight: 400 !important;
    font-size: 1rem !important;
}

/* Tabs */
.tabs {
    background: transparent !important;
}

.tab-nav {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: var(--space-2) !important;
    gap: var(--space-2) !important;
    box-shadow: var(--shadow-sm) !important;
}

.tab-nav button {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: var(--space-3) var(--space-4) !important;
    transition: all 0.2s ease !important;
}

.tab-nav button:hover {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
}

.tab-nav button.selected {
    background: var(--color-steel-dim) !important;
    color: var(--color-navy) !important;
    font-weight: 600 !important;
}

/* Tab content */
.tabitem {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: var(--space-6) !important;
    margin-top: var(--space-4) !important;
    box-shadow: var(--shadow-md) !important;
}

/* Input fields */
.input-container, .textbox {
    font-family: 'JetBrains Mono', monospace !important;
}

input, textarea, .textbox textarea {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    transition: all 0.15s ease !important;
}

input:focus, textarea:focus {
    border-color: var(--color-steel) !important;
    box-shadow: 0 0 0 3px var(--color-steel-dim) !important;
    outline: none !important;
}

/* Labels */
label, .label-wrap span {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8125rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* Buttons */
button.primary {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    background: var(--color-steel) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: var(--space-3) var(--space-5) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s ease !important;
}

button.primary:hover {
    background: var(--color-navy) !important;
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md) !important;
}

button.secondary {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bg-secondary) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
}

button.secondary:hover {
    background: var(--bg-tertiary) !important;
    border-color: var(--color-steel) !important;
}

/* Code blocks */
.code-wrap, pre, code {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--color-navy-dark) !important;
    color: #e2e8f0 !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-subtle) !important;
}

/* Examples */
.examples-row {
    gap: var(--space-2) !important;
}

.examples-row button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8125rem !important;
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    transition: all 0.15s ease !important;
}

.examples-row button:hover {
    background: var(--color-steel-dim) !important;
    border-color: var(--color-steel) !important;
    color: var(--color-navy) !important;
}

/* Markdown */
.markdown-text, .prose {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-primary) !important;
}

.markdown-text h2, .prose h2 {
    color: var(--color-navy) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--color-pink-dim) !important;
    padding-bottom: var(--space-2) !important;
}

.markdown-text h3, .prose h3 {
    color: var(--color-steel) !important;
    font-weight: 600 !important;
}

.markdown-text table {
    border-collapse: collapse !important;
    width: 100% !important;
}

.markdown-text th {
    background: var(--bg-tertiary) !important;
    color: var(--text-tertiary) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: var(--space-3) var(--space-4) !important;
    border-bottom: 1px solid var(--border-default) !important;
}

.markdown-text td {
    padding: var(--space-3) var(--space-4) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
}

.markdown-text tr:hover td {
    background: var(--bg-tertiary) !important;
}

/* Status indicators */
.status-success {
    color: var(--color-success) !important;
    background: var(--color-success-dim) !important;
    padding: var(--space-2) var(--space-3) !important;
    border-radius: var(--radius-sm) !important;
}

.status-error {
    color: var(--color-error) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-tertiary);
}

::-webkit-scrollbar-thumb {
    background: var(--color-steel-dim);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--color-steel);
}

/* Row and column spacing */
.row, .column {
    gap: var(--space-4) !important;
}

/* Step list styling */
.step-list {
    background: var(--bg-tertiary) !important;
    padding: var(--space-4) !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-subtle) !important;
    max-height: 300px;
    overflow-y: auto;
}

.step-list table {
    font-size: 0.8rem !important;
}

/* Number input styling */
input[type="number"] {
    width: 80px !important;
}
"""

# ============ Core Functions ============

def parse_instruction_with_data(instruction: str):
    """Parse a single instruction and return both display text and parsed data"""
    if not instruction.strip():
        return "Please enter an instruction", None, instruction

    if USE_PLUGIN_SYSTEM:
        plugin_name, parsed = agent.parse_instruction(instruction)

        if not plugin_name:
            return "UNRECOGNIZED INSTRUCTION\nNo plugin could handle this instruction.", None, instruction

        # Format parameters for display
        params_lines = []
        for k, v in parsed.get('params', {}).items():
            params_lines.append(f"  • {k}: {v}")
        params_str = "\n".join(params_lines) if params_lines else "  (no parameters)"

        result = f"""DEVICE TYPE: {parsed.get('action', '').split('.')[0] if parsed.get('action') else 'unknown'}
OPERATION: {parsed.get('operation', 'unknown')}
ACTION: {parsed.get('action', 'unknown')}
LANGUAGE: {"Chinese" if parsed.get('language') == "zh" else "English"}
CONFIDENCE: {parsed.get('confidence', 0):.0%}

EXTRACTED PARAMETERS:
{params_str}"""

        # Return display text, parsed data for workflow, and original instruction
        return result, parsed, instruction
    else:
        intent = agent.parse(instruction)
        op_type = intent.operation_type.value if intent.operation_type else "Unrecognized"

        params_lines = []
        for k, v in intent.params.items():
            params_lines.append(f"  • {k}: {v}")
        params_str = "\n".join(params_lines) if params_lines else "  (no parameters)"

        result = f"""OPERATION TYPE: {op_type}
LANGUAGE: {"Chinese" if intent.language == "zh" else "English"}
CONFIDENCE: {intent.confidence:.0%}

EXTRACTED PARAMETERS:
{params_str}"""

        # Create parsed data compatible with workflow
        parsed = {
            "operation": op_type,
            "action": f"liquid_handler.{op_type.lower()}",
            "params": intent.params,
            "confidence": intent.confidence,
        }
        return result, parsed, instruction


def parse_instruction(instruction: str):
    """Parse a single instruction"""
    if not instruction.strip():
        return "Please enter an instruction"

    if USE_PLUGIN_SYSTEM:
        # New plugin system
        plugin_name, parsed = agent.parse_instruction(instruction)

        if not plugin_name:
            return "UNRECOGNIZED INSTRUCTION\nNo plugin could handle this instruction."

        # Format parameters
        params_lines = []
        for k, v in parsed.get('params', {}).items():
            params_lines.append(f"  • {k}: {v}")
        params_str = "\n".join(params_lines) if params_lines else "  (no parameters)"

        result = f"""DEVICE TYPE: {parsed.get('action', '').split('.')[0] if parsed.get('action') else 'unknown'}
OPERATION: {parsed.get('operation', 'unknown')}
ACTION: {parsed.get('action', 'unknown')}
LANGUAGE: {"Chinese" if parsed.get('language') == "zh" else "English"}
CONFIDENCE: {parsed.get('confidence', 0):.0%}

EXTRACTED PARAMETERS:
{params_str}"""
    else:
        # Legacy OT2Agent
        intent = agent.parse(instruction)

        op_type = intent.operation_type.value if intent.operation_type else "Unrecognized"

        params_lines = []
        for k, v in intent.params.items():
            params_lines.append(f"  • {k}: {v}")
        params_str = "\n".join(params_lines) if params_lines else "  (no parameters)"

        result = f"""OPERATION TYPE: {op_type}
LANGUAGE: {"Chinese" if intent.language == "zh" else "English"}
CONFIDENCE: {intent.confidence:.0%}

EXTRACTED PARAMETERS:
{params_str}"""

    return result


# ============ Workflow Builder Functions ============

def format_step_list(steps):
    """Format step list for display"""
    if not steps:
        return "No steps added yet. Parse instructions in the first tab and click 'Add to Workflow'."

    lines = ["| # | Device | Operation | Action | Parameters |",
             "|---|--------|-----------|--------|------------|"]

    for i, step in enumerate(steps):
        device = step.get('action', '').split('.')[0] if step.get('action') else 'unknown'
        operation = step.get('operation', 'unknown')
        action = step.get('action', 'unknown')
        params = step.get('params', {})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "-"
        if len(params_str) > 30:
            params_str = params_str[:27] + "..."
        lines.append(f"| {i+1} | {device} | {operation} | {action} | {params_str} |")

    return "\n".join(lines)


def add_step_to_workflow(parsed_data, original_instruction, workflow_steps):
    """Add parsed step to workflow"""
    if workflow_steps is None:
        workflow_steps = []

    if parsed_data is None:
        return workflow_steps, format_step_list(workflow_steps), "⚠️ No valid step to add. Parse an instruction first."

    # Add step with original instruction for reference
    step = parsed_data.copy()
    step['original_instruction'] = original_instruction
    workflow_steps.append(step)

    return workflow_steps, format_step_list(workflow_steps), f"✓ Added step {len(workflow_steps)}: {step.get('operation', 'unknown')}"


def move_step_up(step_index, workflow_steps):
    """Move a step up in the list"""
    if workflow_steps is None or len(workflow_steps) == 0:
        return workflow_steps, format_step_list(workflow_steps), "No steps to move"

    try:
        idx = int(step_index) - 1  # Convert to 0-based
        if idx <= 0:
            return workflow_steps, format_step_list(workflow_steps), "Step is already at the top"
        if idx >= len(workflow_steps):
            return workflow_steps, format_step_list(workflow_steps), "Invalid step number"

        workflow_steps[idx], workflow_steps[idx-1] = workflow_steps[idx-1], workflow_steps[idx]
        return workflow_steps, format_step_list(workflow_steps), f"✓ Moved step {idx+1} up"
    except (ValueError, TypeError):
        return workflow_steps, format_step_list(workflow_steps), "Please enter a valid step number"


def move_step_down(step_index, workflow_steps):
    """Move a step down in the list"""
    if workflow_steps is None or len(workflow_steps) == 0:
        return workflow_steps, format_step_list(workflow_steps), "No steps to move"

    try:
        idx = int(step_index) - 1  # Convert to 0-based
        if idx < 0:
            return workflow_steps, format_step_list(workflow_steps), "Invalid step number"
        if idx >= len(workflow_steps) - 1:
            return workflow_steps, format_step_list(workflow_steps), "Step is already at the bottom"

        workflow_steps[idx], workflow_steps[idx+1] = workflow_steps[idx+1], workflow_steps[idx]
        return workflow_steps, format_step_list(workflow_steps), f"✓ Moved step {idx+1} down"
    except (ValueError, TypeError):
        return workflow_steps, format_step_list(workflow_steps), "Please enter a valid step number"


def delete_step(step_index, workflow_steps):
    """Delete a step from the list"""
    if workflow_steps is None or len(workflow_steps) == 0:
        return workflow_steps, format_step_list(workflow_steps), "No steps to delete"

    try:
        idx = int(step_index) - 1  # Convert to 0-based
        if idx < 0 or idx >= len(workflow_steps):
            return workflow_steps, format_step_list(workflow_steps), "Invalid step number"

        removed = workflow_steps.pop(idx)
        return workflow_steps, format_step_list(workflow_steps), f"✓ Deleted step: {removed.get('operation', 'unknown')}"
    except (ValueError, TypeError):
        return workflow_steps, format_step_list(workflow_steps), "Please enter a valid step number"


def clear_all_steps(workflow_steps):
    """Clear all steps"""
    return [], format_step_list([]), "✓ Cleared all steps"


def generate_from_steps(name, description, workflow_steps):
    """Generate workflow from step list"""
    if not name.strip():
        return "Please enter a workflow name", "", "", ""

    if not workflow_steps or len(workflow_steps) == 0:
        return "No steps to generate. Add steps from the Parse tab first.", "", "", ""

    if USE_PLUGIN_SYSTEM:
        # Build workflow from steps
        workflow = agent.create_workflow(
            name=name,
            instructions=[],  # Empty - we'll add steps directly
            description=description
        )

        # Clear auto-generated steps and add our curated steps
        if workflow.phases:
            workflow.phases[0].steps = []

        from lab_automation.core.workflow import Step

        used_devices = set()
        for i, step_data in enumerate(workflow_steps):
            device_type = step_data.get('action', '').split('.')[0] if step_data.get('action') else 'unknown'
            step = Step(
                step_id=f"execution_{i+1:03d}",
                device_type=device_type,
                action=step_data.get('action', 'unknown'),
                params=step_data.get('params', {}),
                description=step_data.get('original_instruction', ''),
            )
            workflow.phases[0].add_step(step)
            used_devices.add(device_type)

        # Build preview
        preview_lines = [f"Workflow: {name}", f"Description: {description}", f"Total Steps: {len(workflow_steps)}", ""]
        preview_lines.append("Steps:")
        for i, step in enumerate(workflow.phases[0].steps):
            preview_lines.append(f"  {i+1}. [{step.device_type}] {step.action}")
            if step.params:
                for k, v in step.params.items():
                    preview_lines.append(f"      {k}: {v}")
        preview = "\n".join(preview_lines)

        # Generate JSON
        workflow_dict = workflow.to_dict()
        workflow_json = json.dumps(workflow_dict, indent=2)

        # Generate Python code
        code_lines = ['"""', f'Protocol: {name}', f'{description}', '"""', '']
        has_liquid_handler = 'liquid_handler' in used_devices

        if has_liquid_handler:
            code_lines.append("from opentrons import protocol_api")
            code_lines.append("")
            code_lines.append("metadata = {")
            code_lines.append(f"    'protocolName': '{name}',")
            code_lines.append(f"    'description': '{description}',")
            code_lines.append("    'apiLevel': '2.13'")
            code_lines.append("}")
            code_lines.append("")
            code_lines.append("def run(protocol: protocol_api.ProtocolContext):")
            code_lines.append("    # Add your labware and pipette setup here")
            code_lines.append("    # Execute steps from the workflow")
            code_lines.append("    pass")
        else:
            code_lines.append("# This workflow contains non-liquid-handler operations")
            code_lines.append("# Use the workflow JSON for execution on your platform")

        code = "\n".join(code_lines)

        status = f"✓ Generated workflow with {len(workflow_steps)} steps"
        return status, preview, code, workflow_json
    else:
        # Legacy mode - just return the steps as text
        preview = f"Workflow: {name}\n\nSteps:\n"
        for i, step in enumerate(workflow_steps):
            preview += f"  {i+1}. {step.get('original_instruction', step.get('operation', 'unknown'))}\n"

        return "Generated (legacy mode)", preview, "# Legacy mode - code generation not available", "{}"


def create_protocol(
    name: str,
    description: str,
    labware_config: str,
    pipette_config: str,
    instructions: str
):
    """Create complete protocol/workflow"""
    if not name.strip():
        return "Please enter a workflow name", "", "", "", ""

    if USE_PLUGIN_SYSTEM:
        # New plugin system - create unified workflow
        instruction_list = [
            line.strip()
            for line in instructions.strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        # Parse device configs from labware/pipette inputs
        device_configs = []

        # Parse labware as liquid_handler config
        for line in labware_config.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                device_configs.append({
                    "device_type": "liquid_handler",
                    "config_type": "labware",
                    "labware_type": parts[0],
                    "slot": int(parts[1]),
                    "name": parts[2] if len(parts) > 2 else None
                })

        # Parse pipette config
        for line in pipette_config.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                device_configs.append({
                    "device_type": "liquid_handler",
                    "config_type": "pipette",
                    "pipette_type": parts[0],
                    "mount": parts[1],
                    "tip_rack": parts[2] if len(parts) > 2 else None
                })

        # Create workflow using plugin system
        workflow = agent.create_workflow(
            name=name,
            instructions=instruction_list,
            description=description
        )

        # Build preview
        preview_lines = [f"Workflow: {name}", f"Description: {description}", ""]
        preview_lines.append("Parsed Steps:")
        for i, phase in enumerate(workflow.phases):
            for j, step in enumerate(phase.steps):
                preview_lines.append(f"  {j+1}. [{step.device_type}] {step.action}")
                if step.params:
                    for k, v in step.params.items():
                        preview_lines.append(f"      {k}: {v}")
        preview = "\n".join(preview_lines)

        # Generate workflow JSON
        import json
        workflow_dict = workflow.to_dict()
        workflow_json = json.dumps(workflow_dict, indent=2)

        # Generate Python code (if liquid_handler operations present)
        code_lines = ['"""', f'Protocol: {name}', f'{description}', '"""', '']
        has_liquid_handler = any(
            step.device_type == "liquid_handler"
            for phase in workflow.phases
            for step in phase.steps
        )

        if has_liquid_handler:
            code_lines.append("from opentrons import protocol_api")
            code_lines.append("")
            code_lines.append("metadata = {")
            code_lines.append(f"    'protocolName': '{name}',")
            code_lines.append(f"    'description': '{description}',")
            code_lines.append("    'apiLevel': '2.13'")
            code_lines.append("}")
            code_lines.append("")
            code_lines.append("def run(protocol: protocol_api.ProtocolContext):")
            code_lines.append("    # Add your labware and pipette setup here")
            code_lines.append("    # Then execute the steps from the workflow JSON")
            code_lines.append("    pass")
        else:
            code_lines.append("# This workflow contains non-liquid-handler operations")
            code_lines.append("# Use the workflow JSON for execution on your platform")

        code = "\n".join(code_lines)

        # Validation (basic for now)
        validation_lines = ["Validation Results:", ""]
        validation_lines.append(f"✓ Parsed {len(instruction_list)} instructions")
        step_count = sum(len(phase.steps) for phase in workflow.phases)
        validation_lines.append(f"✓ Generated {step_count} workflow steps")

        device_types = set(
            step.device_type
            for phase in workflow.phases
            for step in phase.steps
        )
        validation_lines.append(f"✓ Device types: {', '.join(device_types) if device_types else 'none'}")
        validation_str = "\n".join(validation_lines)

        status = "VALID — Workflow ready for export"
        return status, validation_str, preview, code, workflow_json

    else:
        # Legacy OT2Agent
        protocol = agent.create_protocol(name=name, description=description)

        # Parse labware configuration
        for line in labware_config.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                lw_type = parts[0].strip()
                slot = int(parts[1].strip())
                lw_name = parts[2].strip() if len(parts) > 2 else None
                agent.add_labware(protocol, lw_type, slot, lw_name)

        # Parse pipette configuration
        for line in pipette_config.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                pip_type = parts[0].strip()
                mount = parts[1].strip()
                tip_rack = parts[2].strip() if len(parts) > 2 else None
                agent.add_pipette(protocol, pip_type, mount, tip_rack)

        # Parse instructions
        for line in instructions.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                agent.add_instruction(protocol, line)

        # Validate
        validation = agent.validate(protocol)
        validation_str = str(validation)

        # Preview
        preview = agent.preview(protocol)

        # Generate code
        code = agent.generate(protocol)

        # Generate workflow JSON
        workflow_json = agent.to_json(protocol)

        # Status
        status = "VALID — Protocol ready for execution" if validation.is_valid else "INVALID — Please check errors"

        return status, validation_str, preview, code, workflow_json


def get_examples():
    """Get instruction examples"""
    if USE_PLUGIN_SYSTEM:
        examples = """
## Instruction Examples

### Liquid Handler (OT-2, Hamilton, etc.)
- pick up tip
- aspirate 100ul from A1
- dispense 50ul to B2
- mix 3 times with 100ul
- transfer 100ul from A1 to B1

### Potentiostat (SquidStat, Gamry, etc.)
- run OCV for 60 seconds
- run EIS from 100000Hz to 0.1Hz with 10mV amplitude
- run cyclic voltammetry from -0.5V to 0.5V at 50mV/s
- run chronopotentiometry at 1mA for 300 seconds

### Pump Controller (PLC, Peristaltic, etc.)
- dispense 10ml at 5ml/min
- set flow rate to 2.5 ml/min
- prime the pump for 30 seconds
- turn pump on

### Camera (USB, SSH, IP)
- start video stream
- capture image as sample_001
- start recording for 60 seconds
- stop recording

### Multi-Step Instructions
```
Step 1: pick up tip
Step 2: aspirate 100ul from A1
Step 3: dispense to B1
Step 4: drop tip
```
"""
    else:
        examples = """
## Instruction Examples

### Basic Commands
- pick up tip
- aspirate 100ul from A1
- dispense 50ul to B2
- mix 3 times with 100ul
- drop tip
- wait 30 seconds
- pause

### Multi-Step Instructions
```
Step 1: pick up tip
Step 2: aspirate 100ul from A1
Step 3: dispense to B1
Step 4: drop tip
```

### Transfer Operations
- transfer 100ul from A1 to B1
- distribute 50ul from A1 to B1-B4
- consolidate from A1-A4 to B1
"""
    return examples


def get_labware_info():
    """Get labware/device information"""
    if USE_PLUGIN_SYSTEM:
        lines = ["## Supported Device Types\n"]
        lines.append("| Device Type | Description | Adapters |")
        lines.append("|-------------|-------------|----------|")
        lines.append("| liquid_handler | Liquid handling robots | OT-2, Hamilton, etc. |")
        lines.append("| potentiostat | Electrochemical instruments | SquidStat, Gamry, etc. |")
        lines.append("| pump_controller | Pump systems | PLC, peristaltic, syringe |")
        lines.append("| camera | Imaging systems | USB, SSH, IP cameras |")
        return "\n".join(lines)
    else:
        lines = ["## Supported Labware Aliases\n"]
        lines.append("| Alias | Opentrons Name |")
        lines.append("|-------|----------------|")
        if mapper:
            for alias, name in mapper.LABWARE_ALIASES.items():
                lines.append(f"| {alias} | `{name}` |")
        return "\n".join(lines)


def get_pipette_info():
    """Get pipette/adapter information"""
    if USE_PLUGIN_SYSTEM:
        lines = ["## JSON Workflow Format\n"]
        lines.append("The generated JSON follows a unified schema:\n")
        lines.append("```json")
        lines.append("{")
        lines.append('  "workflow_name": "...",')
        lines.append('  "version": "1.0",')
        lines.append('  "phases": [')
        lines.append('    {')
        lines.append('      "phase_name": "...",')
        lines.append('      "steps": [')
        lines.append('        {')
        lines.append('          "step_id": "...",')
        lines.append('          "device_type": "liquid_handler",')
        lines.append('          "action": "liquid_handler.aspirate",')
        lines.append('          "params": {...}')
        lines.append('        }')
        lines.append('      ]')
        lines.append('    }')
        lines.append('  ]')
        lines.append("}")
        lines.append("```")
        return "\n".join(lines)
    else:
        lines = ["## Supported Pipette Aliases\n"]
        lines.append("| Alias | Opentrons Name |")
        lines.append("|-------|----------------|")
        if mapper:
            for alias, name in mapper.PIPETTE_ALIASES.items():
                lines.append(f"| {alias} | `{name}` |")
        return "\n".join(lines)


# ============ Plan Experiment Functions ============

def plan_experiment(intent: str, conditions_json: str):
    """Plan an experiment from natural language intent"""
    if not HAS_PLANNER:
        return "Planner module not available. Please install ot2_agent with planner support.", [], ""

    if not intent.strip():
        return "Please enter your experiment intent.", [], ""

    try:
        # Parse conditions JSON if provided
        context = {}
        if conditions_json.strip():
            try:
                context = json.loads(conditions_json)
            except json.JSONDecodeError as e:
                return f"Invalid JSON in conditions: {e}", [], ""

        # Initialize planner
        planner = Planner(OERDomainKnowledge())

        # Generate workflow candidates
        output = planner.plan(intent, context)

        # Format result
        result_lines = [
            f"## Intent Analysis",
            f"**Goal**: {output.intent.goal}",
            f"**Domain**: {output.intent.domain}",
            f"**Language**: {'Chinese' if output.intent.language == 'zh' else 'English'}",
            f"**Confidence**: {output.intent.confidence:.0%}",
            "",
            f"**Target Metrics**: {', '.join(output.intent.target_metrics) if output.intent.target_metrics else 'None specified'}",
            "",
            f"## Generated {len(output.candidates)} Candidate Workflow(s)",
            ""
        ]

        # Format each candidate
        candidates_data = []
        for i, candidate in enumerate(output.candidates):
            is_recommended = (i == output.recommended_idx)
            rec_marker = " ⭐ (Recommended)" if is_recommended else ""

            result_lines.append(f"### Candidate {i+1}: {candidate.name}{rec_marker}")
            result_lines.append(f"*{candidate.description}*")
            result_lines.append(f"**Confidence**: {candidate.confidence:.0%}")
            result_lines.append("")

            # List unit operations
            result_lines.append("**Steps:**")
            for j, uo in enumerate(candidate.unit_operations):
                result_lines.append(f"  {j+1}. {uo.name} ({uo.uo_type.value})")
            result_lines.append("")

            # List assumptions
            if candidate.assumptions:
                result_lines.append("**Assumptions:**")
                for assumption in candidate.assumptions:
                    result_lines.append(f"  - {assumption}")
                result_lines.append("")

            # Collect missing info
            missing_info = []
            for uo in candidate.unit_operations:
                for param_name, info in uo.placeholders.items():
                    missing_info.append({
                        "parameter": param_name,
                        "question": info.question,
                        "question_zh": info.question_zh,
                        "default": info.default,
                        "required": info.required,
                        "uo_name": uo.name
                    })

            if missing_info:
                result_lines.append("**Missing Parameters:**")
                for info in missing_info:
                    req = "required" if info["required"] else "optional"
                    default = f" (default: {info['default']})" if info["default"] is not None else ""
                    result_lines.append(f"  - `{info['parameter']}`: {info['question']}{default} [{req}]")
                result_lines.append("")

            candidates_data.append({
                "index": i,
                "name": candidate.name,
                "description": candidate.description,
                "confidence": candidate.confidence,
                "is_recommended": is_recommended,
                "missing_info": missing_info,
                "workflow_draft": candidate
            })

        result_text = "\n".join(result_lines)

        # Create dropdown choices
        choices = [f"{c['index']+1}. {c['name']}" + (" ⭐" if c['is_recommended'] else "") for c in candidates_data]

        return result_text, candidates_data, choices

    except Exception as e:
        import traceback
        return f"Error during planning: {e}\n\n{traceback.format_exc()}", [], ""


def get_missing_params_form(candidates_data, selected_candidate_str):
    """Get the missing parameters form for a selected candidate"""
    if not candidates_data or not selected_candidate_str:
        return "Select a workflow candidate first.", "{}"

    try:
        # Parse selection index
        idx = int(selected_candidate_str.split(".")[0]) - 1
        if idx < 0 or idx >= len(candidates_data):
            return "Invalid selection.", "{}"

        candidate = candidates_data[idx]
        missing_info = candidate.get("missing_info", [])

        if not missing_info:
            # No missing parameters
            form_text = f"### {candidate['name']}\n\nNo parameters need to be filled in. You can compile directly."
            default_params = {}
        else:
            form_lines = [
                f"### {candidate['name']}",
                "",
                "Fill in the following parameters:",
                ""
            ]

            default_params = {}
            for info in missing_info:
                req = "**required**" if info["required"] else "optional"
                default_str = f"(default: `{info['default']}`)" if info["default"] is not None else ""
                form_lines.append(f"- **{info['parameter']}** ({info['uo_name']}): {info['question']} {default_str} [{req}]")
                form_lines.append(f"  - 中文: {info['question_zh']}")

                # Pre-fill with default if available
                if info["default"] is not None:
                    default_params[info["parameter"]] = info["default"]

            form_text = "\n".join(form_lines)

        return form_text, json.dumps(default_params, indent=2, ensure_ascii=False)

    except Exception as e:
        return f"Error: {e}", "{}"


def compile_workflow(candidates_data, selected_candidate_str, params_json: str):
    """Compile the selected workflow with filled parameters"""
    if not HAS_PLANNER:
        return "Compiler module not available.", "", "", ""

    if not candidates_data or not selected_candidate_str:
        return "Please select a workflow candidate first.", "", "", ""

    try:
        # Parse selection index
        idx = int(selected_candidate_str.split(".")[0]) - 1
        if idx < 0 or idx >= len(candidates_data):
            return "Invalid selection.", "", "", ""

        candidate_data = candidates_data[idx]
        workflow_draft = candidate_data.get("workflow_draft")

        if not workflow_draft:
            return "Workflow draft not found.", "", "", ""

        # Parse parameters
        params = {}
        if params_json.strip():
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError as e:
                return f"Invalid JSON in parameters: {e}", "", "", ""

        # Create confirmed workflow
        confirmed = ConfirmedWorkflow(
            draft=workflow_draft,
            filled_parameters=params
        )

        # Initialize compiler
        compiler = Compiler(DeviceRegistry())

        # Compile
        output = compiler.compile(confirmed)

        # Format validation result
        validation_lines = ["## Validation Result", ""]
        validation_lines.append(f"**Status**: {'✅ Valid' if output.validation_result.is_valid else '❌ Invalid'}")
        validation_lines.append(f"**Errors**: {output.validation_result.error_count}")
        validation_lines.append(f"**Warnings**: {output.validation_result.warning_count}")
        validation_lines.append("")

        if output.validation_result.issues:
            validation_lines.append("### Issues:")
            for issue in output.validation_result.issues:
                icon = "❌" if issue.severity == "ERROR" else "⚠️"
                validation_lines.append(f"- {icon} [{issue.severity}] {issue.message}")
            validation_lines.append("")

        if output.validation_result.checkpoints:
            validation_lines.append("### Human Checkpoints:")
            for cp in output.validation_result.checkpoints:
                validation_lines.append(f"- **Step {cp.step_index}** ({cp.step_name}): {cp.message}")
                if cp.message_zh:
                    validation_lines.append(f"  - 中文: {cp.message_zh}")
            validation_lines.append("")

        status = "✅ Workflow compiled successfully!" if output.validation_result.is_valid else "⚠️ Compiled with warnings/errors"

        return status, "\n".join(validation_lines), output.python_code, output.workflow_json

    except Exception as e:
        import traceback
        return f"Error during compilation: {e}", traceback.format_exc(), "", ""


def get_operations_info():
    """Get operation keyword information"""
    if USE_PLUGIN_SYSTEM:
        lines = ["## Operations by Device Type\n"]

        lines.append("### Liquid Handler")
        lines.append("| Operation | Keywords |")
        lines.append("|-----------|----------|")
        lines.append("| aspirate | aspirate, suck, draw, uptake |")
        lines.append("| dispense | dispense, release, add, eject |")
        lines.append("| transfer | transfer, move liquid |")
        lines.append("| mix | mix, stir, blend |")
        lines.append("| pick_up_tip | pick up tip, get tip |")
        lines.append("| drop_tip | drop tip, discard tip |")
        lines.append("")

        lines.append("### Potentiostat")
        lines.append("| Operation | Keywords |")
        lines.append("|-----------|----------|")
        lines.append("| ocv | open circuit, OCV, rest potential |")
        lines.append("| eis | EIS, impedance, nyquist |")
        lines.append("| cv | cyclic voltammetry, CV, sweep |")
        lines.append("| cp | chronopotentiometry, CP, galvanostatic |")
        lines.append("| ca | chronoamperometry, CA, potentiostatic |")
        lines.append("")

        lines.append("### Pump Controller")
        lines.append("| Operation | Keywords |")
        lines.append("|-----------|----------|")
        lines.append("| dispense | dispense, pump out, deliver |")
        lines.append("| aspirate | aspirate, pump in, withdraw |")
        lines.append("| set_flow_rate | set flow, flow rate |")
        lines.append("| prime | prime, flush, purge |")
        lines.append("")

        lines.append("### Camera")
        lines.append("| Operation | Keywords |")
        lines.append("|-----------|----------|")
        lines.append("| start_stream | start stream, start video |")
        lines.append("| capture_image | capture, take photo, snapshot |")
        lines.append("| start_recording | start recording, record video |")
        lines.append("| stop_recording | stop recording, end recording |")

        return "\n".join(lines)
    else:
        lines = ["## Supported Operations\n"]
        if mapper:
            for op_type, keywords in mapper.KEYWORDS.items():
                en_kw = ", ".join(keywords["en"][:4])
                lines.append(f"### {op_type.value}")
                lines.append(f"Keywords: {en_kw}\n")
        return "\n".join(lines)


# ============ Build Interface ============

TITLE = "Lab Automation Agent" if USE_PLUGIN_SYSTEM else "OT-2 NLP Agent"
SUBTITLE = "Multi-Instrument Lab Automation Platform" if USE_PLUGIN_SYSTEM else "Natural Language Control for Opentrons OT-2"

with gr.Blocks(title=TITLE) as demo:

    # Header
    if USE_PLUGIN_SYSTEM:
        gr.Markdown(f"""
        # Lab Automation Agent
        ### Multi-Instrument Lab Automation Platform

        Control liquid handlers, potentiostats, pumps, and cameras using natural language.
        """, elem_classes="main-header")
    else:
        gr.Markdown("""
        # OT-2 NLP Agent
        ### Natural Language Control for Opentrons OT-2 Liquid Handler

        Create lab protocols using plain English — no programming required.
        """, elem_classes="main-header")

    # State for workflow steps and parsed data
    workflow_steps = gr.State([])
    last_parsed_data = gr.State(None)
    last_instruction = gr.State("")

    with gr.Tabs():

        # ========== Tab 1: Instruction Parser ==========
        with gr.TabItem("Parse Instruction", id=1):
            gr.Markdown("### Test natural language instruction parsing")

            with gr.Row():
                with gr.Column(scale=1):
                    instruction_input = gr.Textbox(
                        label="INSTRUCTION",
                        placeholder="e.g., aspirate 100ul from A1" if not USE_PLUGIN_SYSTEM else "e.g., aspirate 100ul from A1, run OCV for 60s, capture image",
                        lines=2
                    )
                    parse_btn = gr.Button("Parse Instruction", variant="primary")

                    if USE_PLUGIN_SYSTEM:
                        gr.Examples(
                            examples=[
                                # Liquid Handler
                                ["aspirate 100ul from A1"],
                                ["dispense 50ul to B1"],
                                ["transfer 100ul from A1 to B1"],
                                ["mix 3 times with 100ul"],
                                ["pick up tip"],
                                ["drop tip"],
                                # Potentiostat
                                ["run OCV for 60 seconds"],
                                ["run EIS from 100000Hz to 0.1Hz"],
                                ["run cyclic voltammetry from -0.5V to 0.5V at 50mV/s"],
                                # Pump
                                ["set flow rate to 5 ml/min"],
                                ["dispense 10ml"],
                                # Camera
                                ["capture image as sample_001"],
                                ["start video stream"],
                            ],
                            inputs=instruction_input,
                            label="EXAMPLES (ALL DEVICE TYPES)"
                        )
                    else:
                        gr.Examples(
                            examples=[
                                ["aspirate 100ul from A1"],
                                ["dispense 50ul to B1-B4"],
                                ["mix 3 times with 100ul"],
                                ["pick up tip"],
                                ["aspirate 200ul from C3"],
                                ["transfer 100ul from A1 to B1"],
                                ["wait 30 seconds"],
                            ],
                            inputs=instruction_input,
                            label="EXAMPLES"
                        )

                with gr.Column(scale=1):
                    parse_result = gr.Textbox(
                        label="PARSE RESULT",
                        lines=10,
                        interactive=False
                    )
                    with gr.Row():
                        add_to_workflow_btn = gr.Button("➕ Add to Workflow", variant="primary")
                    add_status = gr.Textbox(label="", lines=1, interactive=False, show_label=False)

            # Parse button - updates display and stores parsed data
            parse_btn.click(
                fn=parse_instruction_with_data,
                inputs=[instruction_input],
                outputs=[parse_result, last_parsed_data, last_instruction]
            )
            instruction_input.submit(
                fn=parse_instruction_with_data,
                inputs=[instruction_input],
                outputs=[parse_result, last_parsed_data, last_instruction]
            )

        # ========== Tab 2: Plan Experiment (Planner/Compiler) ==========
        with gr.TabItem("Plan Experiment", id=2):
            gr.Markdown("""### Design experiments using natural language

Describe your experiment goal and the system will generate candidate workflows.
Supports OER (Oxygen Evolution Reaction) electrochemistry experiments.
            """)

            # State for planning
            candidates_state = gr.State([])

            with gr.Row():
                # Left column: Input
                with gr.Column(scale=1):
                    intent_input = gr.Textbox(
                        label="EXPERIMENT INTENT / 实验意图",
                        placeholder="e.g., 我想做OER测量，用的是NiFe催化剂\ne.g., I want to perform OER measurement with NiFe catalyst",
                        lines=3
                    )

                    conditions_input = gr.Textbox(
                        label="KNOWN CONDITIONS (JSON, optional) / 已知条件",
                        placeholder='{"catalyst": "NiFe", "electrolyte": "1M KOH"}',
                        lines=3
                    )

                    plan_btn = gr.Button("🔬 Generate Workflow Candidates", variant="primary")

                    gr.Examples(
                        examples=[
                            ["我想做OER测量，用的是NiFe催化剂", '{"catalyst": "NiFe"}'],
                            ["I want to perform OER stability test", '{"duration_hours": 2}'],
                            ["测量OER过电位和Tafel斜率", '{"electrode_area_cm2": 0.196}'],
                            ["OER electrochemical characterization with EIS", '{}'],
                        ],
                        inputs=[intent_input, conditions_input],
                        label="EXAMPLES / 示例"
                    )

                # Right column: Planning result
                with gr.Column(scale=1):
                    plan_result = gr.Markdown(
                        value="Enter your experiment intent and click 'Generate Workflow Candidates'.",
                        label="PLANNING RESULT"
                    )

            gr.Markdown("---")
            gr.Markdown("### Step 2: Select and Configure Workflow")

            with gr.Row():
                with gr.Column(scale=1):
                    candidate_dropdown = gr.Dropdown(
                        label="SELECT WORKFLOW CANDIDATE / 选择候选方案",
                        choices=[],
                        interactive=True
                    )

                    params_form = gr.Markdown(
                        value="Select a workflow candidate to see required parameters.",
                        label="PARAMETERS TO FILL"
                    )

                    params_input = gr.Textbox(
                        label="FILL PARAMETERS (JSON) / 填写参数",
                        placeholder='{"electrode_area_cm2": 0.196, "target_current_density_mA_cm2": 10}',
                        lines=6
                    )

                    compile_btn = gr.Button("⚡ Compile Workflow", variant="primary")

                with gr.Column(scale=1):
                    compile_status = gr.Textbox(label="COMPILATION STATUS", lines=1, interactive=False)
                    validation_result = gr.Markdown(value="", label="VALIDATION RESULT")

            with gr.Row():
                with gr.Column():
                    compiled_code = gr.Code(
                        label="GENERATED PYTHON CODE",
                        language="python",
                        lines=20
                    )
                with gr.Column():
                    compiled_json = gr.Code(
                        label="WORKFLOW JSON",
                        language="json",
                        lines=20
                    )

            # Wire up plan button
            plan_btn.click(
                fn=plan_experiment,
                inputs=[intent_input, conditions_input],
                outputs=[plan_result, candidates_state, candidate_dropdown]
            )

            # Wire up candidate selection
            candidate_dropdown.change(
                fn=get_missing_params_form,
                inputs=[candidates_state, candidate_dropdown],
                outputs=[params_form, params_input]
            )

            # Wire up compile button
            compile_btn.click(
                fn=compile_workflow,
                inputs=[candidates_state, candidate_dropdown, params_input],
                outputs=[compile_status, validation_result, compiled_code, compiled_json]
            )

        # ========== Tab 3: Workflow Builder ==========
        with gr.TabItem("Build Workflow" if USE_PLUGIN_SYSTEM else "Generate Protocol", id=3):
            if USE_PLUGIN_SYSTEM:
                gr.Markdown("### Build workflow by adding and arranging steps")
            else:
                gr.Markdown("### Create a complete OT-2 experiment protocol")

            with gr.Row():
                # Left column: Workflow settings and step list
                with gr.Column(scale=1):
                    protocol_name = gr.Textbox(
                        label="WORKFLOW NAME",
                        placeholder="e.g., Zinc Deposition Experiment",
                        value="My Experiment"
                    )
                    protocol_desc = gr.Textbox(
                        label="DESCRIPTION",
                        placeholder="Describe the purpose of this workflow",
                        value="Multi-instrument workflow"
                    )

                    gr.Markdown("#### Workflow Steps")
                    step_list_display = gr.Markdown(
                        value="No steps added yet. Parse instructions in the first tab and click 'Add to Workflow'.",
                        elem_classes="step-list"
                    )

                    # Step management controls
                    gr.Markdown("#### Step Controls")
                    with gr.Row():
                        step_index_input = gr.Number(
                            label="STEP #",
                            value=1,
                            minimum=1,
                            precision=0,
                            scale=1
                        )
                        move_up_btn = gr.Button("⬆️ Up", scale=1)
                        move_down_btn = gr.Button("⬇️ Down", scale=1)
                        delete_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)

                    with gr.Row():
                        clear_btn = gr.Button("Clear All Steps", variant="secondary")

                    step_status = gr.Textbox(label="", lines=1, interactive=False, show_label=False)

                    gr.Markdown("---")
                    generate_btn = gr.Button("🚀 Generate Workflow", variant="primary", size="lg")

                # Right column: Output
                with gr.Column(scale=1):
                    status_output = gr.Textbox(label="STATUS", lines=1, interactive=False)
                    preview_output = gr.Textbox(label="WORKFLOW PREVIEW", lines=12, interactive=False)

            with gr.Row():
                with gr.Column():
                    code_output = gr.Code(
                        label="GENERATED PYTHON CODE",
                        language="python",
                        lines=20
                    )
                with gr.Column():
                    json_output = gr.Code(
                        label="WORKFLOW JSON (for cross-platform import)",
                        language="json",
                        lines=20
                    )

            # Wire up the add button from Tab 1
            add_to_workflow_btn.click(
                fn=add_step_to_workflow,
                inputs=[last_parsed_data, last_instruction, workflow_steps],
                outputs=[workflow_steps, step_list_display, add_status]
            )

            # Step management buttons
            move_up_btn.click(
                fn=move_step_up,
                inputs=[step_index_input, workflow_steps],
                outputs=[workflow_steps, step_list_display, step_status]
            )
            move_down_btn.click(
                fn=move_step_down,
                inputs=[step_index_input, workflow_steps],
                outputs=[workflow_steps, step_list_display, step_status]
            )
            delete_btn.click(
                fn=delete_step,
                inputs=[step_index_input, workflow_steps],
                outputs=[workflow_steps, step_list_display, step_status]
            )
            clear_btn.click(
                fn=clear_all_steps,
                inputs=[workflow_steps],
                outputs=[workflow_steps, step_list_display, step_status]
            )

            # Generate workflow
            generate_btn.click(
                fn=generate_from_steps,
                inputs=[protocol_name, protocol_desc, workflow_steps],
                outputs=[status_output, preview_output, code_output, json_output]
            )

        # ========== Tab 4: Reference Manual ==========
        with gr.TabItem("Reference", id=4):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(get_examples())
                with gr.Column():
                    gr.Markdown(get_operations_info())

            with gr.Row():
                with gr.Column():
                    gr.Markdown(get_labware_info())
                with gr.Column():
                    gr.Markdown(get_pipette_info())

        # ========== Tab 5: About ==========
        with gr.TabItem("About", id=5):
            if USE_PLUGIN_SYSTEM:
                gr.Markdown("""
            ## About Lab Automation Agent

            A plugin-based natural language processing platform for multi-instrument lab automation.

            ### Key Features

            1. **Multi-Instrument Support** — Controls liquid handlers, potentiostats, pumps, and cameras
            2. **Plugin Architecture** — Easily extend with new instrument types and adapters
            3. **Natural Language Parsing** — Understands experiment instructions in English and Chinese
            4. **Unified JSON Export** — Cross-platform workflow format for integration
            5. **Hardware Abstraction** — Swap between different brands (OT-2 ↔ Hamilton, SquidStat ↔ Gamry)

            ### Architecture

            ```
            User Input → Plugin Router → Device Parser → Workflow Builder → JSON/Code Generator
                                ↓
                    ┌──────────┼──────────┬──────────┐
                    ↓          ↓          ↓          ↓
              Liquid      Potentiostat   Pump      Camera
              Handler      Plugin       Controller  Plugin
               Plugin                    Plugin
                    ↓          ↓          ↓          ↓
                Adapters   Adapters    Adapters   Adapters
              (OT-2, etc) (SquidStat) (PLC, etc) (USB, SSH)
            ```

            ### Supported Device Types

            | Device Type | Operations | Adapters |
            |-------------|------------|----------|
            | Liquid Handler | aspirate, dispense, transfer, mix | OT-2, Hamilton |
            | Potentiostat | OCV, EIS, CV, CP, CA | SquidStat, Gamry |
            | Pump Controller | dispense, set flow rate, prime | PLC, peristaltic |
            | Camera | stream, capture, record | USB, SSH, IP |

            ### JSON Workflow Format

            Generated workflows use a unified schema compatible with external platforms:
            - Phases with sequential/parallel steps
            - Device-specific actions with parameters
            - Cross-platform import/export

            ---

            **Version**: 2.0.0 | **License**: MIT
            """)
            else:
                gr.Markdown("""
            ## About OT-2 NLP Agent

            A natural language processing agent that enables control of the Opentrons OT-2 liquid handling robot using plain English instructions.

            ### Key Features

            1. **Natural Language Parsing** — Understands experiment instructions in English
            2. **Protocol Validation** — Checks for errors before execution
            3. **Code Generation** — Produces standard Opentrons Python protocols
            4. **Multi-Step Support** — Parses complex multi-step instructions

            ### Architecture

            ```
            User Input → NLParser → OperationMapper → Protocol → Validator → Generator
            ```

            ### Supported Operations

            | Operation | Keywords |
            |-----------|----------|
            | Aspirate | aspirate, suck, draw, uptake |
            | Dispense | dispense, release, add, eject |
            | Transfer | transfer, move liquid |
            | Pick Up Tip | pick up tip, get tip |
            | Drop Tip | drop tip, eject tip, discard |
            | Mix | mix, stir, blend, homogenize |
            | Pause | pause, stop, halt |
            | Wait | wait, delay, sleep |

            ### Contact

            For issues or suggestions, please submit an Issue on the GitHub repository.

            ---

            **Version**: 1.0.0 | **License**: MIT
            """)

# Launch
if __name__ == "__main__":
    import os
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7863))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        css=CUSTOM_CSS
    )

#!/usr/bin/env python3
"""
Exp-Agent Live Dashboard

A real-time terminal dashboard showing:
- Device status panels (4 devices)
- Recovery pipeline state
- Event log with timestamps
- Decision statistics

Usage:
    python -m exp_agent.cli.dashboard
    python -m exp_agent.cli.dashboard --auto  # Auto-run scenarios
"""
import os
import sys
import time
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import deque

# ============================================================================
# Terminal Drawing
# ============================================================================

# ANSI codes
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"

# Box drawing
H_LINE = "─"
V_LINE = "│"
TL = "┌"
TR = "┐"
BL = "└"
BR = "┘"
T_DOWN = "┬"
T_UP = "┴"
T_RIGHT = "├"
T_LEFT = "┤"
CROSS = "┼"


def clear_screen():
    os.system("clear" if os.name == "posix" else "cls")


def move_cursor(row: int, col: int):
    print(f"\033[{row};{col}H", end="")


def draw_box(row: int, col: int, width: int, height: int, title: str = "", color: str = WHITE):
    """Draw a box with optional title."""
    # Top border
    move_cursor(row, col)
    print(f"{color}{TL}{H_LINE * (width - 2)}{TR}{RESET}", end="")

    # Title
    if title:
        move_cursor(row, col + 2)
        print(f"{color}{BOLD} {title} {RESET}", end="")

    # Sides
    for i in range(1, height - 1):
        move_cursor(row + i, col)
        print(f"{color}{V_LINE}{RESET}", end="")
        move_cursor(row + i, col + width - 1)
        print(f"{color}{V_LINE}{RESET}", end="")

    # Bottom border
    move_cursor(row + height - 1, col)
    print(f"{color}{BL}{H_LINE * (width - 2)}{BR}{RESET}", end="")


def write_at(row: int, col: int, text: str, max_width: int = 0):
    """Write text at position, optionally truncating."""
    move_cursor(row, col)
    if max_width > 0 and len(text) > max_width:
        text = text[:max_width - 3] + "..."
    print(text, end="")


# ============================================================================
# Dashboard State
# ============================================================================

@dataclass
class DeviceStatus:
    name: str
    device_type: str
    status: str = "idle"
    telemetry: Dict[str, Any] = field(default_factory=dict)
    last_error: Optional[str] = None
    color: str = WHITE


@dataclass
class PipelineState:
    current_stage: int = 0  # 0-7
    stages: List[str] = field(default_factory=lambda: [
        "IDLE", "SENSE", "CLASSIFY", "ANALYZE", "DECIDE", "EXECUTE", "VERIFY", "MEMORY"
    ])
    active_device: Optional[str] = None
    current_decision: Optional[str] = None


@dataclass
class DashboardState:
    devices: Dict[str, DeviceStatus] = field(default_factory=dict)
    pipeline: PipelineState = field(default_factory=PipelineState)
    events: deque = field(default_factory=lambda: deque(maxlen=12))
    stats: Dict[str, int] = field(default_factory=lambda: {
        "abort": 0, "degrade": 0, "retry": 0, "skip": 0, "total": 0
    })
    start_time: float = field(default_factory=time.time)


# ============================================================================
# Dashboard Renderer
# ============================================================================

class Dashboard:
    """Terminal dashboard for exp-agent monitoring."""

    def __init__(self):
        self.state = DashboardState()
        self.width = 100
        self.height = 35

        # Initialize devices
        self.state.devices = {
            "heater_1": DeviceStatus("heater_1", "Heater", color=RED),
            "pump_1": DeviceStatus("pump_1", "Pump", color=BLUE),
            "stage_1": DeviceStatus("stage_1", "Positioner", color=GREEN),
            "spec_1": DeviceStatus("spec_1", "Spectrometer", color=MAGENTA),
        }

    def render(self):
        """Render the full dashboard."""
        clear_screen()
        self._draw_header()
        self._draw_device_panels()
        self._draw_pipeline_panel()
        self._draw_stats_panel()
        self._draw_event_log()
        self._draw_footer()
        sys.stdout.flush()

    def _draw_header(self):
        """Draw title bar."""
        title = "🔬 EXP-AGENT RECOVERY DASHBOARD"
        uptime = time.time() - self.state.start_time
        time_str = f"Uptime: {int(uptime)}s"

        move_cursor(1, 1)
        print(f"{BG_BLUE}{WHITE}{BOLD} {title:^70} {time_str:>20} {RESET}")

    def _draw_device_panels(self):
        """Draw 4 device status panels in a 2x2 grid."""
        devices = list(self.state.devices.values())
        positions = [(3, 1), (3, 51), (12, 1), (12, 51)]  # (row, col)

        for i, (dev, (row, col)) in enumerate(zip(devices, positions)):
            self._draw_device_panel(dev, row, col, 48, 8)

    def _draw_device_panel(self, dev: DeviceStatus, row: int, col: int, width: int, height: int):
        """Draw a single device panel."""
        # Determine status color
        status_color = GREEN if dev.status == "idle" else YELLOW if dev.status == "running" else RED

        # Draw box
        draw_box(row, col, width, height, f"{dev.device_type}: {dev.name}", dev.color)

        # Status line
        write_at(row + 2, col + 2, f"Status: {status_color}{BOLD}{dev.status.upper():10}{RESET}")

        # Telemetry (up to 3 values)
        tel_items = list(dev.telemetry.items())[:3]
        for i, (key, val) in enumerate(tel_items):
            if isinstance(val, float):
                val_str = f"{val:.1f}"
            else:
                val_str = str(val)
            write_at(row + 3 + i, col + 2, f"{DIM}{key:15}{RESET} {val_str:>10}")

        # Last error (if any)
        if dev.last_error:
            write_at(row + 6, col + 2, f"{RED}Error: {dev.last_error[:35]}{RESET}")

    def _draw_pipeline_panel(self):
        """Draw the recovery pipeline state."""
        row, col = 21, 1
        width, height = 65, 6

        draw_box(row, col, width, height, "Recovery Pipeline", CYAN)

        # Draw pipeline stages
        stages = self.state.pipeline.stages
        stage_width = 7
        current = self.state.pipeline.current_stage

        move_cursor(row + 2, col + 2)
        for i, stage in enumerate(stages):
            if i == current:
                color = f"{BG_GREEN}{WHITE}{BOLD}"
            elif i < current:
                color = f"{GREEN}"
            else:
                color = f"{DIM}"
            print(f"{color}{stage[:6]:^7}{RESET}", end=" ")

        # Decision
        if self.state.pipeline.current_decision:
            dec = self.state.pipeline.current_decision.upper()
            dec_color = {"ABORT": RED, "DEGRADE": MAGENTA, "RETRY": YELLOW, "SKIP": CYAN}.get(dec, WHITE)
            write_at(row + 4, col + 2, f"Decision: {dec_color}{BOLD}{dec}{RESET}")

        if self.state.pipeline.active_device:
            write_at(row + 4, col + 30, f"Device: {self.state.pipeline.active_device}")

    def _draw_stats_panel(self):
        """Draw decision statistics."""
        row, col = 21, 67
        width, height = 32, 6

        draw_box(row, col, width, height, "Statistics", YELLOW)

        stats = self.state.stats
        total = stats["total"] or 1  # Avoid div by zero

        write_at(row + 2, col + 2, f"{RED}ABORT:  {stats['abort']:3}{RESET}  "
                                    f"{MAGENTA}DEGRADE: {stats['degrade']:3}{RESET}")
        write_at(row + 3, col + 2, f"{YELLOW}RETRY:  {stats['retry']:3}{RESET}  "
                                    f"{CYAN}SKIP:    {stats['skip']:3}{RESET}")
        write_at(row + 4, col + 2, f"Total decisions: {stats['total']}")

    def _draw_event_log(self):
        """Draw scrolling event log."""
        row, col = 28, 1
        width, height = 98, 7

        draw_box(row, col, width, height, "Event Log", WHITE)

        for i, event in enumerate(self.state.events):
            if i >= height - 2:
                break
            write_at(row + 1 + i, col + 2, event[:width - 4])

    def _draw_footer(self):
        """Draw footer with controls."""
        move_cursor(self.height, 1)
        print(f"{DIM}[Q] Quit  [1-4] Inject Fault  [R] Reset  [A] Auto Demo{RESET}", end="")

    # ========================================================================
    # State Updates
    # ========================================================================

    def log_event(self, msg: str, level: str = "INFO"):
        """Add event to log."""
        ts = datetime.now().strftime("%H:%M:%S")
        colors = {"INFO": GREEN, "WARN": YELLOW, "ERROR": RED}
        color = colors.get(level, WHITE)
        self.state.events.appendleft(f"{DIM}{ts}{RESET} {color}[{level}]{RESET} {msg}")

    def update_device(self, name: str, status: str = None, telemetry: Dict = None, error: str = None):
        """Update device state."""
        if name in self.state.devices:
            dev = self.state.devices[name]
            if status:
                dev.status = status
            if telemetry:
                dev.telemetry.update(telemetry)
            if error:
                dev.last_error = error

    def set_pipeline_stage(self, stage: int, device: str = None, decision: str = None):
        """Update pipeline state."""
        self.state.pipeline.current_stage = stage
        if device:
            self.state.pipeline.active_device = device
        if decision:
            self.state.pipeline.current_decision = decision
            self.state.stats[decision.lower()] = self.state.stats.get(decision.lower(), 0) + 1
            self.state.stats["total"] += 1

    def reset_pipeline(self):
        """Reset pipeline to idle."""
        self.state.pipeline.current_stage = 0
        self.state.pipeline.current_decision = None


# ============================================================================
# Demo Scenarios
# ============================================================================

def run_heater_fault(dashboard: Dashboard):
    """Simulate heater overshoot scenario."""
    dashboard.log_event("Starting heater overshoot scenario", "WARN")
    dashboard.update_device("heater_1", status="running", telemetry={"temperature": 100.0, "target": 120.0})
    dashboard.render()
    time.sleep(0.5)

    # Temperature rising
    for temp in [110.0, 120.0, 128.0, 135.0, 140.0]:
        dashboard.update_device("heater_1", telemetry={"temperature": temp})
        if temp > 130:
            dashboard.update_device("heater_1", status="error", error="Overshoot detected")
        dashboard.render()
        time.sleep(0.3)

    # Run recovery pipeline
    stages = ["SENSE", "CLASSIFY", "ANALYZE", "DECIDE", "EXECUTE", "VERIFY", "MEMORY"]
    for i, stage in enumerate(stages, 1):
        dashboard.set_pipeline_stage(i, device="heater_1")
        dashboard.log_event(f"Pipeline: {stage}", "INFO")
        dashboard.render()
        time.sleep(0.4)

    dashboard.set_pipeline_stage(4, decision="DEGRADE")
    dashboard.log_event("Decision: DEGRADE to 110°C", "WARN")
    dashboard.update_device("heater_1", status="running", telemetry={"temperature": 110.0}, error=None)
    dashboard.render()
    time.sleep(0.5)

    dashboard.reset_pipeline()
    dashboard.update_device("heater_1", status="idle")
    dashboard.render()


def run_pump_fault(dashboard: Dashboard):
    """Simulate pump flow blocked scenario."""
    dashboard.log_event("Starting pump blockage scenario", "WARN")
    dashboard.update_device("pump_1", status="running", telemetry={"flow_rate": 50.0, "pressure": 3.0})
    dashboard.render()
    time.sleep(0.5)

    # Flow dropping
    for flow in [40.0, 25.0, 10.0, 2.0, 0.0]:
        dashboard.update_device("pump_1", telemetry={"flow_rate": flow, "pressure": 3.0 + (50 - flow) / 10})
        if flow < 5:
            dashboard.update_device("pump_1", status="error", error="Flow blocked")
        dashboard.render()
        time.sleep(0.3)

    # Pipeline
    for i in range(1, 8):
        dashboard.set_pipeline_stage(i, device="pump_1")
        dashboard.render()
        time.sleep(0.3)

    dashboard.set_pipeline_stage(4, decision="DEGRADE")
    dashboard.log_event("Decision: DEGRADE (stop + prime)", "WARN")
    dashboard.update_device("pump_1", status="idle", telemetry={"flow_rate": 0.0, "pressure": 1.0}, error=None)
    dashboard.render()
    time.sleep(0.5)

    dashboard.reset_pipeline()
    dashboard.render()


def run_positioner_fault(dashboard: Dashboard):
    """Simulate positioner collision scenario."""
    dashboard.log_event("Starting positioner collision scenario", "ERROR")
    dashboard.update_device("stage_1", status="moving", telemetry={"x": 0.0, "y": 0.0, "z": 0.0})
    dashboard.render()
    time.sleep(0.5)

    # Moving then collision
    for x in [5.0, 10.0, 15.0, 15.0]:
        dashboard.update_device("stage_1", telemetry={"x": x, "y": 0.0, "z": 0.0})
        if x == 15.0:
            dashboard.update_device("stage_1", status="error", error="COLLISION at x=15")
        dashboard.render()
        time.sleep(0.3)

    # Pipeline
    for i in range(1, 8):
        dashboard.set_pipeline_stage(i, device="stage_1")
        dashboard.render()
        time.sleep(0.3)

    dashboard.set_pipeline_stage(4, decision="ABORT")
    dashboard.log_event("Decision: ABORT (non-recoverable)", "ERROR")
    dashboard.render()
    time.sleep(0.5)

    dashboard.reset_pipeline()
    dashboard.update_device("stage_1", status="error")
    dashboard.render()


def run_spectrometer_fault(dashboard: Dashboard):
    """Simulate spectrometer saturation scenario."""
    dashboard.log_event("Starting spectrometer saturation scenario", "WARN")
    dashboard.update_device("spec_1", status="acquiring", telemetry={"signal": 30000, "integration_ms": 100})
    dashboard.render()
    time.sleep(0.5)

    # Signal increasing
    for sig in [40000, 50000, 58000, 63000, 65000]:
        dashboard.update_device("spec_1", telemetry={"signal": sig})
        if sig > 60000:
            dashboard.update_device("spec_1", status="error", error="Signal saturated")
        dashboard.render()
        time.sleep(0.3)

    # Pipeline
    for i in range(1, 8):
        dashboard.set_pipeline_stage(i, device="spec_1")
        dashboard.render()
        time.sleep(0.3)

    dashboard.set_pipeline_stage(4, decision="DEGRADE")
    dashboard.log_event("Decision: DEGRADE (reduce integration)", "WARN")
    dashboard.update_device("spec_1", status="acquiring",
                            telemetry={"signal": 45000, "integration_ms": 50}, error=None)
    dashboard.render()
    time.sleep(0.5)

    dashboard.reset_pipeline()
    dashboard.update_device("spec_1", status="idle")
    dashboard.render()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp-Agent Live Dashboard")
    parser.add_argument("--auto", action="store_true", help="Auto-run all scenarios")
    args = parser.parse_args()

    dashboard = Dashboard()

    # Initial render
    dashboard.log_event("Dashboard started", "INFO")
    dashboard.log_event("Press 1-4 to inject faults, A for auto demo", "INFO")
    dashboard.render()

    if args.auto:
        time.sleep(1)
        run_heater_fault(dashboard)
        time.sleep(1)
        run_pump_fault(dashboard)
        time.sleep(1)
        run_positioner_fault(dashboard)
        time.sleep(1)
        run_spectrometer_fault(dashboard)
        dashboard.log_event("Auto demo complete", "INFO")
        dashboard.render()
        time.sleep(3)
        return

    # Interactive mode
    import select
    import tty
    import termios

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            # Check for input
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1).lower()

                if key == 'q':
                    break
                elif key == '1':
                    run_heater_fault(dashboard)
                elif key == '2':
                    run_pump_fault(dashboard)
                elif key == '3':
                    run_positioner_fault(dashboard)
                elif key == '4':
                    run_spectrometer_fault(dashboard)
                elif key == 'a':
                    run_heater_fault(dashboard)
                    time.sleep(0.5)
                    run_pump_fault(dashboard)
                    time.sleep(0.5)
                    run_positioner_fault(dashboard)
                    time.sleep(0.5)
                    run_spectrometer_fault(dashboard)
                elif key == 'r':
                    dashboard = Dashboard()
                    dashboard.log_event("Dashboard reset", "INFO")

                dashboard.render()

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        clear_screen()
        print("Dashboard closed.")


if __name__ == "__main__":
    main()

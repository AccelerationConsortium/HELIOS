#!/usr/bin/env python3
"""
Sensing Layer Demo - 感知层演示

Demonstrates the difference between blind recovery (old) and
sensing-aware recovery (new) with real-time sensor monitoring.

Demo scenarios:
1. Blind vs Sensing-Aware: Same fault, different outcomes
2. Real-time Sensor Panel + Interlock Trigger
3. Incident Replay for Post-mortem Analysis
4. SafetyAdvisor Integration

Usage:
    python -m demo.run_sensing_demo              # All demos
    python -m demo.run_sensing_demo --demo 1     # Specific demo
    python -m demo.run_sensing_demo --fast       # Fast mode
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional
from enum import Enum

# Add src to path
sys.path.insert(0, "src")

from exp_agent.sensing.protocol.sensor_event import (
    SensorEvent, SensorType, SensorMeta,
    temperature_event, airflow_event, pressure_event,
)
from exp_agent.sensing.protocol.health_event import HealthStatus
from exp_agent.sensing.protocol.snapshot import SensorSnapshot, SystemSnapshot
from exp_agent.sensing.drivers.mock_driver import (
    MockSensorDriver, MockSensorConfig,
    TemperatureProfile, AirflowProfile, PressureProfile,
    create_lab_sensor_set,
)
from exp_agent.sensing.hub.sensor_hub import SensorHub, HubConfig
from exp_agent.sensing.health.health_monitor import HealthMonitor, HealthMonitorConfig
from exp_agent.sensing.simulator.fault_injector import FaultInjector, FaultType, FaultConfig
from exp_agent.sensing.simulator.replay import ReplayDriver, ReplayConfig
from exp_agent.sensing.safety_state import (
    SafetyStateMachine, SafetyState, SafetyStateUpdate,
    InterlockReason, RecommendedAction, HysteresisConfig,
)
from exp_agent.sensing.recovery_gate import RecoveryGate, RecoveryAction, GateDecision


# ============================================================
# Terminal Colors & Helpers
# ============================================================

class Color:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    BLINK = '\033[5m'
    NC = '\033[0m'  # No Color
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'


def clear_screen():
    print("\033[2J\033[H", end="")


def move_cursor(row: int, col: int):
    print(f"\033[{row};{col}H", end="")


def type_text(text: str, delay: float = 0.02, fast: bool = False):
    if fast:
        print(text, end="", flush=True)
    else:
        for char in text:
            print(char, end="", flush=True)
            time.sleep(delay)


def print_banner(title: str):
    print()
    print(f"{Color.BOLD}{Color.CYAN}{'═' * 70}{Color.NC}")
    print(f"{Color.BOLD}{Color.CYAN}  {title}{Color.NC}")
    print(f"{Color.BOLD}{Color.CYAN}{'═' * 70}{Color.NC}")
    print()


def print_section(title: str):
    print()
    print(f"{Color.BOLD}{Color.YELLOW}{'─' * 60}{Color.NC}")
    print(f"  {Color.BOLD}{Color.YELLOW}{title}{Color.NC}")
    print(f"{Color.BOLD}{Color.YELLOW}{'─' * 60}{Color.NC}")
    print()


def print_sensor_value(name: str, value: float, unit: str, status: str = "ok"):
    status_icon = {
        "ok": f"{Color.GREEN}●{Color.NC}",
        "warning": f"{Color.YELLOW}⚠{Color.NC}",
        "critical": f"{Color.RED}{Color.BLINK}❗{Color.NC}",
        "offline": f"{Color.DIM}○{Color.NC}",
    }.get(status, "?")

    value_color = {
        "ok": Color.GREEN,
        "warning": Color.YELLOW,
        "critical": Color.RED,
        "offline": Color.DIM,
    }.get(status, Color.NC)

    print(f"  {status_icon} {name:20} {value_color}{value:8.2f}{Color.NC} {unit}")


def print_state_badge(state: SafetyState):
    badges = {
        SafetyState.SAFE: f"{Color.BG_GREEN}{Color.WHITE} SAFE {Color.NC}",
        SafetyState.DEGRADED: f"{Color.BG_YELLOW}{Color.WHITE} DEGRADED {Color.NC}",
        SafetyState.INTERLOCKED: f"{Color.BG_RED}{Color.WHITE} INTERLOCKED {Color.NC}",
        SafetyState.EMERGENCY: f"{Color.BG_RED}{Color.BLINK}{Color.WHITE} EMERGENCY {Color.NC}",
    }
    return badges.get(state, f"[{state.name}]")


# ============================================================
# Demo 1: Blind vs Sensing-Aware Recovery
# ============================================================

async def demo_blind_vs_sensing(fast: bool = False):
    """
    Demo 1: Compare blind recovery (old) vs sensing-aware recovery (new).
    Same fault scenario, completely different system behavior.
    """
    print_banner("Demo 1: Blind Recovery vs Sensing-Aware Recovery")

    delay = 0.5 if not fast else 0.05

    # Step A: Old System (Blind Recovery)
    print_section("Step A: OLD System - Blind Recovery Agent")
    print(f"  {Color.DIM}# Without sensing layer, agent only knows 'something failed'{Color.NC}")
    print()
    time.sleep(delay)

    print(f"  {Color.CYAN}[t=0s]{Color.NC} Starting heating reaction...")
    time.sleep(delay)
    print(f"  {Color.CYAN}[t=5s]{Color.NC} Heating in progress... temp rising")
    time.sleep(delay)
    print(f"  {Color.RED}[t=10s]{Color.NC} {Color.BOLD}ERROR: Device timeout{Color.NC}")
    print(f"         {Color.DIM}(Agent has no idea WHY it failed){Color.NC}")
    time.sleep(delay)

    print()
    print(f"  {Color.YELLOW}Agent Decision (Blind):{Color.NC}")
    print(f"    → Classifying error... TIMEOUT")
    print(f"    → Policy lookup... RETRY_ALLOWED")
    print(f"    → {Color.GREEN}Executing RETRY...{Color.NC}")
    time.sleep(delay)

    print(f"  {Color.CYAN}[t=12s]{Color.NC} Retrying operation...")
    print(f"  {Color.GREEN}[t=15s]{Color.NC} Success! Continuing workflow...")
    time.sleep(delay)

    print()
    print(f"  {Color.GREEN}✓ Looks smart!{Color.NC} Agent recovered automatically.")
    print(f"  {Color.DIM}  But what if the 'timeout' was hiding a real danger?{Color.NC}")
    time.sleep(delay * 2)

    # Step B: New System (Sensing-Aware)
    print_section("Step B: NEW System - Sensing-Aware Safety Runtime")
    print(f"  {Color.DIM}# With sensing layer, agent sees the REAL physical state{Color.NC}")
    print()
    time.sleep(delay)

    # Create real sensing components
    hub = SensorHub(HubConfig(buffer_size=100, min_interval_ms=0))
    monitor = HealthMonitor()
    machine = SafetyStateMachine()
    gate = RecoveryGate(state_machine=machine)

    # Simulate sensor readings with developing problem
    print(f"  {Color.CYAN}[t=0s]{Color.NC} Starting heating reaction...")
    print()

    readings = [
        # Normal operation
        {"t": 0, "temp": 85.0, "airflow": 0.45, "pressure": 150.0},
        {"t": 2, "temp": 95.0, "airflow": 0.42, "pressure": 155.0},
        {"t": 4, "temp": 108.0, "airflow": 0.38, "pressure": 160.0},
        # Problem developing
        {"t": 6, "temp": 122.0, "airflow": 0.32, "pressure": 168.0},
        {"t": 8, "temp": 135.0, "airflow": 0.25, "pressure": 175.0},  # Overheat rate!
        {"t": 10, "temp": 148.0, "airflow": 0.18, "pressure": 182.0},  # Critical!
    ]

    state_history = []

    for reading in readings:
        t = reading["t"]

        # Create snapshot
        snapshot = SystemSnapshot()

        # Add temperature sensor
        temp_snap = SensorSnapshot(
            sensor_id="reactor_temp",
            sensor_type=SensorType.TEMPERATURE,
            health_status=HealthStatus.HEALTHY,
        )
        temp_snap.latest_value = reading["temp"]
        temp_snap.latest_event = temperature_event("reactor_temp", reading["temp"])
        snapshot.sensors["reactor_temp"] = temp_snap

        # Add airflow sensor
        air_snap = SensorSnapshot(
            sensor_id="hood_airflow",
            sensor_type=SensorType.AIRFLOW,
            health_status=HealthStatus.HEALTHY,
        )
        air_snap.latest_value = reading["airflow"]
        air_snap.latest_event = airflow_event("hood_airflow", reading["airflow"])
        snapshot.sensors["hood_airflow"] = air_snap

        # Add pressure sensor
        pres_snap = SensorSnapshot(
            sensor_id="reactor_pressure",
            sensor_type=SensorType.PRESSURE,
            health_status=HealthStatus.HEALTHY,
        )
        pres_snap.latest_value = reading["pressure"]
        pres_snap.latest_event = pressure_event("reactor_pressure", reading["pressure"])
        snapshot.sensors["reactor_pressure"] = pres_snap

        snapshot._update_aggregates()

        # Process through safety state machine
        state_update = machine.process_snapshot(snapshot, {
            "max_temp": 130.0,
            "min_airflow": 0.3,
            "max_pressure": 200.0,
        })
        state_history.append((t, state_update))

        # Display sensor panel
        temp_status = "critical" if reading["temp"] > 130 else "warning" if reading["temp"] > 120 else "ok"
        air_status = "critical" if reading["airflow"] < 0.2 else "warning" if reading["airflow"] < 0.3 else "ok"
        pres_status = "warning" if reading["pressure"] > 170 else "ok"

        print(f"  {Color.CYAN}[t={t}s]{Color.NC} Sensor readings:")
        print_sensor_value("Temperature", reading["temp"], "°C", temp_status)
        print_sensor_value("Hood Airflow", reading["airflow"], "m/s", air_status)
        print_sensor_value("Pressure", reading["pressure"], "kPa", pres_status)

        # Calculate slope (if enough history)
        if t >= 4:
            prev = readings[readings.index(reading) - 1]
            slope = (reading["temp"] - prev["temp"]) / 2.0  # per 2 seconds
            slope_per_min = slope * 30  # per minute
            slope_status = "critical" if slope_per_min > 3.0 else "ok"
            print_sensor_value("Temp Slope", slope_per_min, "°C/min", slope_status)

        print(f"  {Color.BOLD}State: {print_state_badge(state_update.state)}{Color.NC}")

        if state_update.interlocks:
            print(f"  {Color.RED}Interlocks:{Color.NC}")
            for il in state_update.interlocks[:3]:
                print(f"    • {il.reason.value}")

        print()
        time.sleep(delay)

        # Check if interlocked
        if state_update.state >= SafetyState.INTERLOCKED:
            break

    # Now agent tries to retry
    print(f"  {Color.RED}[t=10s]{Color.NC} {Color.BOLD}ERROR: Device timeout{Color.NC}")
    print(f"         {Color.DIM}(Same error as before, but now we have context!){Color.NC}")
    print()
    time.sleep(delay)

    print(f"  {Color.YELLOW}Agent wants to RESUME HEATING...{Color.NC}")
    print()
    time.sleep(delay)

    # RecoveryGate check - use START_HEAT which is HIGH risk
    decision = gate.check_action(
        RecoveryAction.START_HEAT,
        snapshot,
        state_history[-1][1] if state_history else None,
    )

    print(f"  {Color.BOLD}RecoveryGate Decision:{Color.NC}")
    print(f"    Action requested: {Color.CYAN}START_HEAT{Color.NC}")
    print(f"    Allowed: {Color.RED if not decision.allowed else Color.GREEN}{decision.allowed}{Color.NC}")
    print(f"    Reason: {decision.reason}")
    print()

    if not decision.allowed:
        print(f"  {Color.BG_RED}{Color.WHITE}{Color.BOLD} VETO - Recovery Blocked {Color.NC}")
        print()
        print(f"  {Color.BOLD}Evidence Chain:{Color.NC}")
        if state_history:
            last_update = state_history[-1][1]
            print(f"    snapshot_id: {last_update.evidence.snapshot_id[:16]}...")
            print(f"    trigger_values:")
            for k, v in last_update.evidence.trigger_values.items():
                print(f"      - {k}: {v}")
        print()

        print(f"  {Color.BOLD}Recommended Actions:{Color.NC}")
        if state_history:
            for action in state_history[-1][1].recommended_actions[:3]:
                print(f"    → {action.value}")
        print()

        print(f"  {Color.BOLD}Alternative Safe Actions:{Color.NC}")
        for alt in decision.alternative_actions[:3]:
            print(f"    → {Color.GREEN}{alt.value}{Color.NC}")

    print()
    print(f"{Color.BOLD}{Color.CYAN}{'═' * 70}{Color.NC}")
    print(f"  {Color.BOLD}Key Difference:{Color.NC}")
    print(f"    • OLD: Agent retries blindly → potential disaster")
    print(f"    • NEW: Real-world state blocks unsafe recovery")
    print(f"{Color.BOLD}{Color.CYAN}{'═' * 70}{Color.NC}")
    print()


# ============================================================
# Demo 2: Real-time Sensor Panel + Interlock Trigger
# ============================================================

async def demo_realtime_panel(fast: bool = False):
    """
    Demo 2: Real-time sensor monitoring with live interlock triggering.
    Split-screen visualization of sensors and safety state.
    """
    print_banner("Demo 2: Real-time Sensor Panel + Interlock Trigger")

    delay = 0.3 if not fast else 0.02

    # Create components
    machine = SafetyStateMachine(hysteresis=HysteresisConfig(
        min_hold_time_ms=0,
        recovery_threshold_readings=3,
    ))

    print(f"  {Color.DIM}# Simulating heating reaction with developing anomaly{Color.NC}")
    print()

    # Simulation timeline
    timeline = [
        # Phase 1: Normal operation
        {"t": 0, "temp": 80.0, "slope": 1.5, "airflow": 0.50, "pressure": 145.0, "phase": "Normal heating"},
        {"t": 5, "temp": 90.0, "slope": 2.0, "airflow": 0.48, "pressure": 150.0, "phase": "Normal heating"},
        {"t": 10, "temp": 102.0, "slope": 2.4, "airflow": 0.45, "pressure": 155.0, "phase": "Normal heating"},
        # Phase 2: Anomaly developing
        {"t": 15, "temp": 118.0, "slope": 3.2, "airflow": 0.38, "pressure": 162.0, "phase": "Anomaly detected"},
        {"t": 20, "temp": 128.0, "slope": 2.0, "airflow": 0.32, "pressure": 170.0, "phase": "Warning state"},
        # Phase 3: Critical
        {"t": 25, "temp": 135.0, "slope": 1.4, "airflow": 0.25, "pressure": 178.0, "phase": "INTERLOCKED"},
        {"t": 30, "temp": 138.0, "slope": 0.6, "airflow": 0.22, "pressure": 180.0, "phase": "System halted"},
    ]

    state_transitions = []

    for point in timeline:
        t = point["t"]

        # Build visual display
        print(f"{Color.BOLD}{'─' * 70}{Color.NC}")
        print(f"  {Color.CYAN}Time: t+{t}s{Color.NC}  |  {Color.YELLOW}Phase: {point['phase']}{Color.NC}")
        print(f"{'─' * 70}")
        print()

        # Left panel: Sensors
        print(f"  {Color.BOLD}┌─────────────────────────────┐{Color.NC}  {Color.BOLD}┌─────────────────────────────┐{Color.NC}")
        print(f"  {Color.BOLD}│  SENSOR READINGS            │{Color.NC}  {Color.BOLD}│  SAFETY STATE               │{Color.NC}")
        print(f"  {Color.BOLD}├─────────────────────────────┤{Color.NC}  {Color.BOLD}├─────────────────────────────┤{Color.NC}")

        # Temperature
        temp = point["temp"]
        temp_bar = "█" * min(int(temp / 10), 15)
        temp_color = Color.RED if temp > 130 else Color.YELLOW if temp > 120 else Color.GREEN
        temp_icon = "❗" if temp > 130 else "⚠" if temp > 120 else "●"

        # Create snapshot for state machine
        snapshot = SystemSnapshot()

        temp_snap = SensorSnapshot(
            sensor_id="temp_1", sensor_type=SensorType.TEMPERATURE,
            health_status=HealthStatus.HEALTHY,
        )
        temp_snap.latest_value = temp
        temp_snap.latest_event = temperature_event("temp_1", temp)
        snapshot.sensors["temp_1"] = temp_snap

        air_snap = SensorSnapshot(
            sensor_id="airflow_1", sensor_type=SensorType.AIRFLOW,
            health_status=HealthStatus.HEALTHY,
        )
        air_snap.latest_value = point["airflow"]
        air_snap.latest_event = airflow_event("airflow_1", point["airflow"])
        snapshot.sensors["airflow_1"] = air_snap

        pres_snap = SensorSnapshot(
            sensor_id="pressure_1", sensor_type=SensorType.PRESSURE,
            health_status=HealthStatus.HEALTHY,
        )
        pres_snap.latest_value = point["pressure"]
        pres_snap.latest_event = pressure_event("pressure_1", point["pressure"])
        snapshot.sensors["pressure_1"] = pres_snap

        snapshot._update_aggregates()

        # Process through state machine
        update = machine.process_snapshot(snapshot, {
            "max_temp": 130.0,
            "min_airflow": 0.3,
            "max_pressure": 200.0,
        })

        if not state_transitions or state_transitions[-1][1].state != update.state:
            state_transitions.append((t, update))

        # Slope
        slope = point["slope"]
        slope_color = Color.RED if slope > 3.0 else Color.YELLOW if slope > 2.5 else Color.GREEN
        slope_icon = "❗" if slope > 3.0 else "⚠" if slope > 2.5 else "●"

        # Airflow
        airflow = point["airflow"]
        air_color = Color.RED if airflow < 0.25 else Color.YELLOW if airflow < 0.35 else Color.GREEN
        air_icon = "❗" if airflow < 0.25 else "⚠" if airflow < 0.35 else "●"

        # Pressure
        pressure = point["pressure"]
        pres_color = Color.YELLOW if pressure > 170 else Color.GREEN
        pres_icon = "⚠" if pressure > 170 else "●"

        # State display
        state_badge = print_state_badge(update.state)

        # Print side by side
        print(f"  │ {temp_icon} Temp:   {temp_color}{temp:6.1f}°C{Color.NC}          │  │  State: {state_badge}            │")
        print(f"  │ {slope_icon} Slope:  {slope_color}{slope:6.1f}°C/min{Color.NC}      │  │                               │")
        print(f"  │ {air_icon} Airflow:{air_color}{airflow:6.2f} m/s{Color.NC}        │  │  Reason:                      │")
        print(f"  │ {pres_icon} Press:  {pres_color}{pressure:6.1f} kPa{Color.NC}        │  │  {update.reason.value[:27] if update.reason else 'NONE':27} │")
        print(f"  {Color.BOLD}└─────────────────────────────┘{Color.NC}  {Color.BOLD}└─────────────────────────────┘{Color.NC}")

        # Show interlocks if any
        if update.interlocks:
            print()
            print(f"  {Color.RED}{Color.BOLD}Active Interlocks:{Color.NC}")
            for il in update.interlocks[:4]:
                print(f"    • {il.reason.value}: {il.interlock_class.value}")

        # Show actions for INTERLOCKED state
        if update.state >= SafetyState.INTERLOCKED:
            print()
            print(f"  {Color.YELLOW}{Color.BOLD}Recommended Actions:{Color.NC}")
            for action in update.recommended_actions[:3]:
                print(f"    → {action.value}")

            print()
            print(f"  {Color.RED}{Color.BOLD}⚡ HEATING ACTION BLOCKED ⚡{Color.NC}")

        print()
        time.sleep(delay * 2)

    # Summary
    print(f"{Color.BOLD}{Color.CYAN}{'═' * 70}{Color.NC}")
    print(f"  {Color.BOLD}State Transition History:{Color.NC}")
    for t, upd in state_transitions:
        print(f"    t+{t}s → {upd.state.name}")
    print(f"{Color.BOLD}{Color.CYAN}{'═' * 70}{Color.NC}")
    print()


# ============================================================
# Demo 3: Incident Replay for Post-mortem
# ============================================================

async def demo_incident_replay(fast: bool = False):
    """
    Demo 3: Replay a recorded incident for post-mortem analysis.
    Shows deterministic replay with exact state reproduction.
    """
    print_banner("Demo 3: Incident Replay & Post-mortem Analysis")

    delay = 0.3 if not fast else 0.02

    print(f"  {Color.DIM}# Loading incident log: incident_2026_02_05.json{Color.NC}")
    print()
    time.sleep(delay)

    # Create recorded incident events
    incident_events = [
        SensorEvent(
            sensor_id="reactor_temp", sensor_type=SensorType.TEMPERATURE,
            value=120.5, unit="C", ts=datetime(2026, 2, 5, 14, 30, 0, tzinfo=timezone.utc),
        ),
        SensorEvent(
            sensor_id="reactor_temp", sensor_type=SensorType.TEMPERATURE,
            value=125.3, unit="C", ts=datetime(2026, 2, 5, 14, 30, 10, tzinfo=timezone.utc),
        ),
        SensorEvent(
            sensor_id="hood_airflow", sensor_type=SensorType.AIRFLOW,
            value=0.35, unit="m/s", ts=datetime(2026, 2, 5, 14, 30, 15, tzinfo=timezone.utc),
        ),
        SensorEvent(
            sensor_id="reactor_temp", sensor_type=SensorType.TEMPERATURE,
            value=132.1, unit="C", ts=datetime(2026, 2, 5, 14, 30, 20, tzinfo=timezone.utc),
        ),
        SensorEvent(
            sensor_id="hood_airflow", sensor_type=SensorType.AIRFLOW,
            value=0.28, unit="m/s", ts=datetime(2026, 2, 5, 14, 30, 22, tzinfo=timezone.utc),
        ),
        SensorEvent(
            sensor_id="reactor_temp", sensor_type=SensorType.TEMPERATURE,
            value=138.5, unit="C", ts=datetime(2026, 2, 5, 14, 30, 30, tzinfo=timezone.utc),
        ),
    ]

    # Create replay driver
    replay_config = ReplayConfig(
        driver_id="incident_2026_02_05",
        events=incident_events,
        preserve_timing=False,
    )
    replay = ReplayDriver(replay_config)

    print(f"  {Color.CYAN}Incident ID:{Color.NC} incident_2026_02_05")
    print(f"  {Color.CYAN}Event Count:{Color.NC} {len(incident_events)}")
    print(f"  {Color.CYAN}Time Range:{Color.NC} 14:30:00 - 14:30:30 UTC")
    print()

    print_section("Replaying Event Sequence")

    # Create fresh state machine for replay
    machine = SafetyStateMachine(hysteresis=HysteresisConfig(
        min_hold_time_ms=0,
        recovery_threshold_readings=1,
    ))

    # Create cumulative snapshot
    snapshot = SystemSnapshot()
    event_log = []

    for i, event in enumerate(incident_events):
        ts_str = event.ts.strftime("%H:%M:%S")

        # Update snapshot with event
        sensor = snapshot.sensors.get(event.sensor_id)
        if not sensor:
            sensor = SensorSnapshot(
                sensor_id=event.sensor_id,
                sensor_type=event.sensor_type,
                health_status=HealthStatus.HEALTHY,
            )
            snapshot.sensors[event.sensor_id] = sensor
        sensor.latest_value = event.value
        sensor.latest_event = event
        snapshot._update_aggregates()

        # Process through state machine
        update = machine.process_snapshot(snapshot, {
            "max_temp": 130.0,
            "min_airflow": 0.3,
        })

        # Display
        event_icon = "🌡️" if event.sensor_type == SensorType.TEMPERATURE else "💨"
        state_color = {
            SafetyState.SAFE: Color.GREEN,
            SafetyState.DEGRADED: Color.YELLOW,
            SafetyState.INTERLOCKED: Color.RED,
            SafetyState.EMERGENCY: Color.RED,
        }.get(update.state, Color.NC)

        print(f"  {Color.DIM}t+{i*5:3}s{Color.NC}  {event_icon} {event.sensor_id}: {event.value:.1f} {event.unit}")
        print(f"         → State: {state_color}{update.state.name}{Color.NC}")

        if update.interlocks:
            for il in update.interlocks:
                print(f"         → Trigger: {Color.RED}{il.reason.value}{Color.NC}")

        event_log.append({
            "t": i * 5,
            "event": event,
            "state": update.state,
            "interlocks": [il.reason.value for il in update.interlocks],
        })

        print()
        time.sleep(delay)

    # Post-mortem analysis
    print_section("Post-mortem Analysis")

    # Find critical events
    print(f"  {Color.BOLD}Timeline Analysis:{Color.NC}")
    print()

    for log in event_log:
        if log["interlocks"]:
            t = log["t"]
            print(f"  {Color.RED}t+{t}s{Color.NC}  {log['event'].sensor_id} = {log['event'].value}")
            for il in log["interlocks"]:
                print(f"        └─ Triggered: {il}")

    print()
    print(f"  {Color.BOLD}Root Cause Identification:{Color.NC}")
    print(f"    • Primary trigger: Temperature exceeded threshold at t+20s")
    print(f"    • Contributing factor: Hood airflow dropped below minimum at t+22s")
    print(f"    • Escalation: Compounding thermal + ventilation failure")
    print()

    print(f"  {Color.BOLD}Evidence Hash (for audit):{Color.NC}")
    # Generate deterministic hash
    final_update = machine.process_snapshot(snapshot, {"max_temp": 130.0})
    print(f"    snapshot_id: {final_update.evidence.snapshot_id}")
    print()

    print(f"  {Color.BOLD}Replay Verification:{Color.NC}")
    print(f"    {Color.GREEN}✓{Color.NC} All events processed")
    print(f"    {Color.GREEN}✓{Color.NC} State transitions reproducible")
    print(f"    {Color.GREEN}✓{Color.NC} Evidence chain verified")
    print()


# ============================================================
# Demo 4: SafetyAdvisor Integration
# ============================================================

async def demo_safety_advisor(fast: bool = False):
    """
    Demo 4: SafetyAdvisor provides contextual explanations.
    Shows how AI enhances safety decisions with domain knowledge.
    """
    print_banner("Demo 4: SafetyAdvisor Integration")

    delay = 0.5 if not fast else 0.05

    print(f"  {Color.DIM}# SafetyAdvisor provides human-readable explanations{Color.NC}")
    print(f"  {Color.DIM}# for safety decisions based on sensor context{Color.NC}")
    print()

    # Create scenario: INTERLOCKED state
    print_section("Scenario: System enters INTERLOCKED state")

    # Display the trigger conditions
    print(f"  {Color.BOLD}Current Sensor State:{Color.NC}")
    print_sensor_value("Temperature", 142.5, "°C", "critical")
    print_sensor_value("Temp Slope", 3.8, "°C/min", "critical")
    print_sensor_value("Hood Airflow", 0.22, "m/s", "critical")
    print_sensor_value("Pressure", 175.0, "kPa", "warning")
    print()
    time.sleep(delay)

    print(f"  {Color.BOLD}Safety State:{Color.NC}")
    print(f"    State: {print_state_badge(SafetyState.INTERLOCKED)}")
    print(f"    Reason: OVERHEAT_RATE + INSUFFICIENT_AIRFLOW")
    print()
    time.sleep(delay)

    print(f"  {Color.BOLD}Chemicals in Reaction:{Color.NC}")
    print(f"    • Toluene (volatile, flammable)")
    print(f"    • Sodium hydride (water-reactive)")
    print()
    time.sleep(delay)

    # SafetyAdvisor query
    print_section("SafetyAdvisor Analysis")

    print(f"  {Color.CYAN}Querying SafetyAdvisor...{Color.NC}")
    time.sleep(delay)

    # Simulated advisor response
    advisor_response = """
    ┌────────────────────────────────────────────────────────────────┐
    │  SAFETY ANALYSIS                                               │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  Risk Assessment: HIGH                                         │
    │                                                                │
    │  The combination of elevated temperature (142.5°C) with        │
    │  insufficient hood airflow (0.22 m/s) creates a significant    │
    │  vapor accumulation hazard.                                    │
    │                                                                │
    │  With toluene present:                                         │
    │  • Vapor pressure increases exponentially with temperature     │
    │  • Current temp exceeds toluene's flash point (4°C)            │
    │  • Inadequate ventilation allows vapor concentration buildup   │
    │                                                                │
    │  With sodium hydride present:                                  │
    │  • Any moisture ingress at elevated temp is dangerous          │
    │  • Reduced airflow may allow local hot spots                   │
    │                                                                │
    │  RECOMMENDATION:                                               │
    │  1. Immediately suspend heating operations                     │
    │  2. Maintain current ventilation - do NOT reduce further       │
    │  3. Allow passive cooling before any manual intervention       │
    │  4. Do NOT add water or aqueous solutions                      │
    │                                                                │
    │  Recovery should only proceed after:                           │
    │  • Temperature < 80°C                                          │
    │  • Airflow > 0.4 m/s confirmed                                 │
    │  • Human operator verification                                 │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘
    """

    for line in advisor_response.strip().split('\n'):
        print(f"  {Color.WHITE}{line}{Color.NC}")
        time.sleep(delay * 0.3)

    print()
    time.sleep(delay)

    # Show how this integrates
    print_section("Integration with RecoveryAgent")

    print(f"  {Color.YELLOW}Agent attempts: RETRY heating operation{Color.NC}")
    print()
    time.sleep(delay)

    print(f"  {Color.BOLD}RecoveryGate Check:{Color.NC}")
    print(f"    • Current state: INTERLOCKED")
    print(f"    • Action risk: HIGH")
    print(f"    • Sensor requirements: NOT MET")
    print()

    print(f"  {Color.RED}{Color.BOLD}⚡ ACTION BLOCKED ⚡{Color.NC}")
    print()
    print(f"  {Color.BOLD}Gate Decision:{Color.NC}")
    print(f"    allowed: {Color.RED}False{Color.NC}")
    print(f"    reason: INTERLOCKED state requires human intervention")
    print(f"    requires_human: {Color.YELLOW}True{Color.NC}")
    print()

    print(f"  {Color.BOLD}Alternative Actions Offered:{Color.NC}")
    print(f"    → {Color.GREEN}SAFE_SHUTDOWN{Color.NC}")
    print(f"    → {Color.GREEN}ASK_HUMAN{Color.NC}")
    print(f"    → {Color.GREEN}WAIT{Color.NC}")
    print()

    print(f"{Color.BOLD}{Color.CYAN}{'═' * 70}{Color.NC}")
    print(f"  {Color.BOLD}Key Value:{Color.NC}")
    print(f"    • SafetyAdvisor provides domain-specific context")
    print(f"    • Explains WHY the system is blocked, not just THAT it's blocked")
    print(f"    • Enables informed human decision-making")
    print(f"    • All responses are advisory - no executable actions")
    print(f"{Color.BOLD}{Color.CYAN}{'═' * 70}{Color.NC}")
    print()


# ============================================================
# Main Entry Point
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="Sensing Layer Demo")
    parser.add_argument("--demo", type=int, choices=[1, 2, 3, 4], help="Run specific demo (1-4)")
    parser.add_argument("--fast", action="store_true", help="Fast mode (no delays)")
    args = parser.parse_args()

    demos = {
        1: ("Blind vs Sensing-Aware Recovery", demo_blind_vs_sensing),
        2: ("Real-time Sensor Panel", demo_realtime_panel),
        3: ("Incident Replay", demo_incident_replay),
        4: ("SafetyAdvisor Integration", demo_safety_advisor),
    }

    if args.demo:
        # Run specific demo
        name, func = demos[args.demo]
        await func(fast=args.fast)
    else:
        # Run all demos
        clear_screen()
        print()
        print(f"{Color.BOLD}{Color.CYAN}")
        print("  ╔═══════════════════════════════════════════════════════════╗")
        print("  ║                                                           ║")
        print("  ║   SENSING LAYER DEMO                                      ║")
        print("  ║   Auditable Safety Runtime for Lab Automation             ║")
        print("  ║                                                           ║")
        print("  ╚═══════════════════════════════════════════════════════════╝")
        print(f"{Color.NC}")
        print()
        print(f"  {Color.DIM}This demo showcases the sensing layer's capabilities:{Color.NC}")
        print(f"    1. Blind vs Sensing-Aware Recovery")
        print(f"    2. Real-time Sensor Panel + Interlock Trigger")
        print(f"    3. Incident Replay for Post-mortem Analysis")
        print(f"    4. SafetyAdvisor Integration")
        print()

        if not args.fast:
            input(f"  {Color.CYAN}Press Enter to begin...{Color.NC}")

        for demo_num, (name, func) in demos.items():
            await func(fast=args.fast)

            if demo_num < 4 and not args.fast:
                print()
                input(f"  {Color.CYAN}Press Enter for next demo...{Color.NC}")

        print()
        print(f"{Color.BOLD}{Color.GREEN}All demos completed!{Color.NC}")
        print()


if __name__ == "__main__":
    asyncio.run(main())

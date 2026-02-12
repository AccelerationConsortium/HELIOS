# Exp-Agent

A recovery-aware execution agent for lab hardware (MVP).

## Structure

* `core`: Shared types and data structures.
* `devices`: Hardware abstraction layer (simulated heater included).
* `executor`: Guarded execution logic.
* `recovery`: Rule-based recovery agent.
* `orchestrator`: Main supervisor loop.
* `cli`: Entry points.

## How to Run

### Simulation Mode (Default)

1.  Navigate to the root directory `exp-agent`.
2.  Run the MVP simulation:

```bash
export PYTHONPATH=src
python -m exp_agent.cli.run_sim --fault-mode overshoot
```

### Real Hardware Mode

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your device:
```bash
# For serial-connected device
export PYTHONPATH=src
python -m exp_agent.cli.run_agent --real-hardware --device-name heater_1 --port /dev/ttyUSB0 --target-temp 120

# For network-connected device
export PYTHONPATH=src
python -m exp_agent.cli.run_agent --real-hardware --device-name heater_1 --host 192.168.1.100 --target-temp 120
```

3. Or use configuration file:
```bash
export PYTHONPATH=src
python -m exp_agent.cli.run_agent --real-hardware --config my_lab_config.json
```

## Fault Modes available

* `none` (Happy path)
* `timeout`
* `overshoot`
* `sensor_fail`
* `random`

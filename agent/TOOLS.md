# TOOLS.md — Lab Environment Configuration

Skills define *what* each instrument can do. This file records *your* specific setup —
the addresses, ports, and physical layout that are unique to this lab.

## Hardware Network

### OT-2 Robot
- **Address:** `ROBOT_IP` (default: `localhost` in simulated mode)
- **Port:** 31950 (Opentrons HTTP API)
- **Deck Layout:**
  - Slot 1-3: Sample labware (varies by protocol)
  - Slot 4-6: Reagent reservoirs
  - Slot 7-9: Tip racks (300 uL, 20 uL)
  - Slot 10: Waste
  - Slot 11: Wash station

### PLC Controller
- **Protocol:** Modbus TCP
- **Address:** `PLC_ADDRESS` (default: `192.168.1.100`)
- **Port:** 502
- **Pumps:**
  - Pump 1: Electrolyte delivery
  - Pump 2: Rinse solution
  - Pump 3: Waste extraction
- **Ultrasonic Stirrers:**
  - Unit 1: Primary mixing
  - Unit 2: Secondary mixing

### Relay Board
- **Connection:** USB Serial
- **Port:** `RELAY_PORT` (default: `/dev/ttyUSB0`)
- **Baud Rate:** 9600
- **Channels:**
  - CH1-CH4: Working electrode selection
  - CH5-CH8: Counter electrode selection
  - CH9-CH12: Reference electrode selection
  - CH13-CH16: Auxiliary / spare

### Squidstat Potentiostat
- **Connection:** USB Serial
- **Port:** `SQUIDSTAT_PORT` (default: `/dev/ttyUSB1`)
- **Capabilities:** CV, LSV, EIS, CP, CA, OCV
- **Channels:** 1-4 (depends on model)

## Software Stack

- **Orchestrator:** FastAPI on `http://localhost:8000`
- **Database:** SQLite with WAL mode at `DATA_DIR/orchestrator.db`
- **Artifact Store:** `OBJECT_STORE_DIR/` (file-based, SHA-256 checksums)
- **Adapter Mode:** `ADAPTER_MODE` (simulated | battery_lab)
- **Dry Run:** `ADAPTER_DRY_RUN` (true = log actions without hardware calls)

## Current Mode

This lab is running in **simulated mode** by default.
To connect real hardware, set environment variables:
```
ADAPTER_MODE=battery_lab
ADAPTER_DRY_RUN=false
ROBOT_IP=<robot-ip>
PLC_ADDRESS=<plc-ip>
RELAY_PORT=<serial-port>
SQUIDSTAT_PORT=<serial-port>
```

---

*Update this file when hardware changes. IP addresses, deck layouts, and channel assignments are lab-specific and should not leak into shared skill definitions.*

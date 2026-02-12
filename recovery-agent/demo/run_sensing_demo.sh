#!/usr/bin/env bash
# ============================================================
# Sensing Layer Demo Script
# Auditable Safety Runtime for Lab Automation
# ============================================================
#
# This demo showcases the sensing layer's capabilities:
#   1. Blind vs Sensing-Aware Recovery
#   2. Real-time Sensor Panel + Interlock Trigger
#   3. Incident Replay for Post-mortem Analysis
#   4. SafetyAdvisor Integration
#
# Usage:
#   ./demo/run_sensing_demo.sh           # All demos
#   ./demo/run_sensing_demo.sh --demo 1  # Specific demo
#   ./demo/run_sensing_demo.sh --fast    # Fast mode
#
# ============================================================

set -e
cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/src"

# Run the Python demo script
uv run python -m demo.run_sensing_demo "$@"

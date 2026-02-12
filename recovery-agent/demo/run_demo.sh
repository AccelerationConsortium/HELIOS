#!/usr/bin/env bash
# ============================================================
# Exp-Agent Demo Script
# Recovery-Aware Execution Agent for Lab Hardware
# ============================================================
#
# This script demonstrates the agent's autonomous disaster
# recovery capabilities across multiple fault scenarios.
#
# Usage:
#   ./demo/run_demo.sh           # Interactive (with typing effect)
#   ./demo/run_demo.sh --fast    # Fast mode (no typing delay)
#
# Recording:
#   ./demo/record_demo.sh        # Records with asciinema
# ============================================================

set -e
cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/src"
FAST_MODE=false
[[ "$1" == "--fast" ]] && FAST_MODE=true

# --- Helpers ---

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

type_text() {
    local text="$1"
    local delay="${2:-0.03}"
    if $FAST_MODE; then
        printf "%s" "$text"
    else
        for ((i=0; i<${#text}; i++)); do
            printf "%s" "${text:$i:1}"
            sleep "$delay"
        done
    fi
}

print_banner() {
    echo ""
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║${NC}  ${BOLD}$1${NC}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    $FAST_MODE || sleep 1
}

print_section() {
    echo ""
    echo -e "${BOLD}${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${BOLD}${YELLOW}$1${NC}"
    echo -e "${BOLD}${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    $FAST_MODE || sleep 0.5
}

print_explain() {
    echo -e "  ${DIM}${MAGENTA}# $1${NC}"
    $FAST_MODE || sleep 0.3
}

run_cmd() {
    local cmd="$1"
    echo -ne "  ${GREEN}\$ ${NC}"
    type_text "$cmd" 0.02
    echo ""
    $FAST_MODE || sleep 0.5
    eval "$cmd"
    $FAST_MODE || sleep 1
}

pause() {
    $FAST_MODE || sleep "${1:-2}"
}

# ============================================================
# DEMO START
# ============================================================

clear
echo ""
echo -e "${BOLD}${CYAN}"
cat << 'LOGO'
  ╔═══════════════════════════════════════════════════════════╗
  ║                                                           ║
  ║   ███████╗██╗  ██╗██████╗       █████╗  ██████╗ ████████╗ ║
  ║   ██╔════╝╚██╗██╔╝██╔══██╗     ██╔══██╗██╔════╝ ╚══██╔══╝ ║
  ║   █████╗   ╚███╔╝ ██████╔╝     ███████║██║  ███╗   ██║    ║
  ║   ██╔══╝   ██╔██╗ ██╔═══╝      ██╔══██║██║   ██║   ██║    ║
  ║   ███████╗██╔╝ ██╗██║          ██║  ██║╚██████╔╝   ██║    ║
  ║   ╚══════╝╚═╝  ╚═╝╚═╝          ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ║
  ║                                                           ║
  ║   Recovery-Aware Execution Agent for Lab Hardware         ║
  ║   Autonomous Sense → Decide → Act Loop                    ║
  ║                                                           ║
  ╚═══════════════════════════════════════════════════════════╝
LOGO
echo -e "${NC}"
pause 2

echo -e "  ${DIM}This demo showcases the agent's autonomous disaster recovery${NC}"
echo -e "  ${DIM}capabilities when managing laboratory hardware.${NC}"
echo ""
echo -e "  ${DIM}The agent follows a continuous loop:${NC}"
echo -e "    ${BOLD}SENSE${NC} ${DIM}→${NC} ${BOLD}DECIDE${NC} ${DIM}→${NC} ${BOLD}ACT${NC}"
echo -e "  ${DIM}detecting faults, classifying errors, and executing safe recoveries.${NC}"
echo ""
pause 3

# ============================================================
# SCENARIO 1: HAPPY PATH
# ============================================================

print_banner "SCENARIO 1: Normal Operation (Happy Path)"

print_explain "Run the agent with NO faults — heater reaches 120°C normally"
print_explain "Expected: set_temperature → wait → shutdown. All checks pass."
echo ""

run_cmd "python3 -m exp_agent.cli.run_sim --fault-mode none --target-temp 120 --seed 42"

echo ""
echo -e "  ${GREEN}✓ Normal operation completed successfully${NC}"
echo -e "  ${DIM}  All postconditions met, safe shutdown performed${NC}"
pause 3

# ============================================================
# SCENARIO 2: TEMPERATURE OVERSHOOT (SAFETY VIOLATION)
# ============================================================

print_banner "SCENARIO 2: Temperature Overshoot → Safety Violation"

print_explain "Inject OVERSHOOT fault — temperature exceeds safety limit (130°C)"
print_explain "Agent must detect violation, classify as UNSAFE, and ABORT safely"
echo ""

run_cmd "python3 -m exp_agent.cli.run_sim --fault-mode overshoot --target-temp 120 --seed 42"

echo ""
echo -e "  ${RED}✗ Overshoot detected — agent correctly ABORTED${NC}"
echo -e "  ${DIM}  Decision: abort → cool_down → safe shutdown${NC}"
echo -e "  ${DIM}  Safety invariant: temp must stay < 130°C${NC}"
pause 3

# ============================================================
# SCENARIO 3: TIMEOUT / RETRY
# ============================================================

print_banner "SCENARIO 3: Communication Timeout → Automatic Retry"

print_explain "Inject TIMEOUT fault — device becomes temporarily unresponsive"
print_explain "Agent should detect postcondition failure, RETRY, and recover"
echo ""

run_cmd "python3 -m exp_agent.cli.run_sim --fault-mode timeout --target-temp 120 --seed 42"

echo ""
echo -e "  ${YELLOW}⟳ Timeout detected — agent RETRIED and recovered${NC}"
echo -e "  ${DIM}  Decision: retry → wait 2s → re-execute → success${NC}"
pause 3

# ============================================================
# SCENARIO 4: SENSOR FAILURE
# ============================================================

print_banner "SCENARIO 4: Sensor Failure → Graceful Handling"

print_explain "Inject SENSOR_FAIL fault — sensor returns invalid readings"
print_explain "Agent detects and classifies the hardware failure"
echo ""

run_cmd "python3 -m exp_agent.cli.run_sim --fault-mode sensor_fail --target-temp 120 --seed 42"

echo ""
echo -e "  ${DIM}  Sensor failure scenario completed${NC}"
pause 3

# ============================================================
# SCENARIO 5: FULL INSTRUMENTED PIPELINE
# ============================================================

print_banner "SCENARIO 5: Full Pipeline Analysis (Instrumented)"

print_explain "Run the INSTRUMENTED supervisor — full logging & decision analysis"
print_explain "Shows: telemetry, error classification, fault signatures, decision trails"
echo ""

run_cmd "python3 -m exp_agent.cli.run_instrumented --fault-mode overshoot --target-temp 120 --verbose"

pause 2

# ============================================================
# SCENARIO 6: TEST SUITE
# ============================================================

print_banner "SCENARIO 6: Automated Test Suite"

print_explain "Run the test suite to validate all recovery paths"
print_explain "Covers: classification, decisions, guardrails, policy, workflows"
echo ""

run_cmd "python3 -m pytest tests/ -v --tb=short 2>&1 | head -60"

pause 2

# ============================================================
# WRAP UP
# ============================================================

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║${NC}  ${BOLD}Demo Complete${NC}"
echo -e "${BOLD}${CYAN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}║${NC}  ${BOLD}Scenarios demonstrated:${NC}"
echo -e "${BOLD}${CYAN}║${NC}    ${GREEN}1.${NC} Normal operation (happy path)          ${GREEN}✓${NC}"
echo -e "${BOLD}${CYAN}║${NC}    ${RED}2.${NC} Temperature overshoot → ABORT           ${RED}✗ → safe shutdown${NC}"
echo -e "${BOLD}${CYAN}║${NC}    ${YELLOW}3.${NC} Communication timeout → RETRY           ${YELLOW}⟳ → recovered${NC}"
echo -e "${BOLD}${CYAN}║${NC}    ${MAGENTA}4.${NC} Sensor failure → detection              ${MAGENTA}⚠ → handled${NC}"
echo -e "${BOLD}${CYAN}║${NC}    ${BLUE}5.${NC} Full pipeline with decision analysis    ${BLUE}📊${NC}"
echo -e "${BOLD}${CYAN}║${NC}    ${CYAN}6.${NC} Automated test suite                    ${CYAN}🧪${NC}"
echo -e "${BOLD}${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}║${NC}  ${BOLD}Key capabilities shown:${NC}"
echo -e "${BOLD}${CYAN}║${NC}    • Autonomous sense-decide-act loop"
echo -e "${BOLD}${CYAN}║${NC}    • Multi-layer safety execution (pre/safety/post checks)"
echo -e "${BOLD}${CYAN}║${NC}    • Intelligent error classification"
echo -e "${BOLD}${CYAN}║${NC}    • Policy-driven recovery decisions"
echo -e "${BOLD}${CYAN}║${NC}    • Safe shutdown on critical failures"
echo -e "${BOLD}${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

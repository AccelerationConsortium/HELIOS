#!/usr/bin/env bash
# ============================================================
# Demo: Run all 29 tests for Exp-Agent
# Shows the agent's decision-making test coverage:
#   - Error classification & recovery policy
#   - Workflow execution with step cursor
#   - SKIP for optional steps
#   - DEGRADE with PlanPatch cascading
#   - Retry budget exhaustion
#   - Guardrails & safety checks
# ============================================================

set -e
cd "$(dirname "$0")/.."

export TERM=xterm-256color

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

clear

echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  Exp-Agent: Recovery-Aware Execution Agent${NC}"
echo -e "${BOLD}  Test Suite Demo — 29 Tests${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo -e "${CYAN}Test Coverage:${NC}"
echo "  1. Guardrails & Safety Checks"
echo "  2. Error Classification (overshoot, sensor_fail, timeout)"
echo "  3. Signature Analysis (drift, stall, oscillation, stable)"
echo "  4. Recovery Decisions (retry, skip, abort, degrade)"
echo "  5. Policy Integration (full escalation sequence)"
echo "  6. Workflow Execution (step cursor, stage tracking)"
echo "  7. SKIP Decision (optional step failure → skip)"
echo "  8. DEGRADE Cascade (PlanPatch → downstream updates)"
echo "  9. Retry Budget (exhaustion → on_failure fallback)"
echo ""
sleep 2

# ============================================================
# Part 1: Core Policy & Recovery Tests (21 tests)
# ============================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  PART 1: Core Policy & Recovery Tests${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
sleep 1

echo -e "${CYAN}[1/3] Running Guardrails Tests...${NC}"
.venv/bin/python -m pytest tests/test_guardrails.py -v --tb=short 2>&1
echo ""
sleep 1

echo -e "${CYAN}[2/3] Running Recovery Policy Tests...${NC}"
.venv/bin/python -m pytest tests/test_policy.py -v --tb=short 2>&1
echo ""
sleep 1

echo -e "${CYAN}[3/3] Running Recovery Agent Tests...${NC}"
.venv/bin/python -m pytest tests/test_recovery.py -v --tb=short 2>&1
echo ""
sleep 1

# ============================================================
# Part 2: Workflow Supervisor Tests (8 tests)
# ============================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  PART 2: Workflow Supervisor Tests${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
sleep 1

echo -e "${CYAN}Running Workflow Supervisor Tests (verbose with output)...${NC}"
.venv/bin/python -m pytest tests/test_workflow_supervisor.py -v -s --tb=short 2>&1
echo ""
sleep 1

# ============================================================
# Part 3: Full Suite Summary
# ============================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  PART 3: Full Suite Summary${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
sleep 1

.venv/bin/python -m pytest tests/ -v --tb=short 2>&1
echo ""

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  All 29 tests passed!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "${BOLD}Decision types tested:${NC}"
echo "  RETRY   — timeout with backoff, retry budget exhaustion"
echo "  SKIP    — optional step failure → skip, continue workflow"
echo "  DEGRADE — overshoot with drift → lower target, patch downstream"
echo "  ABORT   — sensor failure, safety violation → safe shutdown"
echo ""
echo -e "${BOLD}Workflow features tested:${NC}"
echo "  step_id / stage cursor tracking"
echo "  PlanPatch cascading (overrides + relaxations)"
echo "  Criticality semantics (critical vs optional)"
echo "  on_failure fallback (abort / skip)"
echo ""
sleep 3

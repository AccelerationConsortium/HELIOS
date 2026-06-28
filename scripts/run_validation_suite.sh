#!/usr/bin/env bash
set -euo pipefail

pytest \
  tests/test_candidate_memory.py \
  tests/test_failure_zone_memory.py \
  tests/test_offline_closed_loop_sdl.py \
  tests/test_offline_scenario_benchmarks.py \
  tests/test_policy_evolution.py \
  tests/test_policy_evolution_workflow_e2e.py \
  tests/test_learned_policy.py \
  tests/test_system_validation_report.py \
  tests/test_backend_selection.py

ruff check \
  app/optimization/candidate_memory.py \
  app/optimization/failure_zone_memory.py \
  app/services/campaign_state.py \
  app/services/system_validation_report.py \
  tests/test_candidate_memory.py \
  tests/test_failure_zone_memory.py \
  tests/test_system_validation_report.py \
  tests/test_offline_closed_loop_sdl.py \
  tests/test_offline_scenario_benchmarks.py \
  tests/test_policy_evolution_workflow_e2e.py

git diff --check

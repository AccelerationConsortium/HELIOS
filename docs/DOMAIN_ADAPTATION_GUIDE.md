# OTbot Domain Adaptation Guide

How to adapt OTbot for different experiment types (OER, organic synthesis, etc.) while using the same OT-2 hardware platform.

## Architecture Overview

OTbot's framework is **95% domain-agnostic**. All domain-specific knowledge lives in 4 extensible registries:

```
┌──────────────────────────────────────────────────┐
│              Domain-Agnostic Core                │
│                                                  │
│  Scheduler → Worker → Compiler → Safety Gate     │
│  Dispatcher → Adapter → Memory → Provenance      │
│  Campaign Loop → Convergence → Evolution Engine  │
│  Metrics Store → Reviewer → Candidate Gen        │
│  Failure Learning → Batch System → Templates     │
│                                                  │
├──────────────────────────────────────────────────┤
│           4 Extensible Registries                │
│                                                  │
│  1. Skill Files     → agent/skills/*.md          │
│  2. KPI Extractors  → app/services/metrics.py    │
│  3. Failure Rules   → app/services/failure_sigs  │
│  4. Protocol Patterns → protocols + templates    │
│                                                  │
└──────────────────────────────────────────────────┘
```

**Zero core code changes required** to add a new experiment domain. You only extend registries.

---

## Registry 1: Skill Files (`agent/skills/*.md`)

Skill files define the primitives (actions) available for each instrument. The primitives registry (`app/services/primitives_registry.py`) auto-discovers these at startup.

### Existing Skills

| File | Instrument | Primitives |
|------|-----------|-----------|
| `robot.md` | OT-2 Robot | `robot.home`, `robot.aspirate`, `robot.dispense`, etc. (11 total) |
| `plc.md` | PLC (Modbus) | `plc.dispense_ml`, `plc.set_pump_on_timer`, `plc.set_ultrasonic_on_timer` |
| `relay.md` | USB Relay | `relay.set_channel`, `relay.turn_on/off`, `relay.switch_to` |
| `squidstat.md` | Potentiostat | `squidstat.run_experiment`, `squidstat.get_data`, `squidstat.save_snapshot`, `squidstat.reset_plot` |
| `utility.md` | Virtual | `wait`, `log` |

### Adding a New Skill File

For a new instrument (e.g., UV-Vis spectrometer), create `agent/skills/uvvis.md`:

```markdown
---
name: uvvis-spectrometer
instrument: uvvis
resource_id: uvvis
version: 1.0.0
---

# UV-Vis Spectrometer

## Primitives

### uvvis.measure_absorbance

Measure absorbance spectrum at specified wavelength range.

- error_class: MEDIUM
- safety_class: INFORMATIONAL
- timeout: 30
- retries: 1

**Parameters**:
- wavelength_start_nm: Start wavelength (float, required)
- wavelength_end_nm: End wavelength (float, required)
- integration_time_ms: Integration time (int, default: 100)

### uvvis.measure_single

Single-wavelength absorbance measurement.

- error_class: BYPASS
- safety_class: INFORMATIONAL
- timeout: 10
- retries: 2

**Parameters**:
- wavelength_nm: Target wavelength (float, required)
```

Then add a corresponding dispatcher handler in `app/hardware/dispatcher.py`:

```python
# In ActionDispatcher.__init__(), add to self._handlers:
"uvvis.measure_absorbance": self._handle_uvvis_measure,
"uvvis.measure_single":     self._handle_uvvis_single,

# Handler implementation:
def _handle_uvvis_measure(self, params: dict) -> dict:
    result = self._uvvis.measure_spectrum(
        start=params["wavelength_start_nm"],
        end=params["wavelength_end_nm"],
        integration=params.get("integration_time_ms", 100),
    )
    return {"absorbance_data": result, "status": "ok"}
```

---

## Registry 2: KPI Extractors (`app/services/metrics.py`)

KPIs are declaratively defined and auto-extracted from step results after each run.

### Existing KPI Definitions (V1)

| KPI Name | Unit | Scope | Primitive | Domain |
|----------|------|-------|-----------|--------|
| `volume_accuracy_pct` | pct | step | `aspirate` | General |
| `temp_accuracy_c` | celsius | step | `heat` | General |
| `impedance_ohm` | ohm | step | `eis` | Electrochemistry |
| `step_duration_s` | seconds | step | all | General |
| `overpotential_mv` | mV | step | `squidstat.run_experiment` | Electrochemistry |
| `current_density_ma_cm2` | mA/cm2 | step | `squidstat.run_experiment` | Electrochemistry |
| `coulombic_efficiency` | ratio | step | `squidstat.run_experiment` | Electrochemistry |
| `stability_decay_pct` | pct | step | `squidstat.run_experiment` | Electrochemistry |
| `charge_passed_c` | C | step | `squidstat.run_experiment` | Electrochemistry |

### Adding KPIs for a New Domain

Add entries to `KPI_DEFINITIONS_V1` in `app/services/metrics.py` and register the extractor function:

```python
# Example: OER-specific KPIs
KpiDefinition(
    name="faradaic_efficiency",
    unit="pct",
    scope="step",
    primitive="squidstat.run_experiment",
    extractor="extract_faradaic_efficiency",
),
KpiDefinition(
    name="onset_potential_v",
    unit="V",
    scope="step",
    primitive="squidstat.run_experiment",
    extractor="extract_onset_potential",
),
KpiDefinition(
    name="tafel_slope_mv_dec",
    unit="mV/dec",
    scope="step",
    primitive="squidstat.run_experiment",
    extractor="extract_tafel_slope",
),
KpiDefinition(
    name="o2_production_rate",
    unit="umol/min",
    scope="step",
    primitive="squidstat.run_experiment",
    extractor="extract_o2_rate",
),

# Example: UV-Vis domain KPIs
KpiDefinition(
    name="peak_absorbance",
    unit="AU",
    scope="step",
    primitive="uvvis.measure_absorbance",
    extractor="extract_peak_absorbance",
),
KpiDefinition(
    name="lambda_max_nm",
    unit="nm",
    scope="step",
    primitive="uvvis.measure_absorbance",
    extractor="extract_lambda_max",
),
```

Then add the extractor functions to the `_EXTRACTORS` dict:

```python
def _extract_faradaic_efficiency(step_result: dict, params: dict) -> float | None:
    """Extract faradaic efficiency from electrochemistry result."""
    return step_result.get("faradaic_efficiency_pct")

def _extract_onset_potential(step_result: dict, params: dict) -> float | None:
    """Extract onset potential from LSV/CV data."""
    return step_result.get("onset_potential_v")

# Register in _EXTRACTORS dict:
_EXTRACTORS["extract_faradaic_efficiency"] = _extract_faradaic_efficiency
_EXTRACTORS["extract_onset_potential"] = _extract_onset_potential
```

---

## Registry 3: Failure Rules (`app/services/failure_signatures.py`)

Failure signatures classify step errors into machine-readable types with remediation recommendations.

### Existing Failure Types (17)

```
volume_delivery_failure, temperature_deviation, temperature_overshoot,
impedance_anomaly, electrode_degradation, electrolyte_contamination,
tip_shortage, liquid_insufficient, deck_conflict, instrument_disconnection,
instrument_timeout, sensor_drift, file_missing, protocol_sequence_error,
safety_limit_exceeded, unknown
```

### Existing Likely Causes (16)

```
tip_clog, tip_missing, insufficient_liquid, thermal_runaway,
heater_malfunction, electrode_fouling, bubble_formation,
contaminated_solution, connection_lost, power_interruption,
sensor_calibration_drift, file_system_error, parameter_out_of_range,
hardware_limit, software_error, unknown
```

### Adding Domain-Specific Failure Rules

**Step 1**: Add new failure types and causes to the constants:

```python
# In FAILURE_TYPES, add:
"catalyst_degradation",
"membrane_fouling",
"gas_bubble_blockage",
"ph_drift",
"reference_electrode_drift",

# In LIKELY_CAUSES, add:
"catalyst_poisoning",
"membrane_damage",
"gas_channel_blocked",
"electrolyte_decomposition",
"reference_junction_fouled",
```

**Step 2**: Add regex classification rules to `_CLASSIFICATION_RULES`:

```python
# OER-specific classification rules
ClassificationRule(
    pattern=re.compile(r"catalyst\s+(degrad|poison|deactiv)", re.I),
    failure_type="catalyst_degradation",
    likely_cause="catalyst_poisoning",
    severity="HIGH",
    retryable=False,
),
ClassificationRule(
    pattern=re.compile(r"membrane\s+(foul|block|damag)", re.I),
    failure_type="membrane_fouling",
    likely_cause="membrane_damage",
    severity="HIGH",
    retryable=False,
),
ClassificationRule(
    pattern=re.compile(r"(gas|bubble)\s+(block|stuck|accumul)", re.I),
    failure_type="gas_bubble_blockage",
    likely_cause="gas_channel_blocked",
    severity="MEDIUM",
    retryable=True,
),
```

**Step 3**: Add remediation patches to `_PATCH_LIBRARY`:

```python
("catalyst_degradation", "catalyst_poisoning"): RecommendedPatch(
    action="replace_electrode",
    params={"reason": "catalyst_poisoned"},
    description="Replace the degraded catalyst electrode",
),
("gas_bubble_blockage", "gas_channel_blocked"): RecommendedPatch(
    action="flush_channel",
    params={"flush_volume_ml": 5.0, "flush_cycles": 3},
    description="Flush gas channel with clean electrolyte to remove bubbles",
),
```

---

## Registry 4: Protocol Patterns

Protocol patterns define the experiment workflow templates. These are JSON structures submitted via API or stored as templates.

### Existing Protocol Patterns

| Pattern | Domain | Description |
|---------|--------|-------------|
| `OER_SCREENING` | Electrochemistry | Full OER catalyst screening with EIS |
| Battery charge/discharge | Battery | Charge-discharge cycling with impedance |

### Creating a New Protocol Pattern

**Example: Organic Synthesis Protocol**

```json
{
  "name": "ORGANIC_SYNTHESIS_SCREEN",
  "version": "1.0",
  "steps": [
    {
      "key": "home",
      "primitive": "robot.home",
      "params": {}
    },
    {
      "key": "load_reagents",
      "primitive": "robot.load_labware",
      "params": {"slot": "1", "labware": "opentrons_24_tuberack_2ml"}
    },
    {
      "key": "load_plate",
      "primitive": "robot.load_labware",
      "params": {"slot": "2", "labware": "corning_96_wellplate_360ul_flat"}
    },
    {
      "key": "transfer_reagent_a",
      "primitive": "robot.aspirate",
      "params": {"pipette": "left", "volume_ul": 50, "location": "1:A1"},
      "depends_on": ["load_reagents"]
    },
    {
      "key": "dispense_reagent_a",
      "primitive": "robot.dispense",
      "params": {"pipette": "left", "volume_ul": 50, "location": "2:A1"},
      "depends_on": ["transfer_reagent_a"]
    },
    {
      "key": "heat_reaction",
      "primitive": "plc.set_pump_on_timer",
      "params": {"seconds": 120},
      "depends_on": ["dispense_reagent_a"]
    },
    {
      "key": "measure_product",
      "primitive": "uvvis.measure_absorbance",
      "params": {"wavelength_start_nm": 200, "wavelength_end_nm": 800},
      "depends_on": ["heat_reaction"]
    }
  ]
}
```

### Template Versioning (C5 Evolution)

Templates are automatically versioned by the evolution engine:

```
template_v1 → run → review → evolution proposal → template_v2
                                                      ↓
                                          parent_template_id = v1
```

Each new domain's protocols benefit from the same evolution pipeline — no code changes needed.

---

## Complete Example: Adding OER Domain

OER (Oxygen Evolution Reaction) is **already partially built-in**. The existing `squidstat.run_experiment` primitive and 5 electrochemistry KPIs work for OER. To fully specialize:

### What's Already Working

- All robot primitives (OT-2 is domain-agnostic)
- All relay/PLC primitives
- `squidstat.run_experiment` for electrochemistry
- `impedance_ohm`, `overpotential_mv`, `current_density_ma_cm2`, `coulombic_efficiency`, `stability_decay_pct` KPIs
- Failure types: `impedance_anomaly`, `electrode_degradation`, `electrolyte_contamination`

### What to Add (~50 lines)

1. **KPIs** (in `metrics.py`, ~15 lines):
   ```python
   KpiDefinition(name="faradaic_efficiency", unit="pct", scope="step",
                 primitive="squidstat.run_experiment", extractor="extract_faradaic_efficiency"),
   KpiDefinition(name="onset_potential_v", unit="V", scope="step",
                 primitive="squidstat.run_experiment", extractor="extract_onset_potential"),
   KpiDefinition(name="tafel_slope_mv_dec", unit="mV/dec", scope="step",
                 primitive="squidstat.run_experiment", extractor="extract_tafel_slope"),
   ```

2. **Extractor functions** (~15 lines):
   ```python
   def _extract_faradaic_efficiency(step_result: dict, params: dict) -> float | None:
       return step_result.get("faradaic_efficiency_pct")

   def _extract_onset_potential(step_result: dict, params: dict) -> float | None:
       return step_result.get("onset_potential_v")

   def _extract_tafel_slope(step_result: dict, params: dict) -> float | None:
       return step_result.get("tafel_slope_mv_dec")
   ```

3. **Failure rules** (~10 lines):
   ```python
   ClassificationRule(
       pattern=re.compile(r"catalyst\s+(degrad|poison)", re.I),
       failure_type="catalyst_degradation",
       likely_cause="catalyst_poisoning",
       severity="HIGH",
       retryable=False,
   ),
   ```

4. **Protocol template** (~10 lines JSON): Submit via API as a campaign protocol.

**Total: ~50 lines, 0 core code changes.**

---

## Adding a Completely New Domain: Organic Synthesis

For a domain that needs new instruments:

### Step-by-Step Checklist

| # | Task | File(s) | Lines |
|---|------|---------|-------|
| 1 | Create skill file for new instruments | `agent/skills/uvvis.md` | ~40 |
| 2 | Add dispatcher handlers | `app/hardware/dispatcher.py` | ~30 |
| 3 | Add KPI definitions | `app/services/metrics.py` | ~20 |
| 4 | Add KPI extractor functions | `app/services/metrics.py` | ~20 |
| 5 | Add failure classification rules | `app/services/failure_signatures.py` | ~20 |
| 6 | Add remediation patches | `app/services/failure_signatures.py` | ~10 |
| 7 | Create protocol template | Via API or JSON file | ~30 |
| 8 | Write tests for new extractors | `tests/test_metrics.py` | ~30 |
| **Total** | | | **~200** |

### What NOT to Change

These files are domain-agnostic and should **never** need modification for a new domain:

| Module | Why It's Domain-Agnostic |
|--------|------------------------|
| `app/services/campaign_loop.py` | Operates on abstract `objective_kpi` + direction |
| `app/services/convergence.py` | Pure math on KPI time-series |
| `app/services/candidate_gen.py` | Generic parameter space sampling |
| `app/services/evolution.py` | Generic prior tightening + template versioning |
| `app/services/reviewer.py` | LLM-driven, adapts to any domain via prompt |
| `app/services/run_service.py` | Generic run lifecycle management |
| `app/services/memory.py` | Generic episodic/semantic/procedural memory |
| `app/services/scheduler.py` | Generic time-based scheduling |
| `app/services/safety.py` | Uses safety classes from skill files |
| `app/core/db.py` | Schema is domain-agnostic |
| `app/worker.py` | Generic step execution loop |

---

## Campaign Goal Configuration by Domain

The campaign loop accepts any KPI as its optimization objective:

```python
# Battery: minimize impedance
CampaignGoal(objective_kpi="impedance_ohm", direction="minimize", target_value=50.0)

# OER: maximize faradaic efficiency
CampaignGoal(objective_kpi="faradaic_efficiency", direction="maximize", target_value=95.0)

# OER: minimize overpotential
CampaignGoal(objective_kpi="overpotential_mv", direction="minimize", target_value=300.0)

# Organic synthesis: maximize yield (custom KPI)
CampaignGoal(objective_kpi="reaction_yield_pct", direction="maximize", target_value=90.0)

# UV-Vis: maximize peak absorbance
CampaignGoal(objective_kpi="peak_absorbance", direction="maximize", target_value=2.0)
```

## Sample Preparation Customization

The `sample.prepare_from_csv` handler in `app/hardware/dispatcher.py` currently has hardcoded column names for battery chemistry:

```python
# Current (battery-specific):
chemicals = ["Zn", "TMAC", "TMAB", "DTAB", "MTAB", "CTAC", "CTAB", "DODAB"]
```

**To adapt**: Make this configurable via the protocol step params:

```json
{
  "key": "prepare_samples",
  "primitive": "sample.prepare_from_csv",
  "params": {
    "csv_path": "recipes/oer_catalysts.csv",
    "chemical_columns": ["NiSO4", "FeSO4", "CoSO4", "Na2MoO4"],
    "solvent_column": "KOH_concentration_M"
  }
}
```

This is the **only** place in the codebase where domain-specific chemical names are hardcoded.

---

## Testing New Domains

### Unit Tests

Add tests for new KPI extractors and failure rules:

```python
# tests/test_new_domain_kpis.py
def test_extract_faradaic_efficiency():
    result = {"faradaic_efficiency_pct": 92.5}
    value = _extract_faradaic_efficiency(result, {})
    assert value == 92.5

def test_extract_faradaic_efficiency_missing():
    result = {"other_field": 42}
    value = _extract_faradaic_efficiency(result, {})
    assert value is None
```

### Integration Tests

Use the offline benchmark framework (`benchmarks/`) with SimAdapter:

```python
# Configure SimAdapter for new domain
sim_world = SimWorld(seed=42, ...)
sim_adapter = SimAdapter(
    world=sim_world,
    noise_pct=0.02,
    # Custom primitive handlers for new domain
    custom_handlers={
        "uvvis.measure_absorbance": lambda params: {
            "peak_absorbance": 1.5 + random.gauss(0, 0.05),
            "lambda_max_nm": 450 + random.gauss(0, 2),
        }
    },
)
```

### Campaign Test

Run an offline campaign to validate the full loop:

```python
from app.services.campaign_loop import run_campaign_offline, CampaignGoal

goal = CampaignGoal(
    objective_kpi="faradaic_efficiency",
    direction="maximize",
    target_value=95.0,
    max_rounds=5,
    batch_size=3,
    strategy="prior_guided",
)

result = run_campaign_offline(goal, space, sim_fn)
assert result.best_kpi > 80.0  # reasonable improvement from baseline
```

//! exp_agent_core - High-performance Rust core for exp-agent recovery system.
//!
//! This crate provides:
//! - Core types matching Python Pydantic models
//! - Device driver traits and simulated implementations
//! - Telemetry buffer for high-frequency data
//! - PyO3 bindings for Python integration

pub mod device;
pub mod telemetry;
pub mod types;

use pyo3::prelude::*;

/// Register all Python classes and functions.
#[pymodule]
fn exp_agent_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register type enums
    m.add_class::<types::Effect>()?;
    m.add_class::<types::DecisionType>()?;
    m.add_class::<types::Severity>()?;
    m.add_class::<types::Criticality>()?;
    m.add_class::<types::OnFailure>()?;
    m.add_class::<types::DeviceStatus>()?;
    m.add_class::<types::ErrorType>()?;
    m.add_class::<types::SignatureMode>()?;
    m.add_class::<types::SampleStatus>()?;

    // Register data types
    m.add_class::<types::DeviceState>()?;
    m.add_class::<types::HardwareError>()?;
    m.add_class::<types::Action>()?;
    m.add_class::<types::Decision>()?;
    m.add_class::<types::ErrorProfile>()?;
    m.add_class::<types::SignatureResult>()?;
    m.add_class::<types::RecoveryDecision>()?;
    m.add_class::<types::PlanStep>()?;
    m.add_class::<types::PlanPatch>()?;

    // Register telemetry
    m.add_class::<telemetry::TelemetryBuffer>()?;
    m.add_class::<telemetry::TelemetryPoint>()?;
    m.add_class::<telemetry::TelemetryStats>()?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

// Note: Tests are in individual module files.
// Integration tests with Python require `maturin develop` first.
// Run `cargo test --no-default-features` for Rust-only tests.

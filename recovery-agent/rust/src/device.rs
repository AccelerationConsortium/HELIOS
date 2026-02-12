//! Device driver traits and implementations.
//!
//! This module defines the core trait for device drivers and provides
//! both real and simulated implementations.

use crate::types::{Action, DeviceState, DeviceStatus, ErrorType, HardwareError, Severity};
use async_trait::async_trait;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Result type for device operations.
pub type DeviceResult<T> = Result<T, HardwareError>;

/// Core trait for all device drivers.
#[async_trait]
pub trait Device: Send + Sync {
    /// Get the device name.
    fn name(&self) -> &str;

    /// Read the current device state.
    async fn read_state(&self) -> DeviceResult<DeviceState>;

    /// Execute an action on the device.
    async fn execute(&self, action: &Action) -> DeviceResult<()>;

    /// Check if the device is healthy.
    async fn health(&self) -> bool;

    /// Advance simulation time (for simulated devices).
    async fn tick(&self, dt: f64);
}

// ============================================================================
// Simulated Heater Device
// ============================================================================

/// Fault mode for simulated heater.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaterFaultMode {
    None,
    Random,
    Timeout,
    Overshoot,
    SensorFail,
}

/// Internal state for simulated heater.
struct HeaterInternalState {
    current_temp: f64,
    target_temp: f64,
    heating: bool,
    status: DeviceStatus,
    tick_count: u32,
    last_update_time: std::time::Instant,
    heating_start_time: Option<std::time::Instant>,
}

/// Simulated heater device with fault injection capabilities.
#[pyclass]
pub struct SimHeater {
    name: String,
    fault_mode: HeaterFaultMode,
    max_safe_temp: f64,
    state: Arc<RwLock<HeaterInternalState>>,
}

impl SimHeater {
    pub fn new(name: String, fault_mode: HeaterFaultMode) -> Self {
        Self {
            name,
            fault_mode,
            max_safe_temp: 130.0,
            state: Arc::new(RwLock::new(HeaterInternalState {
                current_temp: 25.0,
                target_temp: 25.0,
                heating: false,
                status: DeviceStatus::Idle,
                tick_count: 0,
                last_update_time: std::time::Instant::now(),
                heating_start_time: None,
            })),
        }
    }

    async fn update_physics(&self) {
        let mut state = self.state.write().await;
        let now = std::time::Instant::now();
        let dt = now.duration_since(state.last_update_time).as_secs_f64() * 2.0; // 2x speed
        state.last_update_time = now;

        if state.heating {
            if state.heating_start_time.is_none() {
                state.heating_start_time = Some(now);
            }

            let diff = state.target_temp - state.current_temp;
            let rate = 10.0; // degrees per second

            if diff > 0.0 {
                let change = (rate * dt).min(diff);
                state.current_temp += change;
            } else if diff < 0.0 {
                let change = (-rate * dt).max(diff);
                state.current_temp += change;
            }

            // Fault injection based on duration
            if let Some(start) = state.heating_start_time {
                let duration = now.duration_since(start).as_secs_f64();

                match self.fault_mode {
                    HeaterFaultMode::Overshoot if duration > 2.0 => {
                        state.current_temp += 20.0 * dt;
                    }
                    HeaterFaultMode::Timeout if duration > 2.0 => {
                        // Drift back, undo progress
                        let change = (rate * dt).min(diff.abs());
                        state.current_temp -= change * 0.9;
                    }
                    _ => {}
                }
            }
        } else {
            state.heating_start_time = None;
            // Cool down to ambient
            let diff = 25.0 - state.current_temp;
            let rate = 2.0;
            if diff.abs() > 0.1 {
                let direction = if diff > 0.0 { 1.0 } else { -1.0 };
                state.current_temp += direction * rate * dt;
            }
        }
    }
}

#[async_trait]
impl Device for SimHeater {
    fn name(&self) -> &str {
        &self.name
    }

    async fn read_state(&self) -> DeviceResult<DeviceState> {
        self.update_physics().await;

        let state = self.state.read().await;

        // Sensor failure simulation
        if self.fault_mode == HeaterFaultMode::SensorFail && state.tick_count > 5 {
            let mut ctx = HashMap::new();
            ctx.insert(
                "status".to_string(),
                serde_json::Value::String(format!("{:?}", state.status)),
            );
            ctx.insert(
                "tick".to_string(),
                serde_json::Value::Number(state.tick_count.into()),
            );
            return Err(HardwareError::with_context(
                self.name.clone(),
                ErrorType::SensorFail,
                Severity::High,
                "Temperature sensor reading failed (got -999)".to_string(),
                chrono::Utc::now().to_rfc3339(),
                None,
                ctx,
            ));
        }

        let mut telemetry = HashMap::new();
        telemetry.insert(
            "temperature".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(state.current_temp).unwrap_or(serde_json::Number::from(0)),
            ),
        );
        telemetry.insert(
            "target".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(state.target_temp).unwrap_or(serde_json::Number::from(0)),
            ),
        );
        telemetry.insert("heating".to_string(), serde_json::Value::Bool(state.heating));

        // Check for overshoot (update status but don't error - let policy decide)
        let mut status = state.status;
        if state.current_temp > self.max_safe_temp {
            status = DeviceStatus::Error;
        }

        Ok(DeviceState::with_telemetry(self.name.clone(), status, telemetry))
    }

    async fn execute(&self, action: &Action) -> DeviceResult<()> {
        self.update_physics().await;

        let mut state = self.state.write().await;
        let params = action.get_params();

        match action.name.as_str() {
            "set_temperature" => {
                let target = params
                    .get("temperature")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(25.0);
                state.target_temp = target;
                state.heating = true;
                state.status = DeviceStatus::Running;
                tracing::info!("[{}] Setting target to {}", self.name, target);
            }
            "wait" => {
                let duration = params
                    .get("duration")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                tracing::info!("[{}] Waiting for {} seconds", self.name, duration);
            }
            "cool_down" => {
                state.target_temp = 25.0;
                state.heating = false;
                state.status = DeviceStatus::Idle;
                tracing::info!("[{}] Cooling down to 25.0", self.name);
            }
            _ => {
                tracing::warn!("[{}] Unknown action: {}", self.name, action.name);
            }
        }

        Ok(())
    }

    async fn health(&self) -> bool {
        true
    }

    async fn tick(&self, _dt: f64) {
        let mut state = self.state.write().await;
        state.tick_count += 1;
        let temp = state.current_temp;
        let target = state.target_temp;
        let tick = state.tick_count;
        drop(state);

        tracing::debug!(
            "[SimHeater] Tick={} T={:.1} (Target={:.1})",
            tick,
            temp,
            target
        );
    }
}

// ============================================================================
// Simulated Pump Device
// ============================================================================

/// Fault mode for simulated pump.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PumpFaultMode {
    None,
    FlowBlocked,
    MotorStall,
}

/// Internal state for simulated pump.
struct PumpInternalState {
    flow_rate: f64,
    target_flow: f64,
    running: bool,
    status: DeviceStatus,
    tick_count: u32,
}

/// Simulated pump device.
#[pyclass]
pub struct SimPump {
    name: String,
    fault_mode: PumpFaultMode,
    max_flow: f64,
    state: Arc<RwLock<PumpInternalState>>,
}

impl SimPump {
    pub fn new(name: String, fault_mode: PumpFaultMode) -> Self {
        Self {
            name,
            fault_mode,
            max_flow: 100.0,
            state: Arc::new(RwLock::new(PumpInternalState {
                flow_rate: 0.0,
                target_flow: 0.0,
                running: false,
                status: DeviceStatus::Idle,
                tick_count: 0,
            })),
        }
    }
}

#[async_trait]
impl Device for SimPump {
    fn name(&self) -> &str {
        &self.name
    }

    async fn read_state(&self) -> DeviceResult<DeviceState> {
        let state = self.state.read().await;

        // Flow blocked simulation
        if self.fault_mode == PumpFaultMode::FlowBlocked && state.running && state.tick_count > 3 {
            return Err(HardwareError::with_context(
                self.name.clone(),
                ErrorType::FlowBlocked,
                Severity::Medium,
                "Flow rate dropped to zero - possible blockage".to_string(),
                chrono::Utc::now().to_rfc3339(),
                None,
                HashMap::new(),
            ));
        }

        let mut telemetry = HashMap::new();
        telemetry.insert(
            "flow_rate".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(state.flow_rate).unwrap_or(serde_json::Number::from(0)),
            ),
        );
        telemetry.insert(
            "target_flow".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(state.target_flow).unwrap_or(serde_json::Number::from(0)),
            ),
        );
        telemetry.insert("running".to_string(), serde_json::Value::Bool(state.running));

        Ok(DeviceState::with_telemetry(self.name.clone(), state.status, telemetry))
    }

    async fn execute(&self, action: &Action) -> DeviceResult<()> {
        let mut state = self.state.write().await;
        let params = action.get_params();

        match action.name.as_str() {
            "set_flow" => {
                let target = params
                    .get("flow_rate")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
                    .min(self.max_flow);
                state.target_flow = target;
                state.flow_rate = target;
                state.running = target > 0.0;
                state.status = if state.running {
                    DeviceStatus::Running
                } else {
                    DeviceStatus::Idle
                };
                tracing::info!("[{}] Setting flow to {}", self.name, target);
            }
            "stop" => {
                state.target_flow = 0.0;
                state.flow_rate = 0.0;
                state.running = false;
                state.status = DeviceStatus::Idle;
                tracing::info!("[{}] Stopped pump", self.name);
            }
            _ => {
                tracing::warn!("[{}] Unknown action: {}", self.name, action.name);
            }
        }

        Ok(())
    }

    async fn health(&self) -> bool {
        true
    }

    async fn tick(&self, _dt: f64) {
        let mut state = self.state.write().await;
        state.tick_count += 1;
    }
}

// ============================================================================
// Simulated Positioner Device
// ============================================================================

/// Fault mode for simulated positioner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionerFaultMode {
    None,
    Collision,
    EncoderError,
}

/// Internal state for simulated positioner.
struct PositionerInternalState {
    position: (f64, f64, f64),
    target: (f64, f64, f64),
    moving: bool,
    status: DeviceStatus,
    tick_count: u32,
}

/// Simulated positioner (XYZ stage).
#[pyclass]
pub struct SimPositioner {
    name: String,
    fault_mode: PositionerFaultMode,
    bounds: ((f64, f64), (f64, f64), (f64, f64)),
    state: Arc<RwLock<PositionerInternalState>>,
}

impl SimPositioner {
    pub fn new(name: String, fault_mode: PositionerFaultMode) -> Self {
        Self {
            name,
            fault_mode,
            bounds: ((0.0, 100.0), (0.0, 100.0), (0.0, 50.0)),
            state: Arc::new(RwLock::new(PositionerInternalState {
                position: (0.0, 0.0, 0.0),
                target: (0.0, 0.0, 0.0),
                moving: false,
                status: DeviceStatus::Idle,
                tick_count: 0,
            })),
        }
    }
}

#[async_trait]
impl Device for SimPositioner {
    fn name(&self) -> &str {
        &self.name
    }

    async fn read_state(&self) -> DeviceResult<DeviceState> {
        let state = self.state.read().await;

        // Collision simulation
        if self.fault_mode == PositionerFaultMode::Collision && state.moving && state.tick_count > 2 {
            let mut ctx = HashMap::new();
            ctx.insert(
                "position".to_string(),
                serde_json::json!([state.position.0, state.position.1, state.position.2]),
            );
            return Err(HardwareError::with_context(
                self.name.clone(),
                ErrorType::Collision,
                Severity::High,
                "Collision detected during movement".to_string(),
                chrono::Utc::now().to_rfc3339(),
                None,
                ctx,
            ));
        }

        let mut telemetry = HashMap::new();
        telemetry.insert(
            "x".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(state.position.0).unwrap_or(serde_json::Number::from(0)),
            ),
        );
        telemetry.insert(
            "y".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(state.position.1).unwrap_or(serde_json::Number::from(0)),
            ),
        );
        telemetry.insert(
            "z".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(state.position.2).unwrap_or(serde_json::Number::from(0)),
            ),
        );
        telemetry.insert("moving".to_string(), serde_json::Value::Bool(state.moving));

        Ok(DeviceState::with_telemetry(self.name.clone(), state.status, telemetry))
    }

    async fn execute(&self, action: &Action) -> DeviceResult<()> {
        let mut state = self.state.write().await;
        let params = action.get_params();

        match action.name.as_str() {
            "move_to" => {
                let x = params.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = params.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let z = params.get("z").and_then(|v| v.as_f64()).unwrap_or(0.0);

                // Clamp to bounds
                let x = x.clamp(self.bounds.0 .0, self.bounds.0 .1);
                let y = y.clamp(self.bounds.1 .0, self.bounds.1 .1);
                let z = z.clamp(self.bounds.2 .0, self.bounds.2 .1);

                state.target = (x, y, z);
                state.position = (x, y, z); // Instant move for simulation
                state.moving = true;
                state.status = DeviceStatus::Running;
                tracing::info!("[{}] Moving to ({}, {}, {})", self.name, x, y, z);

                // Movement complete
                state.moving = false;
                state.status = DeviceStatus::Idle;
            }
            "home" => {
                state.target = (0.0, 0.0, 0.0);
                state.position = (0.0, 0.0, 0.0);
                state.moving = false;
                state.status = DeviceStatus::Idle;
                tracing::info!("[{}] Homing complete", self.name);
            }
            _ => {
                tracing::warn!("[{}] Unknown action: {}", self.name, action.name);
            }
        }

        Ok(())
    }

    async fn health(&self) -> bool {
        true
    }

    async fn tick(&self, _dt: f64) {
        let mut state = self.state.write().await;
        state.tick_count += 1;
    }
}

// ============================================================================
// Simulated Spectrometer Device
// ============================================================================

/// Fault mode for simulated spectrometer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrometerFaultMode {
    None,
    SignalSaturated,
    Timeout,
}

/// Internal state for simulated spectrometer.
struct SpectrometerInternalState {
    integration_time: f64,
    acquiring: bool,
    status: DeviceStatus,
    tick_count: u32,
    last_spectrum: Vec<f64>,
}

/// Simulated spectrometer device.
#[pyclass]
pub struct SimSpectrometer {
    name: String,
    fault_mode: SpectrometerFaultMode,
    state: Arc<RwLock<SpectrometerInternalState>>,
}

impl SimSpectrometer {
    pub fn new(name: String, fault_mode: SpectrometerFaultMode) -> Self {
        Self {
            name,
            fault_mode,
            state: Arc::new(RwLock::new(SpectrometerInternalState {
                integration_time: 100.0,
                acquiring: false,
                status: DeviceStatus::Idle,
                tick_count: 0,
                last_spectrum: vec![],
            })),
        }
    }

    fn generate_mock_spectrum(&self, size: usize) -> Vec<f64> {
        (0..size)
            .map(|i| {
                let x = i as f64 / size as f64;
                // Gaussian-ish peak
                let peak = (-((x - 0.5) * 10.0).powi(2)).exp() * 1000.0;
                // Add some noise
                let noise = (x * 137.0).sin() * 50.0;
                (peak + noise + 100.0).max(0.0)
            })
            .collect()
    }
}

#[async_trait]
impl Device for SimSpectrometer {
    fn name(&self) -> &str {
        &self.name
    }

    async fn read_state(&self) -> DeviceResult<DeviceState> {
        let state = self.state.read().await;

        // Signal saturated simulation
        if self.fault_mode == SpectrometerFaultMode::SignalSaturated && state.tick_count > 3 {
            return Err(HardwareError::with_context(
                self.name.clone(),
                ErrorType::SignalSaturated,
                Severity::Medium,
                "Detector signal saturated - reduce integration time".to_string(),
                chrono::Utc::now().to_rfc3339(),
                Some("reduce_integration_time".to_string()),
                HashMap::new(),
            ));
        }

        let mut telemetry = HashMap::new();
        telemetry.insert(
            "integration_time".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(state.integration_time).unwrap_or(serde_json::Number::from(0)),
            ),
        );
        telemetry.insert(
            "acquiring".to_string(),
            serde_json::Value::Bool(state.acquiring),
        );
        telemetry.insert(
            "spectrum_points".to_string(),
            serde_json::Value::Number(serde_json::Number::from(state.last_spectrum.len())),
        );

        Ok(DeviceState::with_telemetry(self.name.clone(), state.status, telemetry))
    }

    async fn execute(&self, action: &Action) -> DeviceResult<()> {
        let mut state = self.state.write().await;
        let params = action.get_params();

        match action.name.as_str() {
            "set_integration_time" => {
                let time = params
                    .get("integration_time")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(100.0)
                    .clamp(1.0, 10000.0);
                state.integration_time = time;
                tracing::info!("[{}] Integration time set to {}ms", self.name, time);
            }
            "acquire" => {
                state.acquiring = true;
                state.status = DeviceStatus::Running;
                tracing::info!("[{}] Starting acquisition", self.name);

                // Generate mock spectrum
                state.last_spectrum = self.generate_mock_spectrum(1024);
                state.acquiring = false;
                state.status = DeviceStatus::Idle;
                tracing::info!("[{}] Acquisition complete", self.name);
            }
            _ => {
                tracing::warn!("[{}] Unknown action: {}", self.name, action.name);
            }
        }

        Ok(())
    }

    async fn health(&self) -> bool {
        true
    }

    async fn tick(&self, _dt: f64) {
        let mut state = self.state.write().await;
        state.tick_count += 1;
    }
}

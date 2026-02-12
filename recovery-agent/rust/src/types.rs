//! Core types for exp-agent matching Python Pydantic models.
//!
//! These types are designed to be serializable and compatible with
//! the Python side via PyO3 bindings.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Type Aliases (matching Python Literals)
// ============================================================================

/// Effect of an action on the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass(eq, eq_int)]
pub enum Effect {
    #[serde(rename = "read")]
    Read,
    #[serde(rename = "write")]
    Write,
}

/// Type of recovery decision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass(eq, eq_int)]
pub enum DecisionType {
    #[serde(rename = "retry")]
    Retry,
    #[serde(rename = "skip")]
    Skip,
    #[serde(rename = "abort")]
    Abort,
    #[serde(rename = "degrade")]
    Degrade,
}

/// Severity level of an error
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass(eq, eq_int)]
pub enum Severity {
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
}

/// Criticality of a plan step
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass(eq, eq_int)]
pub enum Criticality {
    #[serde(rename = "critical")]
    Critical,
    #[serde(rename = "optional")]
    Optional,
}

/// Action to take on failure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass(eq, eq_int)]
pub enum OnFailure {
    #[serde(rename = "abort")]
    Abort,
    #[serde(rename = "retry")]
    Retry,
    #[serde(rename = "skip")]
    Skip,
}

/// Status of a device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass(eq, eq_int)]
pub enum DeviceStatus {
    #[serde(rename = "idle")]
    Idle,
    #[serde(rename = "running")]
    Running,
    #[serde(rename = "error")]
    Error,
}

impl Default for DeviceStatus {
    fn default() -> Self {
        DeviceStatus::Idle
    }
}

/// Type of hardware error
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[pyclass(eq, eq_int)]
pub enum ErrorType {
    #[serde(rename = "overshoot")]
    Overshoot,
    #[serde(rename = "timeout")]
    Timeout,
    #[serde(rename = "sensor_fail")]
    SensorFail,
    #[serde(rename = "safety_violation")]
    SafetyViolation,
    #[serde(rename = "postcondition_failed")]
    PostconditionFailed,
    #[serde(rename = "flow_blocked")]
    FlowBlocked,
    #[serde(rename = "collision")]
    Collision,
    #[serde(rename = "signal_saturated")]
    SignalSaturated,
    #[serde(rename = "motor_stall")]
    MotorStall,
    #[serde(rename = "encoder_error")]
    EncoderError,
}

/// Telemetry signature mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass(eq, eq_int)]
pub enum SignatureMode {
    #[serde(rename = "drift")]
    Drift,
    #[serde(rename = "oscillation")]
    Oscillation,
    #[serde(rename = "stall")]
    Stall,
    #[serde(rename = "noisy")]
    Noisy,
    #[serde(rename = "stable")]
    Stable,
    #[serde(rename = "unknown")]
    Unknown,
}

/// Sample status after error
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass(eq, eq_int)]
pub enum SampleStatus {
    #[serde(rename = "intact")]
    Intact,
    #[serde(rename = "compromised")]
    Compromised,
    #[serde(rename = "destroyed")]
    Destroyed,
    #[serde(rename = "anomalous")]
    Anomalous,
}

impl Default for SampleStatus {
    fn default() -> Self {
        SampleStatus::Intact
    }
}

// ============================================================================
// Core Data Types
// ============================================================================

/// State of a device at a point in time.
///
/// Telemetry is stored as JSON string internally for PyO3 compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct DeviceState {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub status: DeviceStatus,
    /// JSON-encoded telemetry data
    telemetry_json: String,
}

impl DeviceState {
    /// Create with HashMap telemetry (for Rust usage)
    pub fn with_telemetry(name: String, status: DeviceStatus, telemetry: HashMap<String, serde_json::Value>) -> Self {
        Self {
            name,
            status,
            telemetry_json: serde_json::to_string(&telemetry).unwrap_or_else(|_| "{}".to_string()),
        }
    }
}

#[pymethods]
impl DeviceState {
    #[new]
    #[pyo3(signature = (name, status = DeviceStatus::Idle, telemetry_json = None))]
    pub fn new(
        name: String,
        status: DeviceStatus,
        telemetry_json: Option<String>,
    ) -> Self {
        Self {
            name,
            status,
            telemetry_json: telemetry_json.unwrap_or_else(|| "{}".to_string()),
        }
    }

    /// Get telemetry as Python dict
    #[getter]
    pub fn telemetry(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        if let Ok(map) = serde_json::from_str::<HashMap<String, serde_json::Value>>(&self.telemetry_json) {
            for (k, v) in map {
                let py_val = json_to_py(py, &v)?;
                dict.set_item(k, py_val)?;
            }
        }
        Ok(dict.into())
    }

    /// Set telemetry from Python dict
    #[setter]
    pub fn set_telemetry(&mut self, py: Python<'_>, value: PyObject) -> PyResult<()> {
        let dict = value.downcast_bound::<PyDict>(py)?;
        let mut map = HashMap::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            let val = py_to_json(v.as_ref())?;
            map.insert(key, val);
        }
        self.telemetry_json = serde_json::to_string(&map).unwrap_or_else(|_| "{}".to_string());
        Ok(())
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Serialization error: {}", e))
        })
    }

    #[staticmethod]
    pub fn from_json(json_str: &str) -> PyResult<Self> {
        serde_json::from_str(json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Deserialization error: {}", e))
        })
    }
}

/// Hardware error with context for recovery decisions.
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
#[error("HardwareError({device}): [{severity:?}] {error_type:?} - {message}")]
#[pyclass]
pub struct HardwareError {
    #[pyo3(get, set)]
    pub device: String,
    #[pyo3(get, set)]
    #[serde(rename = "type")]
    pub error_type: ErrorType,
    #[pyo3(get, set)]
    pub severity: Severity,
    #[pyo3(get, set)]
    pub message: String,
    #[pyo3(get, set)]
    pub when: String,
    #[pyo3(get, set)]
    pub action: Option<String>,
    /// JSON-encoded context data
    context_json: String,
}

impl HardwareError {
    /// Create with HashMap context (for Rust usage)
    pub fn with_context(
        device: String,
        error_type: ErrorType,
        mut severity: Severity,
        message: String,
        when: String,
        action: Option<String>,
        context: HashMap<String, serde_json::Value>,
    ) -> Self {
        // Auto-escalate critical errors
        if matches!(error_type, ErrorType::Collision | ErrorType::SafetyViolation) {
            severity = Severity::High;
        }

        Self {
            device,
            error_type,
            severity,
            message,
            when,
            action,
            context_json: serde_json::to_string(&context).unwrap_or_else(|_| "{}".to_string()),
        }
    }
}

#[pymethods]
impl HardwareError {
    #[new]
    #[pyo3(signature = (device, error_type, severity, message, when = String::new(), action = None, context_json = None))]
    pub fn new(
        device: String,
        error_type: ErrorType,
        mut severity: Severity,
        message: String,
        when: String,
        action: Option<String>,
        context_json: Option<String>,
    ) -> Self {
        // Auto-escalate critical errors
        if matches!(error_type, ErrorType::Collision | ErrorType::SafetyViolation) {
            severity = Severity::High;
        }

        Self {
            device,
            error_type,
            severity,
            message,
            when,
            action,
            context_json: context_json.unwrap_or_else(|| "{}".to_string()),
        }
    }

    /// Get context as Python dict
    #[getter]
    pub fn context(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        if let Ok(map) = serde_json::from_str::<HashMap<String, serde_json::Value>>(&self.context_json) {
            for (k, v) in map {
                let py_val = json_to_py(py, &v)?;
                dict.set_item(k, py_val)?;
            }
        }
        Ok(dict.into())
    }

    pub fn model_dump(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("device", &self.device)?;
        dict.set_item("type", format!("{:?}", self.error_type).to_lowercase())?;
        dict.set_item("severity", format!("{:?}", self.severity).to_lowercase())?;
        dict.set_item("message", &self.message)?;
        dict.set_item("when", &self.when)?;
        dict.set_item("action", &self.action)?;
        dict.set_item("context", self.context(py)?)?;
        Ok(dict.into())
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Serialization error: {}", e))
        })
    }
}

/// An action to be executed on a device.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct Action {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub effect: Effect,
    /// JSON-encoded params
    params_json: String,
    #[pyo3(get, set)]
    pub irreversible: bool,
    #[pyo3(get, set)]
    pub preconditions: Vec<String>,
    #[pyo3(get, set)]
    pub postconditions: Vec<String>,
    #[pyo3(get, set)]
    pub safety_constraints: Vec<String>,
    #[pyo3(get, set)]
    pub device: Option<String>,
}

impl Action {
    /// Create with HashMap params (for Rust usage)
    pub fn with_params(
        name: String,
        effect: Effect,
        params: HashMap<String, serde_json::Value>,
        irreversible: bool,
        preconditions: Vec<String>,
        postconditions: Vec<String>,
        safety_constraints: Vec<String>,
        device: Option<String>,
    ) -> Self {
        Self {
            name,
            effect,
            params_json: serde_json::to_string(&params).unwrap_or_else(|_| "{}".to_string()),
            irreversible,
            preconditions,
            postconditions,
            safety_constraints,
            device,
        }
    }

    /// Get params as HashMap (for Rust usage)
    pub fn get_params(&self) -> HashMap<String, serde_json::Value> {
        serde_json::from_str(&self.params_json).unwrap_or_default()
    }
}

#[pymethods]
impl Action {
    #[new]
    #[pyo3(signature = (name, effect, params_json = None, irreversible = false, preconditions = None, postconditions = None, safety_constraints = None, device = None))]
    pub fn new(
        name: String,
        effect: Effect,
        params_json: Option<String>,
        irreversible: bool,
        preconditions: Option<Vec<String>>,
        postconditions: Option<Vec<String>>,
        safety_constraints: Option<Vec<String>>,
        device: Option<String>,
    ) -> Self {
        Self {
            name,
            effect,
            params_json: params_json.unwrap_or_else(|| "{}".to_string()),
            irreversible,
            preconditions: preconditions.unwrap_or_default(),
            postconditions: postconditions.unwrap_or_default(),
            safety_constraints: safety_constraints.unwrap_or_default(),
            device,
        }
    }

    /// Get params as Python dict
    #[getter]
    pub fn params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        if let Ok(map) = serde_json::from_str::<HashMap<String, serde_json::Value>>(&self.params_json) {
            for (k, v) in map {
                let py_val = json_to_py(py, &v)?;
                dict.set_item(k, py_val)?;
            }
        }
        Ok(dict.into())
    }

    /// Set params from Python dict
    #[setter]
    pub fn set_params(&mut self, py: Python<'_>, value: PyObject) -> PyResult<()> {
        let dict = value.downcast_bound::<PyDict>(py)?;
        let mut map = HashMap::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            let val = py_to_json(v.as_ref())?;
            map.insert(key, val);
        }
        self.params_json = serde_json::to_string(&map).unwrap_or_else(|_| "{}".to_string());
        Ok(())
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Serialization error: {}", e))
        })
    }
}

/// A recovery decision with rationale and actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct Decision {
    #[pyo3(get, set)]
    pub kind: DecisionType,
    #[pyo3(get, set)]
    pub rationale: String,
    #[pyo3(get, set)]
    pub actions: Vec<Action>,
}

#[pymethods]
impl Decision {
    #[new]
    #[pyo3(signature = (kind, rationale, actions = None))]
    pub fn new(kind: DecisionType, rationale: String, actions: Option<Vec<Action>>) -> PyResult<Self> {
        // Validate ABORT decisions have detailed rationale
        if matches!(kind, DecisionType::Abort) && rationale.len() < 10 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ABORT decisions require detailed rationale (min 10 chars)",
            ));
        }

        Ok(Self {
            kind,
            rationale,
            actions: actions.unwrap_or_default(),
        })
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Serialization error: {}", e))
        })
    }
}

// ============================================================================
// Recovery Policy Types
// ============================================================================

/// Classification profile for an error type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct ErrorProfile {
    #[pyo3(get, set)]
    pub unsafe_error: bool,
    #[pyo3(get, set)]
    pub recoverable: bool,
    #[pyo3(get, set)]
    pub default_strategy: DecisionType,
    #[pyo3(get, set)]
    pub safe_shutdown_required: bool,
    #[pyo3(get, set)]
    pub diagnostics: Vec<String>,
}

#[pymethods]
impl ErrorProfile {
    #[new]
    #[pyo3(signature = (unsafe_error, recoverable, default_strategy, safe_shutdown_required = false, diagnostics = None))]
    pub fn new(
        unsafe_error: bool,
        recoverable: bool,
        default_strategy: DecisionType,
        mut safe_shutdown_required: bool,
        diagnostics: Option<Vec<String>>,
    ) -> Self {
        // Ensure unrecoverable unsafe errors require safe shutdown
        if !recoverable && unsafe_error {
            safe_shutdown_required = true;
        }

        Self {
            unsafe_error,
            recoverable,
            default_strategy,
            safe_shutdown_required,
            diagnostics: diagnostics.unwrap_or_default(),
        }
    }
}

/// Result of telemetry signature analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct SignatureResult {
    #[pyo3(get, set)]
    pub mode: SignatureMode,
    #[pyo3(get, set)]
    pub confidence: f64,
    /// JSON-encoded details
    details_json: String,
}

#[pymethods]
impl SignatureResult {
    #[new]
    #[pyo3(signature = (mode, confidence, details_json = None))]
    pub fn new(
        mode: SignatureMode,
        confidence: f64,
        details_json: Option<String>,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Confidence must be between 0.0 and 1.0",
            ));
        }

        Ok(Self {
            mode,
            confidence,
            details_json: details_json.unwrap_or_else(|| "{}".to_string()),
        })
    }

    /// Get details as Python dict
    #[getter]
    pub fn details(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        if let Ok(map) = serde_json::from_str::<HashMap<String, serde_json::Value>>(&self.details_json) {
            for (k, v) in map {
                let py_val = json_to_py(py, &v)?;
                dict.set_item(k, py_val)?;
            }
        }
        Ok(dict.into())
    }
}

/// Full recovery decision with context.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct RecoveryDecision {
    #[pyo3(get, set)]
    pub kind: DecisionType,
    #[pyo3(get, set)]
    pub rationale: String,
    #[pyo3(get, set)]
    pub actions: Vec<Action>,
    #[pyo3(get, set)]
    pub error_profile: Option<ErrorProfile>,
    #[pyo3(get, set)]
    pub signature: Option<SignatureResult>,
    #[pyo3(get, set)]
    pub degraded_target: Option<f64>,
    #[pyo3(get, set)]
    pub sample_status: SampleStatus,
}

#[pymethods]
impl RecoveryDecision {
    #[new]
    #[pyo3(signature = (kind, rationale, actions = None, error_profile = None, signature = None, degraded_target = None, sample_status = SampleStatus::Intact))]
    pub fn new(
        kind: DecisionType,
        rationale: String,
        actions: Option<Vec<Action>>,
        error_profile: Option<ErrorProfile>,
        signature: Option<SignatureResult>,
        degraded_target: Option<f64>,
        sample_status: SampleStatus,
    ) -> Self {
        Self {
            kind,
            rationale,
            actions: actions.unwrap_or_default(),
            error_profile,
            signature,
            degraded_target,
            sample_status,
        }
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Serialization error: {}", e))
        })
    }
}

// ============================================================================
// Workflow Plan Types
// ============================================================================

/// A step in a workflow plan with criticality and failure semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct PlanStep {
    #[pyo3(get, set)]
    pub step_id: String,
    #[pyo3(get, set)]
    pub stage: String,
    #[pyo3(get, set)]
    pub action: Action,
    #[pyo3(get, set)]
    pub criticality: Criticality,
    #[pyo3(get, set)]
    pub on_failure: OnFailure,
    #[pyo3(get, set)]
    pub max_retries: u32,
    #[pyo3(get, set)]
    pub description: String,
}

#[pymethods]
impl PlanStep {
    #[new]
    #[pyo3(signature = (step_id, stage, action, criticality = Criticality::Critical, on_failure = OnFailure::Abort, max_retries = 2, description = String::new()))]
    pub fn new(
        step_id: String,
        stage: String,
        action: Action,
        criticality: Criticality,
        on_failure: OnFailure,
        max_retries: u32,
        description: String,
    ) -> PyResult<Self> {
        if max_retries > 10 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_retries must be <= 10",
            ));
        }

        Ok(Self {
            step_id,
            stage,
            action,
            criticality,
            on_failure,
            max_retries,
            description,
        })
    }
}

/// Produced when a degrade decision occurs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct PlanPatch {
    /// JSON-encoded overrides
    overrides_json: String,
    /// JSON-encoded relaxations
    relaxations_json: String,
    #[pyo3(get, set)]
    pub notes: Vec<String>,
    #[pyo3(get, set)]
    pub original_target: Option<f64>,
    #[pyo3(get, set)]
    pub degraded_target: Option<f64>,
}

#[pymethods]
impl PlanPatch {
    #[new]
    #[pyo3(signature = (overrides_json = None, relaxations_json = None, notes = None, original_target = None, degraded_target = None))]
    pub fn new(
        overrides_json: Option<String>,
        relaxations_json: Option<String>,
        notes: Option<Vec<String>>,
        original_target: Option<f64>,
        degraded_target: Option<f64>,
    ) -> Self {
        Self {
            overrides_json: overrides_json.unwrap_or_else(|| "{}".to_string()),
            relaxations_json: relaxations_json.unwrap_or_else(|| "{}".to_string()),
            notes: notes.unwrap_or_default(),
            original_target,
            degraded_target,
        }
    }

    /// Get overrides as Python dict
    #[getter]
    pub fn overrides(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        if let Ok(map) = serde_json::from_str::<HashMap<String, HashMap<String, serde_json::Value>>>(&self.overrides_json) {
            for (k, inner) in map {
                let inner_dict = PyDict::new_bound(py);
                for (k2, v) in inner {
                    let py_val = json_to_py(py, &v)?;
                    inner_dict.set_item(k2, py_val)?;
                }
                dict.set_item(k, inner_dict)?;
            }
        }
        Ok(dict.into())
    }

    /// Get relaxations as Python dict
    #[getter]
    pub fn relaxations(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        if let Ok(map) = serde_json::from_str::<HashMap<String, Vec<String>>>(&self.relaxations_json) {
            for (k, v) in map {
                dict.set_item(k, v)?;
            }
        }
        Ok(dict.into())
    }
}

// ============================================================================
// Helper functions for JSON <-> Python conversion
// ============================================================================

fn json_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<PyObject> {
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_py(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py(py))
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.clone().into_py(py)),
        serde_json::Value::Array(arr) => {
            let items: Vec<PyObject> = arr.iter().map(|v| json_to_py(py, v)).collect::<PyResult<_>>()?;
            let list = PyList::new_bound(py, items);
            Ok(list.into())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into())
        }
    }
}

fn py_to_json(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(serde_json::Value::Number(i.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    if let Ok(list) = obj.extract::<Vec<Bound<'_, pyo3::PyAny>>>() {
        let arr: Vec<serde_json::Value> = list.iter().map(|v| py_to_json(v)).collect::<PyResult<_>>()?;
        return Ok(serde_json::Value::Array(arr));
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, py_to_json(&v)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    Ok(serde_json::Value::Null)
}

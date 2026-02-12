//! Telemetry buffer for high-frequency data collection.
//!
//! This module provides a ring buffer for storing telemetry data from devices,
//! allowing for efficient time-series analysis and signature detection.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A single telemetry data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct TelemetryPoint {
    #[pyo3(get, set)]
    pub timestamp: f64,
    #[pyo3(get, set)]
    pub device: String,
    #[pyo3(get, set)]
    pub metric: String,
    #[pyo3(get, set)]
    pub value: f64,
}

#[pymethods]
impl TelemetryPoint {
    #[new]
    pub fn new(timestamp: f64, device: String, metric: String, value: f64) -> Self {
        Self {
            timestamp,
            device,
            metric,
            value,
        }
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Serialization error: {}", e))
        })
    }
}

/// High-performance ring buffer for telemetry data.
///
/// Thread-safe and optimized for high-frequency writes with
/// efficient time-range queries.
#[pyclass]
pub struct TelemetryBuffer {
    capacity: usize,
    buffer: Arc<RwLock<VecDeque<TelemetryPoint>>>,
}

impl TelemetryBuffer {
    fn new_internal(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: Arc::new(RwLock::new(VecDeque::with_capacity(capacity))),
        }
    }
}

#[pymethods]
impl TelemetryBuffer {
    /// Create a new telemetry buffer with specified capacity.
    #[new]
    #[pyo3(signature = (capacity = 10000))]
    pub fn new(capacity: usize) -> Self {
        Self::new_internal(capacity)
    }

    /// Add a new telemetry point to the buffer.
    ///
    /// If the buffer is at capacity, the oldest point is dropped.
    pub fn push(&self, point: TelemetryPoint) -> PyResult<()> {
        // Use blocking lock for Python compatibility
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
        })?;

        rt.block_on(async {
            let mut buffer = self.buffer.write().await;
            if buffer.len() >= self.capacity {
                buffer.pop_front();
            }
            buffer.push_back(point);
        });

        Ok(())
    }

    /// Get all points in the buffer.
    pub fn get_all(&self) -> PyResult<Vec<TelemetryPoint>> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
        })?;

        Ok(rt.block_on(async {
            let buffer = self.buffer.read().await;
            buffer.iter().cloned().collect()
        }))
    }

    /// Get points within a time range.
    #[pyo3(signature = (start_time, end_time = None))]
    pub fn get_range(&self, start_time: f64, end_time: Option<f64>) -> PyResult<Vec<TelemetryPoint>> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
        })?;

        Ok(rt.block_on(async {
            let buffer = self.buffer.read().await;
            buffer
                .iter()
                .filter(|p| {
                    p.timestamp >= start_time
                        && end_time.map_or(true, |end| p.timestamp <= end)
                })
                .cloned()
                .collect()
        }))
    }

    /// Get points for a specific device.
    pub fn get_by_device(&self, device: &str) -> PyResult<Vec<TelemetryPoint>> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
        })?;

        let device = device.to_string();
        Ok(rt.block_on(async {
            let buffer = self.buffer.read().await;
            buffer
                .iter()
                .filter(|p| p.device == device)
                .cloned()
                .collect()
        }))
    }

    /// Get points for a specific device and metric.
    pub fn get_by_device_metric(&self, device: &str, metric: &str) -> PyResult<Vec<TelemetryPoint>> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
        })?;

        let device = device.to_string();
        let metric = metric.to_string();
        Ok(rt.block_on(async {
            let buffer = self.buffer.read().await;
            buffer
                .iter()
                .filter(|p| p.device == device && p.metric == metric)
                .cloned()
                .collect()
        }))
    }

    /// Get the last N points.
    pub fn get_last(&self, n: usize) -> PyResult<Vec<TelemetryPoint>> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
        })?;

        Ok(rt.block_on(async {
            let buffer = self.buffer.read().await;
            let len = buffer.len();
            let skip = if len > n { len - n } else { 0 };
            buffer.iter().skip(skip).cloned().collect()
        }))
    }

    /// Calculate statistics for a device metric.
    pub fn stats(&self, device: &str, metric: &str) -> PyResult<TelemetryStats> {
        let points = self.get_by_device_metric(device, metric)?;

        if points.is_empty() {
            return Ok(TelemetryStats {
                count: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                slope: 0.0,
            });
        }

        let values: Vec<f64> = points.iter().map(|p| p.value).collect();
        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        // Calculate slope using linear regression
        let slope = if count > 1 {
            let x_mean = (count - 1) as f64 / 2.0;
            let numerator: f64 = values
                .iter()
                .enumerate()
                .map(|(i, v)| (i as f64 - x_mean) * (v - mean))
                .sum();
            let denominator: f64 = (0..count).map(|i| (i as f64 - x_mean).powi(2)).sum();
            if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        } else {
            0.0
        };

        Ok(TelemetryStats {
            count,
            min,
            max,
            mean,
            std_dev,
            slope,
        })
    }

    /// Clear the buffer.
    pub fn clear(&self) -> PyResult<()> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
        })?;

        rt.block_on(async {
            let mut buffer = self.buffer.write().await;
            buffer.clear();
        });

        Ok(())
    }

    /// Get the current size of the buffer.
    pub fn len(&self) -> PyResult<usize> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
        })?;

        Ok(rt.block_on(async {
            let buffer = self.buffer.read().await;
            buffer.len()
        }))
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> PyResult<bool> {
        Ok(self.len()? == 0)
    }

    /// Get the buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Statistics for telemetry data.
#[derive(Debug, Clone)]
#[pyclass]
pub struct TelemetryStats {
    #[pyo3(get)]
    pub count: usize,
    #[pyo3(get)]
    pub min: f64,
    #[pyo3(get)]
    pub max: f64,
    #[pyo3(get)]
    pub mean: f64,
    #[pyo3(get)]
    pub std_dev: f64,
    #[pyo3(get)]
    pub slope: f64,
}

#[pymethods]
impl TelemetryStats {
    fn __repr__(&self) -> String {
        format!(
            "TelemetryStats(count={}, min={:.2}, max={:.2}, mean={:.2}, std_dev={:.2}, slope={:.4})",
            self.count, self.min, self.max, self.mean, self.std_dev, self.slope
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_buffer_basic() {
        let buffer = TelemetryBuffer::new(100);

        buffer
            .push(TelemetryPoint::new(
                1.0,
                "heater".to_string(),
                "temperature".to_string(),
                25.0,
            ))
            .unwrap();
        buffer
            .push(TelemetryPoint::new(
                2.0,
                "heater".to_string(),
                "temperature".to_string(),
                30.0,
            ))
            .unwrap();

        assert_eq!(buffer.len().unwrap(), 2);
    }

    #[test]
    fn test_telemetry_buffer_overflow() {
        let buffer = TelemetryBuffer::new(3);

        for i in 0..5 {
            buffer
                .push(TelemetryPoint::new(
                    i as f64,
                    "heater".to_string(),
                    "temperature".to_string(),
                    i as f64 * 10.0,
                ))
                .unwrap();
        }

        // Should only keep last 3 points
        assert_eq!(buffer.len().unwrap(), 3);

        let points = buffer.get_all().unwrap();
        assert_eq!(points[0].timestamp, 2.0);
        assert_eq!(points[2].timestamp, 4.0);
    }

    #[test]
    fn test_telemetry_stats() {
        let buffer = TelemetryBuffer::new(100);

        // Add points with increasing temperature (drift pattern)
        for i in 0..10 {
            buffer
                .push(TelemetryPoint::new(
                    i as f64,
                    "heater".to_string(),
                    "temperature".to_string(),
                    25.0 + i as f64,
                ))
                .unwrap();
        }

        let stats = buffer.stats("heater", "temperature").unwrap();
        assert_eq!(stats.count, 10);
        assert_eq!(stats.min, 25.0);
        assert_eq!(stats.max, 34.0);
        assert!((stats.mean - 29.5).abs() < 0.01);
        assert!(stats.slope > 0.9); // Positive slope indicates drift
    }
}

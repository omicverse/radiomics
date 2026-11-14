//! Minimal owned-vector boundary. Core algorithms never call Python.
use ov_radiomics_core as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn input_error(error: core::InputError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
mod _native {
    use super::*;

    #[pymodule_export]
    const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

    #[pyfunction]
    fn auc(py: Python<'_>, y: Vec<u8>, scores: Vec<f64>) -> PyResult<f64> {
        py.detach(|| core::auc(&y, &scores)).map_err(input_error)
    }

    #[pyfunction]
    fn binary_summary(
        py: Python<'_>,
        y: Vec<u8>,
        scores: Vec<f64>,
        threshold: f64,
    ) -> PyResult<(f64, u64, u64, u64, u64, f64)> {
        let s = py
            .detach(|| core::binary_summary(&y, &scores, threshold))
            .map_err(input_error)?;
        Ok((s.auc, s.tn, s.fp, s.fn_count, s.tp, s.brier))
    }

    #[pyfunction]
    fn delong_summary(py: Python<'_>, y: Vec<u8>, scores: Vec<f64>) -> PyResult<(f64, f64)> {
        py.detach(|| core::delong_summary(&y, &scores))
            .map_err(input_error)
    }

    #[pyfunction]
    fn paired_delong_summary(
        py: Python<'_>,
        y: Vec<u8>,
        a: Vec<f64>,
        b: Vec<f64>,
    ) -> PyResult<(f64, f64, f64)> {
        py.detach(|| core::paired_delong_summary(&y, &a, &b))
            .map_err(input_error)
    }

    #[pyfunction]
    fn net_benefit(
        py: Python<'_>,
        y: Vec<u8>,
        scores: Vec<f64>,
        thresholds: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| core::net_benefit(&y, &scores, &thresholds))
            .map_err(input_error)
    }

    #[pyfunction]
    fn hl_statistic(
        py: Python<'_>,
        counts: Vec<u64>,
        observed: Vec<f64>,
        expected: Vec<f64>,
    ) -> PyResult<f64> {
        py.detach(|| core::hl_statistic(&counts, &observed, &expected))
            .map_err(input_error)
    }
}

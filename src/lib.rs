//! SigmaOS ML/DL component — foundational ML primitives.
//!
//! Linear regression with batch gradient descent, feature standardization,
//! and regression metrics. Pure Rust, deterministic, no unsafe code.

#![forbid(unsafe_code)]

use std::fmt;

/// Errors produced by the ML component.
#[derive(Debug)]
pub enum MlError {
    /// Input dimensions are inconsistent.
    DimensionMismatch { expected: usize, got: usize },
    /// Insufficient data for training.
    InsufficientData { need: usize, got: usize },
    /// Numerical failure (e.g. singular matrix, NaN).
    Numerical(String),
}

impl fmt::Display for MlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InsufficientData { need, got } => {
                write!(f, "need at least {need} samples, got {got}")
            }
            Self::Numerical(m) => write!(f, "numerical error: {m}"),
        }
    }
}

impl std::error::Error for MlError {}

/// Feature standardization parameters (z-score).
#[derive(Debug, Clone, PartialEq)]
pub struct Scaler {
    means: Vec<f64>,
    stds: Vec<f64>,
}

impl Scaler {
    /// Fit scaler on training data (rows = samples, cols = features).
    pub fn fit(data: &[Vec<f64>]) -> Result<Self, MlError> {
        if data.is_empty() || data[0].is_empty() {
            return Err(MlError::InsufficientData { need: 1, got: 0 });
        }
        let cols = data[0].len();
        if data.iter().any(|r| r.len() != cols) {
            return Err(MlError::DimensionMismatch {
                expected: cols,
                got: data.iter().map(|r| r.len()).max().unwrap_or(0),
            });
        }
        let n = data.len() as f64;
        let mut means = vec![0.0; cols];
        let mut stds = vec![0.0; cols];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                means[j] += v;
            }
        }
        for m in means.iter_mut() {
            *m /= n;
        }
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                let d = v - means[j];
                stds[j] += d * d;
            }
        }
        for (j, s) in stds.iter_mut().enumerate() {
            *s = (*s / n).sqrt();
            if *s < 1e-12 {
                *s = 1.0;
            } // constant feature → unit scale
            let _ = j;
        }
        Ok(Self { means, stds })
    }

    /// Transform one sample.
    pub fn transform(&self, row: &[f64]) -> Vec<f64> {
        row.iter()
            .enumerate()
            .map(|(j, &v)| (v - self.means[j]) / self.stds[j])
            .collect()
    }
}

/// Linear regression model (with bias term).
#[derive(Debug, Clone)]
pub struct LinearRegression {
    /// Weights, last element is the bias.
    pub weights: Vec<f64>,
}

impl LinearRegression {
    /// Train with batch gradient descent.
    ///
    /// `x`: samples × features, `y`: targets. `lr`: learning rate,
    /// `epochs`: passes over data, `l2`: L2 regularization strength.
    pub fn train(
        x: &[Vec<f64>],
        y: &[f64],
        lr: f64,
        epochs: usize,
        l2: f64,
    ) -> Result<Self, MlError> {
        if x.len() != y.len() {
            return Err(MlError::DimensionMismatch {
                expected: y.len(),
                got: x.len(),
            });
        }
        if x.len() < 2 {
            return Err(MlError::InsufficientData {
                need: 2,
                got: x.len(),
            });
        }
        let features = x[0].len();
        if x.iter().any(|r| r.len() != features) {
            return Err(MlError::DimensionMismatch {
                expected: features,
                got: x.iter().map(|r| r.len()).max().unwrap_or(0),
            });
        }
        let mut w = vec![0.0; features + 1]; // +1 bias
        let n = x.len() as f64;
        for _ in 0..epochs {
            let mut grad = vec![0.0; features + 1];
            for (row, &target) in x.iter().zip(y.iter()) {
                let pred = Self::predict_raw(&w, row);
                let err = pred - target;
                for j in 0..features {
                    grad[j] += err * row[j];
                }
                grad[features] += err;
            }
            for j in 0..=features {
                let reg = if j < features { l2 * w[j] } else { 0.0 };
                w[j] -= lr * (grad[j] / n + reg);
            }
        }
        Ok(Self { weights: w })
    }

    fn predict_raw(w: &[f64], row: &[f64]) -> f64 {
        let mut s = w[w.len() - 1]; // bias
        for (j, &v) in row.iter().enumerate() {
            s += w[j] * v;
        }
        s
    }

    /// Predict a single sample.
    pub fn predict(&self, row: &[f64]) -> f64 {
        Self::predict_raw(&self.weights, row)
    }
}

/// Mean absolute error.
pub fn mae(y_true: &[f64], y_pred: &[f64]) -> Result<f64, MlError> {
    check_len(y_true, y_pred)?;
    Ok(y_true
        .iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / y_true.len() as f64)
}

/// Root mean squared error.
pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> Result<f64, MlError> {
    check_len(y_true, y_pred)?;
    let mse = y_true
        .iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / y_true.len() as f64;
    Ok(mse.sqrt())
}

/// R² coefficient of determination.
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64, MlError> {
    check_len(y_true, y_pred)?;
    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_tot: f64 = y_true.iter().map(|y| (y - mean) * (y - mean)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    if ss_tot == 0.0 {
        return Err(MlError::Numerical("zero total variance".into()));
    }
    Ok(1.0 - ss_res / ss_tot)
}

fn check_len(a: &[f64], b: &[f64]) -> Result<(), MlError> {
    if a.len() != b.len() {
        return Err(MlError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_linear_function() {
        // y = 2x + 1 with small noise-free data
        let x: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..50).map(|i| 2.0 * i as f64 + 1.0).collect();
        let scaler = Scaler::fit(&x).unwrap();
        let xs: Vec<Vec<f64>> = x.iter().map(|r| scaler.transform(r)).collect();
        let model = LinearRegression::train(&xs, &y, 0.5, 500, 0.0).unwrap();
        let pred: Vec<f64> = xs.iter().map(|r| model.predict(r)).collect();
        let score = r2_score(&y, &pred).unwrap();
        assert!(score > 0.999, "r2 = {score}");
        assert!(rmse(&y, &pred).unwrap() < 0.5);
    }

    #[test]
    fn rejects_bad_dims() {
        assert!(LinearRegression::train(&[vec![1.0]], &[1.0, 2.0], 0.1, 1, 0.0).is_err());
    }
}

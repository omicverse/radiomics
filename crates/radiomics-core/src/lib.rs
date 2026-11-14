//! Numerical kernels without Python, NumPy, imaging-library, or FFI dependencies.
//!
//! Inputs are independent subjects, binary labels, and positive-class probabilities.
//! Distribution quantiles and presentation belong to the calling language's wrapper.

use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputError(pub &'static str);

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl Error for InputError {}

fn validate(y: &[u8], scores: &[f64]) -> Result<(), InputError> {
    if y.len() != scores.len() || y.is_empty() {
        return Err(InputError(
            "Labels and probabilities must have equal nonzero lengths.",
        ));
    }
    if y.iter().any(|&value| value > 1) || !y.contains(&0) || !y.contains(&1) {
        return Err(InputError(
            "Labels must contain both binary classes 0 and 1.",
        ));
    }
    if scores
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err(InputError(
            "Probabilities must be finite and between 0 and 1.",
        ));
    }
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    let center = mean(values);
    values.iter().map(|x| (x - center).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

/// ROC AUC using average ranks for tied scores, including signed zero ties.
pub fn auc(y: &[u8], scores: &[f64]) -> Result<f64, InputError> {
    validate(y, scores)?;
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_unstable_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    let positives = y.iter().filter(|&&label| label == 1).count() as f64;
    let negatives = y.len() as f64 - positives;
    let mut sum = 0.0;
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && scores[order[end]] == scores[order[start]] {
            end += 1;
        }
        let rank = (start + end + 1) as f64 / 2.0;
        let positive_ties = order[start..end].iter().filter(|&&i| y[i] == 1).count();
        sum += positive_ties as f64 * rank;
        start = end;
    }
    Ok((sum - positives * (positives + 1.0) / 2.0) / (positives * negatives))
}

#[derive(Debug, Clone)]
pub struct BinarySummary {
    pub auc: f64,
    pub tn: u64,
    pub fp: u64,
    pub fn_count: u64,
    pub tp: u64,
    pub brier: f64,
}

pub fn binary_summary(
    y: &[u8],
    scores: &[f64],
    threshold: f64,
) -> Result<BinarySummary, InputError> {
    validate(y, scores)?;
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(InputError("Threshold must be finite and between 0 and 1."));
    }
    let mut result = BinarySummary {
        auc: auc(y, scores)?,
        tn: 0,
        fp: 0,
        fn_count: 0,
        tp: 0,
        brier: 0.0,
    };
    for (&label, &p) in y.iter().zip(scores) {
        match (label, p >= threshold) {
            (0, false) => result.tn += 1,
            (0, true) => result.fp += 1,
            (1, false) => result.fn_count += 1,
            (1, true) => result.tp += 1,
            _ => unreachable!("binary labels validated"),
        }
        result.brier += (p - label as f64).powi(2);
    }
    result.brier /= y.len() as f64;
    Ok(result)
}

/// DeLong placements in original within-class order, with half credit for ties.
pub fn delong_components(y: &[u8], scores: &[f64]) -> Result<(Vec<f64>, Vec<f64>), InputError> {
    validate(y, scores)?;
    let positive: Vec<f64> = y
        .iter()
        .zip(scores)
        .filter_map(|(&y, &p)| (y == 1).then_some(p))
        .collect();
    let negative: Vec<f64> = y
        .iter()
        .zip(scores)
        .filter_map(|(&y, &p)| (y == 0).then_some(p))
        .collect();
    if positive.len() < 2 || negative.len() < 2 {
        return Err(InputError(
            "DeLong inference needs at least two observations in each class.",
        ));
    }
    let mut sorted_positive = positive.clone();
    let mut sorted_negative = negative.clone();
    sorted_positive.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_negative.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let placements = |value: f64, sorted: &[f64]| -> f64 {
        (sorted.partition_point(|&x| x < value) + sorted.partition_point(|&x| x <= value)) as f64
            / (2 * sorted.len()) as f64
    };
    let v10 = positive
        .iter()
        .map(|&p| placements(p, &sorted_negative))
        .collect();
    let v01 = negative
        .iter()
        .map(|&p| 1.0 - placements(p, &sorted_positive))
        .collect();
    Ok((v10, v01))
}

/// Return (AUC, estimated AUC variance).
pub fn delong_summary(y: &[u8], scores: &[f64]) -> Result<(f64, f64), InputError> {
    let (v10, v01) = delong_components(y, scores)?;
    Ok((
        mean(&v10),
        variance(&v10) / v10.len() as f64 + variance(&v01) / v01.len() as f64,
    ))
}

/// Return (AUC A, AUC B, variance of the paired AUC difference).
pub fn paired_delong_summary(
    y: &[u8],
    a: &[f64],
    b: &[f64],
) -> Result<(f64, f64, f64), InputError> {
    let (a10, a01) = delong_components(y, a)?;
    let (b10, b01) = delong_components(y, b)?;
    let d10: Vec<f64> = a10.iter().zip(&b10).map(|(a, b)| a - b).collect();
    let d01: Vec<f64> = a01.iter().zip(&b01).map(|(a, b)| a - b).collect();
    Ok((
        mean(&a10),
        mean(&b10),
        variance(&d10) / d10.len() as f64 + variance(&d01) / d01.len() as f64,
    ))
}

/// O(n log n + t log n) threshold evaluation. Preserve threshold order and duplicates.
pub fn net_benefit(y: &[u8], scores: &[f64], thresholds: &[f64]) -> Result<Vec<f64>, InputError> {
    validate(y, scores)?;
    if thresholds
        .iter()
        .any(|t| !t.is_finite() || *t <= 0.0 || *t >= 1.0)
    {
        return Err(InputError(
            "DCA thresholds must lie strictly between 0 and 1.",
        ));
    }
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_unstable_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    let sorted: Vec<f64> = order.iter().map(|&i| scores[i]).collect();
    let mut suffix_positive = vec![0_u64; y.len() + 1];
    for i in (0..y.len()).rev() {
        suffix_positive[i] = suffix_positive[i + 1] + u64::from(y[order[i]]);
    }
    Ok(thresholds
        .iter()
        .map(|&t| {
            let index = sorted.partition_point(|&p| p < t);
            let tp = suffix_positive[index] as f64;
            let fp = (y.len() - index) as f64 - tp;
            (tp - fp * t / (1.0 - t)) / y.len() as f64
        })
        .collect())
}

/// Pearson H-L statistic for precomputed bins; the wrapper controls quantile grouping.
pub fn hl_statistic(counts: &[u64], observed: &[f64], expected: &[f64]) -> Result<f64, InputError> {
    if counts.len() < 3 || counts.len() != observed.len() || counts.len() != expected.len() {
        return Err(InputError("H-L requires at least three aligned groups."));
    }
    let mut statistic = 0.0;
    for ((&n, &o), &e) in counts.iter().zip(observed).zip(expected) {
        let size = n as f64;
        if n == 0
            || !o.is_finite()
            || !e.is_finite()
            || o < 0.0
            || o > size
            || e <= 0.0
            || e >= size
        {
            return Err(InputError(
                "Invalid H-L group counts or expected probability 0 or 1.",
            ));
        }
        statistic += (o - e).powi(2) / (e * (1.0 - e / size));
    }
    Ok(statistic)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn known_auc_and_confusion() {
        let s = binary_summary(&[0, 0, 1, 1], &[0.1, 0.4, 0.35, 0.8], 0.5).unwrap();
        close(s.auc, 0.75);
        assert_eq!((s.tn, s.fp, s.fn_count, s.tp), (2, 0, 1, 1));
        close(s.brier, (0.01 + 0.16 + 0.4225 + 0.04) / 4.0);
    }

    #[test]
    fn ties_signed_zero_and_reversal() {
        close(auc(&[0, 0, 1, 1], &[0.5; 4]).unwrap(), 0.5);
        close(auc(&[0, 0, 1, 1], &[0.0, -0.0, -0.0, 0.0]).unwrap(), 0.5);
        close(auc(&[1, 1, 0, 0], &[0.1, 0.4, 0.35, 0.8]).unwrap(), 0.25);
    }

    #[test]
    fn delong_known_placements() {
        let y = [0, 0, 1, 1];
        let p = [0.1, 0.4, 0.35, 0.8];
        let (v10, v01) = delong_components(&y, &p).unwrap();
        assert_eq!(v10, vec![0.5, 1.0]);
        assert_eq!(v01, vec![1.0, 0.5]);
        let (a, v) = delong_summary(&y, &p).unwrap();
        close(a, 0.75);
        close(v, 0.125);
        close(paired_delong_summary(&y, &p, &p).unwrap().2, 0.0);
    }

    #[test]
    fn decision_curve_boundaries_and_order() {
        let result = net_benefit(&[0, 0, 1, 1], &[0.1, 0.4, 0.35, 0.8], &[0.5, 0.8, 0.5]).unwrap();
        assert_eq!(result, vec![0.25, 0.25, 0.25]);
        assert!(net_benefit(&[0, 1], &[0.1, 0.9], &[1.0]).is_err());
    }

    #[test]
    fn hl_exact_calibration() {
        close(
            hl_statistic(&[10, 10, 10], &[2.0, 5.0, 8.0], &[2.0, 5.0, 8.0]).unwrap(),
            0.0,
        );
        assert!(hl_statistic(&[10, 10, 10], &[2.0, 5.0, 8.0], &[0.0, 5.0, 8.0]).is_err());
    }

    #[test]
    fn reject_invalid_inputs() {
        assert!(auc(&[], &[]).is_err());
        assert!(auc(&[0, 1], &[0.3]).is_err());
        assert!(auc(&[0, 0], &[0.1, 0.3]).is_err());
        assert!(auc(&[0, 2], &[0.1, 0.3]).is_err());
        assert!(auc(&[0, 1], &[f64::NAN, 0.3]).is_err());
        assert!(auc(&[0, 1], &[f64::INFINITY, 0.3]).is_err());
        assert!(auc(&[0, 1], &[-0.1, 0.3]).is_err());
        assert!(delong_summary(&[0, 1], &[0.1, 0.3]).is_err());
    }
}

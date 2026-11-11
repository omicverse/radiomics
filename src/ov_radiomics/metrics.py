"""Binary discrimination, paired DeLong inference, calibration, and net benefit."""
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from sklearn.metrics import (roc_auc_score, confusion_matrix, precision_score, recall_score,
                            f1_score, accuracy_score, brier_score_loss)
from ._validation import probabilities


def _delong_components(y, scores):
    # Independent implementation of pairwise placement values with half credit for ties.
    # Sort/search avoids materializing an n_positive by n_negative matrix.
    pos, neg = np.asarray(scores)[y == 1], np.asarray(scores)[y == 0]
    if min(len(pos), len(neg)) < 2:
        raise ValueError('DeLong inference needs at least two observations in each class.')
    sorted_neg, sorted_pos = np.sort(neg), np.sort(pos)
    v10 = (np.searchsorted(sorted_neg, pos, side='left') + np.searchsorted(sorted_neg, pos, side='right')) / (2 * len(neg))
    v01 = 1 - (np.searchsorted(sorted_pos, neg, side='left') + np.searchsorted(sorted_pos, neg, side='right')) / (2 * len(pos))
    return v10, v01


def binary_metrics(y_true, y_score, threshold=0.5, positive_label=1, ci_method='delong',
                   confidence=0.95, n_bootstrap=2000, random_state=0):
    y, p = probabilities(y_true, y_score, positive_label)
    if not 0 <= threshold <= 1 or not 0 < confidence < 1:
        raise ValueError('Invalid threshold or confidence.')
    auc = roc_auc_score(y, p)
    if ci_method == 'delong':
        v10, v01 = _delong_components(y, p)
        variance = np.var(v10, ddof=1)/len(v10) + np.var(v01, ddof=1)/len(v01)
        delta = norm.ppf((1+confidence)/2) * np.sqrt(max(variance, 0))
        low, high = max(0, auc-delta), min(1, auc+delta)
    elif ci_method == 'bootstrap':
        if n_bootstrap < 2:
            raise ValueError('n_bootstrap must be at least 2.')
        rng = np.random.RandomState(random_state)
        # Stratified bootstrap preserves both classes.
        positives, negatives = np.flatnonzero(y), np.flatnonzero(y == 0)
        estimates = []
        for _ in range(n_bootstrap):
            idx = np.r_[rng.choice(positives, len(positives)), rng.choice(negatives, len(negatives))]
            estimates.append(roc_auc_score(y[idx], p[idx]))
        low, high = np.quantile(estimates, [(1-confidence)/2, (1+confidence)/2])
    elif ci_method is None:
        low, high = np.nan, np.nan
    else:
        raise ValueError('ci_method must be delong, bootstrap, or None.')
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    safe = lambda numerator, denominator: numerator/denominator if denominator else np.nan
    return pd.Series({'n': len(y), 'positive_n': int(y.sum()), 'accuracy': accuracy_score(y, predicted),
        'auc': auc, 'ci_low': low, 'ci_high': high, 'sensitivity': safe(tp, tp+fn),
        'specificity': safe(tn, tn+fp), 'ppv': safe(tp, tp+fp), 'npv': safe(tn, tn+fn),
        'precision': precision_score(y, predicted, zero_division=0),
        'recall': recall_score(y, predicted, zero_division=0), 'f1': f1_score(y, predicted, zero_division=0),
        'brier': brier_score_loss(y, p), 'threshold': threshold})


def delong_test(y_true, score_a, score_b, positive_label=1):
    """Two-sided paired test; rows must refer to the same independent subjects."""
    y, a = probabilities(y_true, score_a, positive_label)
    _, b = probabilities(y_true, score_b, positive_label)
    a10, a01 = _delong_components(y, a)
    b10, b01 = _delong_components(y, b)
    variance = np.var(a10-b10, ddof=1)/len(a10) + np.var(a01-b01, ddof=1)/len(a01)
    difference = float(a10.mean()-b10.mean())
    z = difference/np.sqrt(variance) if variance > 0 else (0.0 if difference == 0 else np.sign(difference)*np.inf)
    return pd.Series({'auc_a': a10.mean(), 'auc_b': b10.mean(), 'difference': difference,
                      'z': z, 'p_value': 2*norm.sf(abs(z))})


def decision_curve(y_true, scores, thresholds=None, positive_label=1):
    if not isinstance(scores, (dict, pd.DataFrame)):
        scores = {'Model': scores}
    thresholds = np.asarray(np.linspace(0.01, 0.99, 99) if thresholds is None else thresholds)
    if thresholds.ndim != 1 or not np.isfinite(thresholds).all() or ((thresholds <= 0) | (thresholds >= 1)).any():
        raise ValueError('DCA thresholds must lie strictly between 0 and 1.')
    if len(scores) == 0 or set(scores) & {'Treat all', 'Treat none'}:
        raise ValueError('Supply models with non-reserved names.')
    rows = []
    for name in scores:
        y, p = probabilities(y_true, scores[name], positive_label)
        for t in thresholds:
            treated = p >= t
            benefit = ((treated & (y == 1)).sum() - (treated & (y == 0)).sum()*t/(1-t))/len(y)
            rows.append((name, t, benefit))
    prevalence = y.mean()
    for t in thresholds:
        rows.extend([('Treat all', t, prevalence-(1-prevalence)*t/(1-t)), ('Treat none', t, 0)])
    return pd.DataFrame(rows, columns=['model', 'threshold', 'net_benefit'])


def hosmer_lemeshow(y_true, y_score, n_bins=10, positive_label=1):
    y, p = probabilities(y_true, y_score, positive_label)
    if n_bins < 3:
        raise ValueError('At least three requested bins are required.')
    frame = pd.DataFrame({'observed': y, 'expected': p})
    frame['bin'] = pd.qcut(frame.expected, n_bins, duplicates='drop')
    grouped = frame.groupby('bin', observed=True).agg(n=('observed', 'size'), observed=('observed', 'sum'), expected=('expected', 'sum'))
    if len(grouped) < 3:
        raise ValueError('Fewer than three distinct probability bins; H-L is undefined.')
    denominator = grouped.expected * (1-grouped.expected/grouped.n)
    if (denominator <= 0).any():
        raise ValueError('H-L is undefined for bins with expected probability 0 or 1.')
    statistic = float(((grouped.observed-grouped.expected)**2/denominator).sum())
    df = len(grouped)-2
    return {'statistic': statistic, 'df': df, 'p_value': float(chi2.sf(statistic, df)), 'table': grouped}

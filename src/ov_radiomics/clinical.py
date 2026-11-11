"""Observed-data baseline tables and explicit logistic inference."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from ._validation import binary_target


def baseline_table(data, target, features, continuous, categorical=None, split=None,
                   continuous_test='mannwhitney', categorical_test='auto'):
    """Compare outcome classes within each split. No imputation, encoding, or selection.

    Continuous summaries: n, mean, SD, median, Q1, Q3. Categorical summaries:
    counts and proportions among nonmissing observations, with missing counts separate.
    """
    features, continuous = list(features), list(continuous)
    categorical = list(categorical) if categorical is not None else [f for f in features if f not in continuous]
    if set(continuous) & set(categorical) or set(continuous+categorical) != set(features):
        raise ValueError('continuous/categorical must partition features.')
    if target in features or split in features:
        raise ValueError('Exclude target and split from features.')
    classes = sorted(pd.unique(data[target].dropna()), key=str)
    if len(classes) != 2 or data[target].isna().any():
        raise ValueError('Baseline analysis expects two nonmissing target classes.')
    cohorts = data.groupby(split) if split else [('all', data)]
    rows = []
    for cohort, subset in cohorts:
        for feature in features:
            row = {'cohort': cohort, 'feature': feature, 'type': 'continuous' if feature in continuous else 'categorical'}
            values = [subset.loc[subset[target] == cls, feature].dropna() for cls in classes]
            for cls, series in zip(classes, values):
                prefix = f'class_{cls}'
                row[prefix+'_n'] = len(series)
                row[prefix+'_missing'] = int(subset.loc[subset[target] == cls, feature].isna().sum())
                if feature in continuous:
                    row.update({prefix+'_mean': series.mean(), prefix+'_sd': series.std(),
                        prefix+'_median': series.median(), prefix+'_q1': series.quantile(.25), prefix+'_q3': series.quantile(.75)})
                else:
                    row[prefix+'_counts'] = series.value_counts().to_dict()
                    row[prefix+'_proportions'] = series.value_counts(normalize=True).to_dict()
            if min(map(len, values)) == 0:
                p, test = np.nan, 'insufficient observations'
            elif feature in continuous:
                if continuous_test == 'mannwhitney':
                    p, test = stats.mannwhitneyu(*values, alternative='two-sided').pvalue, 'mannwhitney'
                elif continuous_test == 'welch':
                    p, test = stats.ttest_ind(*values, equal_var=False).pvalue, 'welch'
                else:
                    raise ValueError('continuous_test must be mannwhitney or welch.')
            else:
                table = pd.crosstab(subset[feature], subset[target])
                if min(table.shape) < 2:
                    p, test = np.nan, 'constant or absent class'
                else:
                    _, chi_p, _, expected = stats.chi2_contingency(table, correction=False)
                    fisher = categorical_test == 'fisher' or (categorical_test == 'auto' and table.shape == (2, 2) and (expected < 5).any())
                    if categorical_test not in ('auto', 'fisher', 'chi2'):
                        raise ValueError('categorical_test must be auto, fisher, or chi2.')
                    if fisher:
                        if table.shape != (2, 2):
                            raise ValueError('Fisher exact test is available for 2x2 tables only.')
                        p, test = stats.fisher_exact(table)[1], 'fisher'
                    else:
                        p, test = chi_p, 'chi2'
            row.update(p_value=p, test=test)
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class LogisticAnalysis:
    table: pd.DataFrame
    model: object
    n_used: int
    n_dropped: int
    reference: dict


def multivariable_logistic(data, target, features, categorical=None, reference=None, confidence=0.95):
    """Complete-case unpenalized Logit with explicit reference categories and Wald CIs.

    No implicit stepwise selection. Perfect separation / rank deficiency are errors.
    """
    if not 0 < confidence < 1 or target in features or not features:
        raise ValueError('Invalid confidence or feature list.')
    categorical, reference = list(categorical or []), dict(reference or {})
    if not set(categorical).issubset(features) or not set(reference).issubset(categorical):
        raise ValueError('Categorical/reference settings must refer to selected features.')
    complete = data[list(features)+[target]].dropna().copy()
    y = binary_target(complete[target])
    used_reference = {}
    for column in categorical:
        levels = sorted(complete[column].unique(), key=str)
        base = reference.get(column, levels[0])
        if base not in levels:
            raise ValueError(f'Reference {base} absent from {column}.')
        levels = [base] + [value for value in levels if value != base]
        complete[column] = pd.Categorical(complete[column], categories=levels)
        used_reference[column] = base
    X = pd.get_dummies(complete[list(features)], columns=categorical, drop_first=True).astype(float)
    X = sm.add_constant(X, has_constant='add')
    if len(X) <= X.shape[1] or np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        raise ValueError('Insufficient complete cases or rank-deficient design matrix.')
    fitted = sm.Logit(y, X).fit(disp=False, maxiter=200)
    if not fitted.mle_retvals['converged']:
        raise ValueError('Logistic inference did not converge.')
    ci = fitted.conf_int(alpha=1-confidence)
    table = pd.DataFrame({'term': X.columns, 'coefficient': fitted.params.to_numpy(),
        'odds_ratio': np.exp(fitted.params.to_numpy()), 'ci_low': np.exp(ci.iloc[:, 0].to_numpy()),
        'ci_high': np.exp(ci.iloc[:, 1].to_numpy()), 'p_value': fitted.pvalues.to_numpy()})
    return LogisticAnalysis(table, fitted, len(complete), len(data)-len(complete), used_reference)


def univariate_logistic(data, target, features, categorical=None, reference=None, confidence=0.95):
    rows = []
    for feature in features:
        result = multivariable_logistic(data, target, [feature],
            categorical=[feature] if feature in (categorical or []) else [],
            reference={feature: reference[feature]} if reference and feature in reference else None,
            confidence=confidence)
        table = result.table.loc[result.table.term != 'const'].copy()
        table['feature'], table['n_used'], table['n_dropped'] = feature, result.n_used, result.n_dropped
        rows.append(table)
    return pd.concat(rows, ignore_index=True)

"""Train-fitted filtering and nested tuning of a LASSO feature selector."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Lasso, lasso_path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests
from ._validation import numeric_frame, binary_target


class _Filter(BaseEstimator, TransformerMixin):
    def __init__(self, scale='standard', missing='drop', univariate='mannwhitney',
                 p_threshold=0.05, p_adjust=None, corr_method='pearson', corr_threshold=0.9):
        self.scale = scale
        self.missing = missing
        self.univariate = univariate
        self.p_threshold = p_threshold
        self.p_adjust = p_adjust
        self.corr_method = corr_method
        self.corr_threshold = corr_threshold

    def fit(self, X, y):
        X = numeric_frame(X)
        y = binary_target(y)
        if self.missing not in ('drop', 'median') or self.scale not in (None, 'standard'):
            raise ValueError('missing must be drop/median; scale must be standard/None.')
        if self.corr_method not in ('pearson', 'spearman', 'kendall'):
            raise ValueError('Unknown correlation method.')
        if not 0 < self.p_threshold <= 1 or (self.corr_threshold is not None and not 0 <= self.corr_threshold <= 1):
            raise ValueError('Invalid p-value or correlation threshold.')
        self.feature_names_in_ = list(X.columns)
        self.medians_ = X.median()
        keep = X.notna().all() if self.missing == 'drop' else X.notna().any()
        self.usable_ = list(X.columns[keep])
        X = X[self.usable_].fillna(self.medians_)
        self.usable_ = list(X.columns[X.var(ddof=0) > 0])
        X = X[self.usable_]
        if X.empty:
            raise ValueError('No nonconstant features remain after missing-value filtering.')
        self.mean_, self.std_ = X.mean(), X.std(ddof=0)
        pvalues = []
        for column in X:
            a, b = X.loc[y == 0, column], X.loc[y == 1, column]
            if self.univariate == 'mannwhitney':
                p = stats.mannwhitneyu(a, b, alternative='two-sided').pvalue
            elif self.univariate == 'welch':
                p = stats.ttest_ind(a, b, equal_var=False).pvalue
            elif self.univariate is None:
                p = 1.0
            else:
                raise ValueError('univariate must be mannwhitney, welch, or None.')
            pvalues.append(float(p) if np.isfinite(p) else 1.0)
        adjusted = multipletests(pvalues, method=self.p_adjust)[1] if self.p_adjust else pvalues
        self.univariate_results_ = pd.DataFrame({'feature': self.usable_, 'p_value': pvalues, 'adjusted_p_value': adjusted})
        columns = [c for c, p in zip(self.usable_, adjusted) if self.univariate is None or p < self.p_threshold]
        if not columns:
            raise ValueError('No features pass the univariate threshold. Adjust it explicitly.')
        self.correlation_ = X[columns].corr(method=self.corr_method)
        # Stable input-column order; use absolute correlation, including anticorrelation.
        selected = []
        for column in columns:
            if self.corr_threshold is None or not any(abs(self.correlation_.loc[column, other]) > self.corr_threshold for other in selected):
                selected.append(column)
        self.selected_features_ = selected
        return self

    def transform(self, X):
        X = numeric_frame(X)
        if not set(self.selected_features_).issubset(X.columns):
            raise ValueError('Prediction data are missing selected features.')
        X = X[self.selected_features_]
        if self.missing == 'median':
            X = X.fillna(self.medians_)
        elif X.isna().any().any():
            raise ValueError('Prediction features contain missing values; train with missing="median".')
        if self.scale == 'standard':
            X = (X - self.mean_[X.columns]) / self.std_[X.columns]
        return X


class FeatureSelector(_Filter):
    """Univariate -> absolute correlation -> optional squared-loss LASSO.

    alpha='cv' tunes the ENTIRE preprocessing/LASSO pipeline inside training data.
    Outer model CV clones this selector, preventing held-out feature-selection leakage.
    The LASSO linear score is not a probability or a logistic model.
    """
    def __init__(self, scale='standard', missing='drop', univariate='mannwhitney',
                 p_threshold=0.05, p_adjust=None, corr_method='pearson', corr_threshold=0.9,
                 lasso=True, alpha='cv', cv=5, coef_threshold=1e-6, random_state=0,
                 alphas=None):
        super().__init__(scale, missing, univariate, p_threshold, p_adjust, corr_method, corr_threshold)
        self.lasso = lasso
        self.alpha = alpha
        self.cv = cv
        self.coef_threshold = coef_threshold
        self.random_state = random_state
        self.alphas = alphas

    def fit(self, X, y):
        X = numeric_frame(X)
        y = binary_target(y)
        params = {key: getattr(self, key) for key in _Filter().get_params()}
        self.lasso_model_ = None
        self.lasso_cv_results_ = pd.DataFrame()
        self.lasso_path_ = None
        if self.lasso:
            pipe = Pipeline([('filter', _Filter(**params)), ('lasso', Lasso(max_iter=10000))])
            if self.alpha == 'cv':
                if self.cv < 2 or np.bincount(y).min() < self.cv:
                    raise ValueError('Each class needs at least cv training samples for LASSO tuning.')
                grid = np.asarray(self.alphas if self.alphas is not None else np.logspace(-4, -0.5, 16))
                cv = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
                search = GridSearchCV(pipe, {'lasso__alpha': grid}, scoring='neg_mean_squared_error', cv=cv, error_score='raise')
                search.fit(X, y)
                pipe = search.best_estimator_
                self.lasso_cv_results_ = pd.DataFrame({'alpha': search.cv_results_['param_lasso__alpha'].astype(float),
                    'mean_mse': -search.cv_results_['mean_test_score'], 'std_mse': search.cv_results_['std_test_score']})
            else:
                pipe.set_params(lasso__alpha=float(self.alpha)).fit(X, y)
            fitted_filter = pipe.named_steps['filter']
            self.lasso_model_ = pipe.named_steps['lasso']
        else:
            fitted_filter = _Filter(**params).fit(X, y)
        for key, value in vars(fitted_filter).items():
            if key.endswith('_'):
                setattr(self, key, value)
        self.pre_lasso_features_ = list(self.selected_features_)
        if self.lasso:
            Z = fitted_filter.transform(X)
            path_alpha, coefs, _ = lasso_path(np.asarray(Z - Z.mean()), y - y.mean(), n_alphas=50)
            self.lasso_path_ = {'alphas': path_alpha, 'coefficients': coefs, 'features': list(Z)}
            self.coefficients_ = pd.Series(self.lasso_model_.coef_, index=Z.columns, name='coefficient')
            self.selected_features_ = self.coefficients_.index[abs(self.coefficients_) > self.coef_threshold].tolist()
            if not self.selected_features_:
                raise ValueError('LASSO selected zero features. Adjust alpha explicitly.')
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)

    def linear_score(self, X):
        """LASSO intercept + retained weighted standardized features (not probability)."""
        if self.lasso_model_ is None:
            raise ValueError('No LASSO model was fitted.')
        return self.transform(X).dot(self.coefficients_.loc[self.selected_features_]) + self.lasso_model_.intercept_

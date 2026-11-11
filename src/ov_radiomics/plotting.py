"""Matplotlib/seaborn plots; no file writes or model refits while plotting."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit, logit
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
from ._validation import probabilities


def _axis(ax=None, figsize=(6, 4), title=None):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)
    return ax


def _scores(scores):
    return scores if isinstance(scores, (dict, pd.DataFrame)) else {'Model': scores}


def plot_roc(y_true, scores, ax=None, figsize=(6, 5), title='ROC', palette=None):
    ax = _axis(ax, figsize, title)
    for i, (name, score) in enumerate(_scores(scores).items()):
        y, p = probabilities(y_true, score)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y,p):.3f})', color=None if palette is None else palette[i])
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1.02))
    ax.legend()
    return ax


def plot_calibration(y_true, scores, n_bins=10, strategy='quantile', ax=None, figsize=(6, 5), title='Calibration'):
    ax = _axis(ax, figsize, title)
    for name, score in _scores(scores).items():
        y, p = probabilities(y_true, score)
        observed, predicted = calibration_curve(y, p, n_bins=n_bins, strategy=strategy)
        ax.plot(predicted, observed, 'o-', label=name)
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='Mean predicted probability', ylabel='Observed proportion', xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    return ax


def plot_dca(curve, ax=None, figsize=(6, 5), title='Decision curve', ylim=None):
    ax = _axis(ax, figsize, title)
    for name, group in curve.groupby('model', sort=False):
        ax.plot(group.threshold, group.net_benefit, label=name)
    ax.set(xlabel='Threshold probability', ylabel='Net benefit')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    return ax


def plot_confusion_matrix(y_true, y_score, threshold=0.5, class_names=('Negative', 'Positive'), normalize=False,
                          ax=None, figsize=(5, 4), title='Confusion matrix', cmap='Blues'):
    import seaborn as sns
    y, p = probabilities(y_true, y_score)
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be between 0 and 1.')
    matrix = confusion_matrix(y, p >= threshold, labels=[0, 1], normalize='true' if normalize else None)
    ax = _axis(ax, figsize, title)
    sns.heatmap(matrix, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
        xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(xlabel='Predicted', ylabel='Observed')
    return ax


def plot_prediction_scores(y_true, y_score, threshold=0.5, ax=None, figsize=(8, 4), title='Sample probabilities'):
    y, p = probabilities(y_true, y_score)
    order = np.argsort(p)
    ax = _axis(ax, figsize, title)
    for cls in (0, 1):
        where = np.flatnonzero(y[order] == cls)
        ax.bar(where, p[order][where], label=f'Class {cls}')
    ax.axhline(threshold, linestyle='--', color='gray')
    ax.set(xlabel='Samples sorted by probability', ylabel='Positive probability')
    ax.legend()
    return ax


def plot_correlation(correlation, cluster=False, ax=None, figsize=(8, 6), cmap='vlag', annot=False, title='Feature correlation'):
    import seaborn as sns
    if cluster:
        if ax is not None:
            raise ValueError('Clustered heatmaps own their figure; do not supply ax.')
        grid = sns.clustermap(correlation, cmap=cmap, vmin=-1, vmax=1, figsize=figsize, annot=annot)
        grid.fig.suptitle(title)
        return grid
    ax = _axis(ax, figsize, title)
    sns.heatmap(correlation, vmin=-1, vmax=1, cmap=cmap, annot=annot, ax=ax)
    return ax


def plot_lasso_path(selector, ax=None, figsize=(7, 5), title='LASSO coefficient path'):
    if selector.lasso_path_ is None:
        raise ValueError('Selector has no fitted LASSO path.')
    ax = _axis(ax, figsize, title)
    path = selector.lasso_path_
    ax.semilogx(path['alphas'], path['coefficients'].T)
    ax.axvline(selector.lasso_model_.alpha, color='gray', linestyle='--')
    ax.set(xlabel='Alpha', ylabel='Coefficient')
    return ax


def plot_lasso_cv(selector, ax=None, figsize=(7, 5), title='LASSO CV error'):
    table = selector.lasso_cv_results_
    if table.empty:
        raise ValueError('Fit selector with alpha="cv" first.')
    ax = _axis(ax, figsize, title)
    ax.errorbar(table.alpha, table.mean_mse, yerr=table.std_mse, fmt='o-')
    ax.set(xscale='log', xlabel='Alpha', ylabel='MSE (mean +/- fold SD)')
    return ax


def plot_feature_weights(weights, ax=None, figsize=(7, 5), title='Feature weights'):
    weights = pd.Series(weights).sort_values()
    ax = _axis(ax, figsize, title)
    ax.barh(weights.index.astype(str), weights.values)
    ax.set_xlabel('Weight / importance')
    return ax


def plot_cv_results(results, ax=None, figsize=(8, 5), title='Cross-validation AUC'):
    import seaborn as sns
    ax = _axis(ax, figsize, title)
    sns.boxplot(data=results, x='model', y='auc', ax=ax)
    sns.stripplot(data=results, x='model', y='auc', color='black', ax=ax)
    ax.set_ylim(0, 1)
    return ax


def plot_metrics(metrics, metric='auc', kind='bar', ax=None, figsize=(8, 5), title=None):
    import seaborn as sns
    ax = _axis(ax, figsize, title or metric)
    if kind not in ('bar', 'line'):
        raise ValueError('kind must be bar or line.')
    function = sns.barplot if kind == 'bar' else sns.lineplot
    function(data=metrics, x='model', y=metric, hue='group' if 'group' in metrics else None, ax=ax)
    return ax


def plot_feature_summary(features, kind='bar', ax=None, figsize=(7, 5), title='Radiomics feature families'):
    columns = list(features.columns) if isinstance(features, pd.DataFrame) else list(features)
    counts = pd.Series([c.split('_')[-2] for c in columns if len(c.split('_')) >= 3 and not c.startswith('diagnostics_')]).value_counts()
    ax = _axis(ax, figsize, title)
    if kind == 'pie':
        ax.pie(counts, labels=counts.index, autopct='%.1f%%')
    elif kind == 'bar':
        ax.bar(counts.index, counts.values)
        ax.set_ylabel('Feature count')
    else:
        raise ValueError('kind must be bar or pie.')
    return ax


def plot_feature_pvalues(table, ax=None, figsize=(9, 5), title='Univariate feature p-values'):
    import seaborn as sns
    frame = table.copy()
    frame['family'] = frame.feature.map(lambda name: name.split('_')[-2] if len(name.split('_')) >= 3 else 'other')
    ax = _axis(ax, figsize, title)
    sns.violinplot(data=frame, x='family', y='p_value', cut=0, ax=ax)
    sns.stripplot(data=frame, x='family', y='p_value', color='black', ax=ax)
    return ax


def plot_embedding(X, y=None, method='pca', random_state=0, ax=None, figsize=(6, 5), **params):
    """Exploratory embedding only; never a model validation statistic."""
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding, SpectralEmbedding
    from sklearn.random_projection import GaussianRandomProjection
    factories = {'pca': PCA, 'svd': TruncatedSVD, 'tsne': TSNE, 'isomap': Isomap,
        'mds': MDS, 'lle': LocallyLinearEmbedding, 'spectral': SpectralEmbedding, 'random_projection': GaussianRandomProjection}
    if method not in factories:
        raise ValueError(f'Unknown embedding; choose from {list(factories)}.')
    estimator = factories[method](n_components=2, **params)
    if 'random_state' in estimator.get_params():
        estimator.set_params(random_state=random_state)
    coordinates = estimator.fit_transform(X)
    ax = _axis(ax, figsize, method.upper() + ' (exploratory)')
    colors = None if y is None else pd.factorize(y)[0]
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, cmap='coolwarm' if colors is not None else None)
    ax.set(xlabel='Component 1', ylabel='Component 2')
    return ax


def plot_learning_curve(estimator, X, y, cv=5, train_sizes=None, scoring='roc_auc', random_state=0, ax=None, figsize=(7, 5)):
    """Use training data and an unfitted full pipeline, including feature selection."""
    from sklearn.model_selection import learning_curve, StratifiedKFold
    sizes, train, valid = learning_curve(estimator, X, y,
        cv=StratifiedKFold(cv, shuffle=True, random_state=random_state),
        train_sizes=np.linspace(.4, 1, 4) if train_sizes is None else train_sizes,
        scoring=scoring, error_score='raise', shuffle=True, random_state=random_state)
    ax = _axis(ax, figsize, 'Learning curve')
    for name, values in [('Train', train), ('Validation', valid)]:
        ax.errorbar(sizes, values.mean(axis=1), yerr=values.std(axis=1), label=name)
    ax.set(xlabel='Training samples', ylabel=scoring)
    ax.legend()
    return ax


def plot_nomogram(fusion, risk_ticks=None, figsize=(10, 6), dpi=100):
    """Additive point nomogram for a fitted LR fusion of probabilities in [0,1].

    Each predictor contributes nonnegative points; their sum maps exactly to the
    fitted logistic probability. No fitting occurs here.
    """
    coefficients = np.asarray(fusion.model.coef_).reshape(-1)
    if len(coefficients) != len(fusion.score_columns) or not np.any(coefficients):
        raise ValueError('Expected a nonconstant binary logistic fusion model.')
    unit = max(abs(coefficients))/100
    baseline = float(fusion.model.intercept_[0] + np.minimum(coefficients, 0).sum())
    maximum = float(abs(coefficients).sum()/unit)
    if risk_ticks is None:
        risk_ticks = expit(np.linspace(baseline, baseline+maximum*unit, 5))
    risk_ticks = np.asarray(risk_ticks, dtype=float)
    if not np.isfinite(risk_ticks).all() or ((risk_ticks <= 0) | (risk_ticks >= 1)).any():
        raise ValueError('Risk ticks must lie strictly between 0 and 1.')
    fig, axes = plt.subplots(len(coefficients)+3, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    axes[0].plot([0, 100], [0, 0], color='black')
    axes[0].set_xticks(np.linspace(0,100,6))
    axes[0].set(xlim=(0, maximum), yticks=[], ylabel='Points')
    for ax, name, beta in zip(axes[1:-2], fusion.score_columns, coefficients):
        values = np.linspace(0, 1, 6)
        points = (beta*values-min(beta, 0))/unit
        ax.plot([points.min(), points.max()], [0, 0], color='black')
        ax.set_xticks(points)
        ax.set_xticklabels([f'{v:.1f}' for v in values])
        ax.set(xlim=(0, maximum), yticks=[], ylabel=name)
    axes[-2].plot([0, maximum], [0, 0], color='black')
    axes[-2].set(xlim=(0, maximum), yticks=[], ylabel='Total points')
    positions = (logit(risk_ticks)-baseline)/unit
    inside = (positions >= 0) & (positions <= maximum)
    axes[-1].plot([0, maximum], [0, 0], color='black')
    axes[-1].set_xticks(positions[inside])
    axes[-1].set_xticklabels([f'{p:.2f}' for p in risk_ticks[inside]])
    axes[-1].set(xlim=(0, maximum), yticks=[], ylabel='Risk')
    for ax in axes:
        ax.spines['bottom'].set_position(('data', 0))
        ax.set_ylim(-1, 1)
        for spine in ('left', 'right', 'top'):
            ax.spines[spine].set_visible(False)
    # Expose the exact conversion for validation and downstream use.
    fig.nomogram_mapping = {'baseline_logit': baseline, 'logit_per_point': unit, 'max_points': maximum}
    return fig

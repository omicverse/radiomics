"""Shared radiomics/clinical training with fixed holdout and out-of-fold predictions."""
from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from ._validation import binary_target, unique_ids
from .metrics import binary_metrics


def split_data(data, target, id_col='ID', test_size=0.2, stratify=True, random_state=0, split='group'):
    """One row per independent patient. Retain a supplied split instead of calling this."""
    unique_ids(data, id_col)
    if split in data:
        raise ValueError(f'{split} already exists; preserve it or explicitly remove it first.')
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between 0 and 1.')
    binary_target(data[target])
    train, test = train_test_split(np.arange(len(data)), test_size=test_size,
        stratify=data[target] if stratify else None, random_state=random_state)
    result = data.copy()
    result[split] = 'train'
    result.iloc[test, result.columns.get_loc(split)] = 'test'
    return result


def make_classifier(model='LR', random_state=0, **model_params):
    """Create an estimator. XGBoost and LightGBM import only when requested."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    factories = {
        'LR': (LogisticRegression, {'max_iter': 2000, 'random_state': random_state}),
        'SVM': (SVC, {'probability': True, 'random_state': random_state}),
        'KNN': (KNeighborsClassifier, {}),
        'RandomForest': (RandomForestClassifier, {'n_estimators': 200, 'random_state': random_state, 'n_jobs': 1}),
        'ExtraTrees': (ExtraTreesClassifier, {'n_estimators': 200, 'random_state': random_state, 'n_jobs': 1}),
        'MLP': (MLPClassifier, {'hidden_layer_sizes': (32,), 'max_iter': 2000, 'random_state': random_state}),
    }
    if model == 'XGBoost':
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError('Install ov-radiomics[boost] for XGBoost.') from exc
        factory, params = XGBClassifier, {'n_estimators': 100, 'max_depth': 3, 'eval_metric': 'logloss', 'random_state': random_state, 'n_jobs': 1}
    elif model == 'LightGBM':
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError('Install ov-radiomics[boost] for LightGBM.') from exc
        factory, params = LGBMClassifier, {'n_estimators': 100, 'verbosity': -1, 'random_state': random_state, 'n_jobs': 1}
    elif model in factories:
        factory, params = factories[model]
    else:
        raise ValueError(f'Unknown model {model}. Use LR, SVM, KNN, RandomForest, ExtraTrees, MLP, XGBoost, LightGBM.')
    params.update(model_params)
    return factory(**params)


def _preprocessor(features, continuous, categorical, scale):
    categorical = list(categorical or [])
    continuous = list(continuous) if continuous is not None else [c for c in features if c not in categorical]
    if set(continuous) & set(categorical) or set(continuous+categorical) != set(features):
        raise ValueError('continuous and categorical must partition the feature names.')
    if scale not in (None, 'standard'):
        raise ValueError('scale must be standard or None.')
    blocks = []
    if continuous:
        steps = [('impute', SimpleImputer(strategy='median'))]
        if scale == 'standard':
            steps.append(('scale', StandardScaler()))
        blocks.append(('numeric', Pipeline(steps), continuous))
    if categorical:
        # Keep compatibility with sklearn 1.0 and current releases.
        try:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        blocks.append(('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', encoder)]), categorical))
    return ColumnTransformer(blocks)


@dataclass
class TrainingResult:
    models: dict
    cv_results: pd.DataFrame
    oof_predictions: pd.DataFrame
    predictions: pd.DataFrame
    features: list
    config: dict

    def predict_proba(self, data, model='LR'):
        return pd.Series(self.models[model].predict_proba(data[self.features])[:, 1], index=data.index, name=model)

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / 'training.joblib')
        for name in ('cv_results', 'oof_predictions', 'predictions'):
            getattr(self, name).to_csv(path / f'{name}.csv', index=False)
        (path / 'config.json').write_text(json.dumps(self.config, indent=2, default=str), encoding='utf-8')


def train_models(data, target, features, split='group', models=('LR',), selector=None,
                 continuous=None, categorical=None, scale=None, cv=5, scoring='roc_auc',
                 model_params=None, random_state=0, id_col='ID', positive_label=1,
                 patient_id=None, smote=False):
    """Fit cloneable pipelines using only train rows; test rows are prediction-only.

    Initial scope: one sample per independent patient. Repeated patients are rejected.
    OOF predictions refit preprocessing, feature selection and the estimator in each fold.
    """
    unique_ids(data, id_col)
    if patient_id is not None:
        unique_ids(data, patient_id)
    features = list(features)
    if not features or len(set(features)) != len(features) or set(features) & {target, split, id_col, patient_id}:
        raise ValueError('Features must be unique and exclude target, split and ID columns.')
    if not set(features).issubset(data.columns):
        raise ValueError('Missing feature columns.')
    if set(data[split]) != {'train', 'test'}:
        raise ValueError('Expected explicit nonempty train/test groups; unknown groups are not test data.')
    if scoring != 'roc_auc' or cv < 2 or not models or len(set(models)) != len(models):
        raise ValueError('Use scoring=roc_auc, cv>=2, and unique model names.')
    train = data.loc[data[split] == 'train'].copy()
    y = binary_target(train[target], positive_label)
    all_classes = set(train[target])
    if data[target].isna().any() or not set(data[target]).issubset(all_classes):
        raise ValueError('All targets must use the two training classes.')
    if np.bincount(y).min() < cv:
        raise ValueError('Each training class needs at least cv samples.')
    if data[features].select_dtypes(include='number').isin([np.inf, -np.inf]).any().any():
        raise ValueError('Feature values contain infinity.')
    X = train[features]
    if X.isna().all().any():
        raise ValueError('A training feature is entirely missing.')
    cv_splitter = list(StratifiedKFold(cv, shuffle=True, random_state=random_state).split(X, y))
    params = model_params or {}
    if set(params) - set(models):
        raise ValueError('model_params contains a model not requested in models.')
    fitted, cv_rows, prediction_rows, oof_rows = {}, [], [], []
    for name in models:
        if selector is not None and categorical:
            raise ValueError('FeatureSelector accepts numeric radiomics only; use clinical preprocessing for categories.')
        preprocessor = clone(selector) if selector is not None else _preprocessor(features, continuous, categorical, scale)
        steps = [('preprocess', preprocessor)]
        if smote:
            if categorical:
                raise ValueError('Plain SMOTE is unsupported for categorical clinical inputs.')
            try:
                from imblearn.over_sampling import SMOTE
                from imblearn.pipeline import Pipeline as TrainingPipeline
            except ImportError as exc:
                raise ImportError('Install ov-radiomics[smote] to use SMOTE.') from exc
            steps.append(('smote', SMOTE(random_state=random_state)))
        else:
            TrainingPipeline = Pipeline
        steps.append(('classifier', make_classifier(name, random_state, **params.get(name, {}))))
        pipeline = TrainingPipeline(steps)
        for fold, (train_idx, valid_idx) in enumerate(cv_splitter):
            current = clone(pipeline).fit(X.iloc[train_idx], y[train_idx])
            probability = current.predict_proba(X.iloc[valid_idx])[:, 1]
            cv_rows.append({'model': name, 'fold': fold, 'auc': roc_auc_score(y[valid_idx], probability)})
            for position, p in zip(valid_idx, probability):
                oof_rows.append({id_col: train.iloc[position][id_col], 'y_true': int(y[position]),
                    'model': name, 'probability': p, 'fold': fold, 'prediction_type': 'oof'})
        pipeline.fit(X, y)
        fitted[name] = pipeline
        for group in ('train', 'test'):
            subset = data.loc[data[split] == group]
            probability = pipeline.predict_proba(subset[features])[:, 1]
            for identifier, target_value, p in zip(subset[id_col], subset[target], probability):
                prediction_rows.append({id_col: identifier, 'y_true': int(target_value == positive_label),
                    'model': name, 'probability': p, 'group': group,
                    'prediction_type': 'apparent' if group == 'train' else 'holdout'})
    config = {'target': target, 'split': split, 'id_col': id_col, 'positive_label': positive_label,
              'random_state': random_state, 'cv': cv, 'features': features,
              'model_params': {name: model.get_params() for name, model in fitted.items()}}
    return TrainingResult(fitted, pd.DataFrame(cv_rows), pd.DataFrame(oof_rows), pd.DataFrame(prediction_rows), features, config)


def evaluate(predictions, threshold=0.5, ci_method='delong', confidence=0.95):
    """Summarize standardized long predictions by model and cohort."""
    frame = predictions.predictions if isinstance(predictions, TrainingResult) else predictions
    groups = ['model'] + (['group'] if 'group' in frame else [])
    rows = []
    for keys, subset in frame.groupby(groups):
        keys = keys if isinstance(keys, tuple) else (keys,)
        result = binary_metrics(subset.y_true, subset.probability, threshold=threshold, ci_method=ci_method, confidence=confidence).to_dict()
        result.update(dict(zip(groups, keys)))
        rows.append(result)
    return pd.DataFrame(rows)


def fusion_data(results, model='LR', cohort='oof'):
    """Strictly align named TrainingResults into fusion input, e.g. {'Rad': rad, 'Clinical': clinic}."""
    merged = None
    for name, result in results.items():
        if name in ('ID', 'y_true', 'prediction_type', 'fold'):
            raise ValueError('Reserved fusion score name.')
        if result.config['id_col'] != 'ID':
            raise ValueError('fusion_data currently requires id_col="ID".')
        selected = result.oof_predictions if cohort == 'oof' else result.predictions.loc[result.predictions.group == cohort]
        selected = selected.loc[selected.model == model].copy()
        if selected.empty:
            raise ValueError(f'No {model}/{cohort} predictions for {name}.')
        unique_ids(selected)
        columns = ['ID', 'y_true', 'prediction_type'] + (['fold'] if cohort == 'oof' else [])
        selected = selected[columns + ['probability']].rename(columns={'probability': name})
        if merged is None:
            merged = selected
        else:
            if set(merged.ID) != set(selected.ID):
                raise ValueError('Fusion components have different patient sets.')
            checked = merged[columns].merge(selected[columns], on='ID', suffixes=('_a', '_b'))
            for column in columns[1:]:
                if not (checked[column+'_a'] == checked[column+'_b']).all():
                    raise ValueError(f'Fusion {column} differs across components.')
            merged = merged.merge(selected[['ID', name]], on='ID', validate='one_to_one')
    if merged is None:
        raise ValueError('Supply at least one training result.')
    return merged


@dataclass
class FusionResult:
    model: object
    score_columns: list

    def predict_proba(self, data):
        values = data[self.score_columns].astype(float)
        if not np.isfinite(values).all().all() or ((values < 0) | (values > 1)).any().any():
            raise ValueError('Fusion inputs must be valid probabilities.')
        return pd.Series(self.model.predict_proba(values)[:, 1], index=data.index, name='Fusion')


def fit_fusion(oof_predictions, target='y_true', score_columns=('Rad', 'Clinical'), model='LR', random_state=0):
    """Fit logistic probability fusion from OOF base predictions, never holdout labels."""
    if model != 'LR':
        raise ValueError('Initial fusion/nomogram interface supports Logistic Regression only.')
    unique_ids(oof_predictions)
    if 'prediction_type' not in oof_predictions or not (oof_predictions.prediction_type == 'oof').all():
        raise ValueError('Fusion training requires explicitly marked OOF predictions.')
    if target in score_columns or len(set(score_columns)) != len(score_columns):
        raise ValueError('Fusion score columns must be unique and exclude the target.')
    X = oof_predictions[list(score_columns)].astype(float)
    if not np.isfinite(X).all().all() or ((X < 0) | (X > 1)).any().any():
        raise ValueError('Fusion inputs must be finite probabilities.')
    fitted = make_classifier(model, random_state).fit(X, binary_target(oof_predictions[target]))
    return FusionResult(fitted, list(score_columns))

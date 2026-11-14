# API reference

Import all functions with `import ov_radiomics as rad`. Signatures below are generated from the implementation.

## Input and result contracts

- `samples`: a DataFrame with unique, nonmissing `ID`, `image`, `mask`; optional per-row `roi_label`.
- Paths in `samples` are used as provided; `match_images` returns absolute paths. `ID` retains the exact filename, including `.nii.gz`.
- `extract(params=...)`: PyRadiomics YAML path or dictionary with `imageType`, `featureClass`, `setting`. `roi_label` and `correct_mask` are explicit call-time controls. `params=None` uses and records backend defaults. It does not silently read a local YAML.
- `ExtractionResult.features`: ID + numeric feature columns. Diagnostics are stored separately. With `on_error="record"`, failed IDs remain in the result and carry missing feature values. Review `.diagnostics` before modeling.
- `data`: one row per independent patient. The initial workflow accepts exactly `train` and `test` in the split column. Labels default to binary 0/1; `train_models(positive_label=...)` supports another positive label and stores predictions as 0/1.
- `features`: explicitly selected column names, excluding ID, outcome, and split. No automatic target guessing.
- `FeatureSelector.fit(X_train,y_train)` returns the fitted selector; `transform(X)` applies stored training parameters. `get_feature_names_out()` exposes retained names. `linear_score()` is a squared-loss LASSO score, not probability.
- `FeatureSelector(missing="drop")` drops columns with training missingness; missing values later appearing in retained columns raise an error. `missing="median"` learns and reuses training medians. Infinite feature values are treated as missing by the selector; the high-level training function rejects infinity at input.
- Correlation filtering uses `abs(correlation)>corr_threshold`. Ties are resolved by stable input-column order, not outcome-based ranking. Set `corr_threshold=None` to disable.
- `univariate=None` disables p-value screening; otherwise choose `mannwhitney` or `welch`. `p_adjust=None` means unadjusted screening; e.g. `p_adjust="fdr_bh"` requests a statsmodels-supported correction. These screening tests are not confirmatory findings.
- `alpha="cv"` tunes the full filtering+LASSO pipeline using an explicit grid (default 16 log-spaced values from 1e-4 to about 0.316). `alphas=` replaces the grid. Numeric `alpha` fixes it. Empty selections raise errors rather than silently changing criteria.
- `baseline_table`: observed-data summaries, numeric `p_value`, test name, counts/missingness. `split=` stratifies tables; it does not compare cohorts to one another.
- Logistic inference: complete cases, binary target 0/1, explicit reference categories, Wald OR confidence intervals. The fitted statsmodels result is available through `.model`. No automatic imputation, stepwise selection or multiplicity correction.
- `TrainingResult`: `.models` (full pipelines), `.cv_results`, `.oof_predictions`, `.predictions`, `.features`, `.config`. `.predict_proba(data, model=...)` uses original feature columns. `.save(directory)` persists the whole result with joblib and tabular outputs. Only load trusted joblib files.
- Training predictions use long columns `ID, y_true, model, probability, group, prediction_type`. OOF rows instead include `fold`. CV AUC is a fraction in [0,1]. Models are not automatically ranked or selected by test results.
- `fusion_data` strictly aligns patients, outcomes and OOF fold IDs. It currently requires `ID` as the ID column. Use the same CV seed, folds and row order for both components.
- `fit_fusion` accepts marked OOF predictions and trains LR on those score columns. The caller is responsible for truthful provenance when supplying custom tables. `.predict_proba()` needs only the named probability columns.
- `binary_metrics`: one 1D positive-class probability vector. Threshold is explicit and never optimized on the supplied outcomes. CI methods: `delong`, stratified `bootstrap`, or `None`. Bootstrap is patient-level only under this workflow's independent-row assumption.
- Metric functions and `evaluate` accept `backend="auto"`, `"rust"`, or `"python"`. Auto uses a compiled extension when available; an absent extension permits Python fallback. A broken DLL does not. `backend_info()` reports the selected default. Distribution functions, quantile grouping and bootstrap random draws remain Python for reproducibility.
- `delong_test` returns ordinary two-sided `p_value` and `z`. Identical scores return p=1; a nonzero difference with zero estimated variance returns p=0. DeLong needs at least two observations per class.
- `decision_curve`: thresholds strictly inside (0,1); includes treat-all and treat-none baselines. No prevalence correction for case-control sampling is performed.
- `hosmer_lemeshow`: quantile groups with duplicate boundaries removed, df = actual groups - 2. Undefined groups raise errors.
- Plotting returns an Axes, except clustered correlation (seaborn ClusterGrid) and nomogram (Figure). No automatic output directory or global styling side effects. Save with `ax.figure.savefig(...)` / `figure.savefig(...)`.
- `plot_nomogram`: probability predictors range over [0,1]. Default risk ticks use the model's attainable range; explicit out-of-range ticks are omitted.
- `plot_embedding`: exploratory PCA/SVD/t-SNE/Isomap/MDS/LLE/spectral/random projection. It fits on supplied X; never interpret this display as test performance.
- `plot_learning_curve`: supply training data and an unfitted full pipeline. It clones and fits it in CV. SMOTE needs its optional dependency and is only allowed on numeric input.

## Classifier adapters

All model parameters can be overridden using `make_classifier(..., **model_params)` or per-model `train_models(model_params={...})`.

| Name | Explicit adapter defaults beyond backend defaults |
|---|---|
| LR | max_iter=2000, random_state |
| SVM | probability=True, random_state |
| KNN | sklearn defaults |
| RandomForest | n_estimators=200, n_jobs=1, random_state |
| ExtraTrees | n_estimators=200, n_jobs=1, random_state |
| MLP | hidden_layer_sizes=(32,), max_iter=2000, random_state |
| XGBoost | n_estimators=100, max_depth=3, eval_metric=logloss, n_jobs=1, random_state |
| LightGBM | n_estimators=100, verbosity=-1, n_jobs=1, random_state |


## Image input and extraction

### `match_images`

```python
match_images(root, image_dir='images', mask_dir='masks')
```

Pair .nii/.nii.gz files by exact filename. IDs retain the filename.

### `check_images`

```python
check_images(samples, roi_label=1, geometry_tolerance=1e-06)
```

Read each pair and report geometry mismatch and absent ROI without modifying files.

### `extract`

```python
extract(samples, params=None, roi_label=1, correct_mask=False, n_jobs=1, on_error='raise')
```

Extract ROI features. params is a PyRadiomics YAML path or parameter dictionary.

Explicit correct_mask overrides the backend setting. No implicit disk cache.
on_error='record' keeps failed IDs in diagnostics and features (NaN values).

### `join_data`

```python
join_data(left, right, on='ID')
```

Strict one-to-one join; never silently discard subjects or duplicate columns.


## Feature processing

### `FeatureSelector`

```python
FeatureSelector(scale='standard', missing='drop', univariate='mannwhitney', p_threshold=0.05, p_adjust=None, corr_method='pearson', corr_threshold=0.9, lasso=True, alpha='cv', cv=5, coef_threshold=1e-06, random_state=0, alphas=None)
```

Univariate -> absolute correlation -> optional squared-loss LASSO.

alpha='cv' tunes the ENTIRE preprocessing/LASSO pipeline inside training data.
Outer model CV clones this selector, preventing held-out feature-selection leakage.
The LASSO linear score is not a probability or a logistic model.


## Clinical statistics

### `baseline_table`

```python
baseline_table(data, target, features, continuous, categorical=None, split=None, continuous_test='mannwhitney', categorical_test='auto')
```

Compare outcome classes within each split. No imputation, encoding, or selection.

Continuous summaries: n, mean, SD, median, Q1, Q3. Categorical summaries:
counts and proportions among nonmissing observations, with missing counts separate.

### `univariate_logistic`

```python
univariate_logistic(data, target, features, categorical=None, reference=None, confidence=0.95)
```

### `multivariable_logistic`

```python
multivariable_logistic(data, target, features, categorical=None, reference=None, confidence=0.95)
```

Complete-case unpenalized Logit with explicit reference categories and Wald CIs.

No implicit stepwise selection. Perfect separation / rank deficiency are errors.


## Training and probability fusion

### `split_data`

```python
split_data(data, target, id_col='ID', test_size=0.2, stratify=True, random_state=0, split='group')
```

One row per independent patient. Retain a supplied split instead of calling this.

### `make_classifier`

```python
make_classifier(model='LR', random_state=0, **model_params)
```

Create an estimator. XGBoost and LightGBM import only when requested.

### `train_models`

```python
train_models(data, target, features, split='group', models=('LR',), selector=None, continuous=None, categorical=None, scale=None, cv=5, scoring='roc_auc', model_params=None, random_state=0, id_col='ID', positive_label=1, patient_id=None, smote=False)
```

Fit cloneable pipelines using only train rows; test rows are prediction-only.

Initial scope: one sample per independent patient. Repeated patients are rejected.
OOF predictions refit preprocessing, feature selection and the estimator in each fold.

### `fusion_data`

```python
fusion_data(results, model='LR', cohort='oof')
```

Strictly align named TrainingResults into fusion input, e.g. {'Rad': rad, 'Clinical': clinic}.

### `fit_fusion`

```python
fit_fusion(oof_predictions, target='y_true', score_columns=('Rad', 'Clinical'), model='LR', random_state=0)
```

Fit logistic probability fusion from OOF base predictions, never holdout labels.

### `evaluate`

```python
evaluate(predictions, threshold=0.5, ci_method='delong', confidence=0.95, backend='auto')
```

Summarize standardized long predictions by model and cohort.


## Evaluation

### `backend_info`

```python
backend_info()
```

Report the actual default metric backend; does not change process-wide settings.

### `binary_metrics`

```python
binary_metrics(y_true, y_score, threshold=0.5, positive_label=1, ci_method='delong', confidence=0.95, n_bootstrap=2000, random_state=0, backend='auto')
```

Return the existing metric Series; backend is auto, rust, or python.

Bootstrap draws retain the NumPy RandomState sequence for cross-backend parity.
Normal quantiles remain SciPy calculations; Rust supplies numerical kernels.

### `delong_test`

```python
delong_test(y_true, score_a, score_b, positive_label=1, backend='auto')
```

Two-sided paired test; rows must refer to the same independent subjects.

### `decision_curve`

```python
decision_curve(y_true, scores, thresholds=None, positive_label=1, backend='auto')
```

### `hosmer_lemeshow`

```python
hosmer_lemeshow(y_true, y_score, n_bins=10, positive_label=1, backend='auto')
```


## Figures

### `plot_roc`

```python
plot_roc(y_true, scores, ax=None, figsize=(6, 5), title='ROC', palette=None)
```

### `plot_calibration`

```python
plot_calibration(y_true, scores, n_bins=10, strategy='quantile', ax=None, figsize=(6, 5), title='Calibration')
```

### `plot_dca`

```python
plot_dca(curve, ax=None, figsize=(6, 5), title='Decision curve', ylim=None)
```

### `plot_confusion_matrix`

```python
plot_confusion_matrix(y_true, y_score, threshold=0.5, class_names=('Negative', 'Positive'), normalize=False, ax=None, figsize=(5, 4), title='Confusion matrix', cmap='Blues')
```

### `plot_prediction_scores`

```python
plot_prediction_scores(y_true, y_score, threshold=0.5, ax=None, figsize=(8, 4), title='Sample probabilities')
```

### `plot_correlation`

```python
plot_correlation(correlation, cluster=False, ax=None, figsize=(8, 6), cmap='vlag', annot=False, title='Feature correlation')
```

### `plot_lasso_path`

```python
plot_lasso_path(selector, ax=None, figsize=(7, 5), title='LASSO coefficient path')
```

### `plot_lasso_cv`

```python
plot_lasso_cv(selector, ax=None, figsize=(7, 5), title='LASSO CV error')
```

### `plot_feature_weights`

```python
plot_feature_weights(weights, ax=None, figsize=(7, 5), title='Feature weights')
```

### `plot_cv_results`

```python
plot_cv_results(results, ax=None, figsize=(8, 5), title='Cross-validation AUC')
```

### `plot_metrics`

```python
plot_metrics(metrics, metric='auc', kind='bar', ax=None, figsize=(8, 5), title=None)
```

### `plot_feature_summary`

```python
plot_feature_summary(features, kind='bar', ax=None, figsize=(7, 5), title='Radiomics feature families')
```

### `plot_feature_pvalues`

```python
plot_feature_pvalues(table, ax=None, figsize=(9, 5), title='Univariate feature p-values')
```

### `plot_embedding`

```python
plot_embedding(X, y=None, method='pca', random_state=0, ax=None, figsize=(6, 5), **params)
```

Exploratory embedding only; never a model validation statistic.

### `plot_learning_curve`

```python
plot_learning_curve(estimator, X, y, cv=5, train_sizes=None, scoring='roc_auc', random_state=0, ax=None, figsize=(7, 5))
```

Use training data and an unfitted full pipeline, including feature selection.

### `plot_nomogram`

```python
plot_nomogram(fusion, risk_ticks=None, figsize=(10, 6), dpi=100)
```

Additive point nomogram for a fitted LR fusion of probabilities in [0,1].

Each predictor contributes nonnegative points; their sum maps exactly to the
fitted logistic probability. No fitting occurs here.

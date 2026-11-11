"""Generate exact public signatures plus hand-maintained parameter contracts."""
import inspect
from pathlib import Path
import ov_radiomics as rad

ROOT = Path(__file__).resolve().parents[1]
groups = {
    'Image input and extraction':['match_images','check_images','extract','join_data'],
    'Feature processing':['FeatureSelector'],
    'Clinical statistics':['baseline_table','univariate_logistic','multivariable_logistic'],
    'Training and probability fusion':['split_data','make_classifier','train_models','fusion_data','fit_fusion','evaluate'],
    'Evaluation':['binary_metrics','delong_test','decision_curve','hosmer_lemeshow'],
    'Figures':[name for name in vars(rad) if name.startswith('plot_')],
}
text = '''# API reference

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

'''
for group,names in groups.items():
    text += '\n## '+group+'\n\n'
    for name in names:
        obj = getattr(rad,name)
        text += '### `'+name+'`\n\n```python\n'+name+str(inspect.signature(obj))+'\n```\n\n'
        if inspect.getdoc(obj):
            text += inspect.getdoc(obj)+'\n\n'
(ROOT/'docs'/'api.md').write_text(text.rstrip()+'\n',encoding='utf-8')

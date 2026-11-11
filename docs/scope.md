# Initial scope: single-center radiomics plus clinical modeling

The functional requirements were derived by inspecting four user-supplied solution notebooks, their configuration, and example extraction settings. The implementation does not import, decrypt, or redistribute those notebooks or their runtime.

| Source workflow | Implemented API |
|---|---|
| Pair images and masks, diagnose geometry | `match_images`, `check_images` |
| Conventional radiomics, optional YAML | `extract`; Original/filter/feature/settings configuration passes to PyRadiomics |
| Join label and clinical records | `join_data`: strict one-to-one matching |
| Z-score, missing values, univariate tests | `FeatureSelector`; separate raw clinical `baseline_table` |
| Pearson/Spearman/Kendall filtering | `FeatureSelector(corr_method=..., corr_threshold=...)` using absolute correlation |
| LASSO CV, paths, formula, weights | `FeatureSelector`, `linear_score`, `plot_lasso_path`, `plot_lasso_cv`, `plot_feature_weights` |
| Eight models | `make_classifier`, `train_models` |
| Shared clinical train/test split | `split_data` once; reuse `group` |
| Baseline, single-/multivariable Logit | `baseline_table`, `univariate_logistic`, `multivariable_logistic` |
| Clinical + radiomics probability combination | `fusion_data`, `fit_fusion` |
| Binary metrics, paired AUC tests | `binary_metrics`, `evaluate`, `delong_test` |
| ROC, DCA, calibration, H–L | `plot_roc`, `decision_curve`, `plot_dca`, `plot_calibration`, `hosmer_lemeshow` |
| Confusion matrix, patient probabilities | `plot_confusion_matrix`, `plot_prediction_scores` |
| Feature categories, p-value distributions, clustered heatmaps | `plot_feature_summary`, `plot_feature_pvalues`, `plot_correlation` |
| Model metric/CV plots | `plot_metrics`, `plot_cv_results` |
| Risk nomogram | `plot_nomogram`: fitted LR coefficients mapped to additive points |
| Optional SMOTE, learning curves | `train_models(smote=True)`, `plot_learning_curve` |
| Exploratory dimensionality reduction | `plot_embedding`: PCA, SVD, t-SNE, Isomap, MDS, LLE, spectral, random projection |

## Deliberate behavior changes

1. The original Step1 assigns a CT YAML path and immediately sets it to `None`. This package records the **effective** PyRadiomics settings, filters, features and version; it does not assume the unused CT YAML was active.
2. Scaling, screening and LASSO are fitted on training data, including in CV folds. The original globally fitted selection is not reproduced.
3. Fixed holdout and all-fold CV replace selection of the best-performing split.
4. Step3's existing fixed clinical split is preserved. A non-`train` string is not automatically accepted as test.
5. Step4's fitting of a second fusion model to test labels is replaced by prediction with the trained model.
6. OOF training probabilities replace base-model in-sample probabilities for fusion fitting.
7. P-values remain numeric. No parsing of formatted strings or implicit loss of a leading character.
8. No blanket warning suppression, global environment changes, or silent drop of unmatched patients.
9. Classifier tuning and plotting are separate; plotting never fits a new fusion model.
10. Test labels do not choose thresholds. The default is an explicit 0.5.

## Intentionally outside this first implementation

Multicenter harmonization, ICC, survival, DICOM conversion, automated segmentation, deep features, habitats, repeated lesions/scans per patient, automatic stepwise inference, and paper generation. NCA/random-tree embeddings from the broad embedding menu are not included. The independent package is ready for a later upstream OmicVerse adapter; it is not merged into that repository.

## Defaults versus source settings

- `correct_mask=False` is conservative; explicitly set `True` to request backend mask correction. Correction cannot make unrelated coordinate systems anatomically aligned.
- `params=None` uses backend extraction defaults and records them. No copied CT/MR/PET/US preset is advertised as universally appropriate.
- Model defaults are documented in `make_classifier`; encrypted Onekey defaults were not recovered or assumed.
- `FeatureSelector` defaults to Mann–Whitney, p < 0.05, absolute Pearson correlation > 0.9, and squared-loss LASSO. A no-feature result is an error.
- Baseline statistics use Mann–Whitney or Welch tests for continuous data, and Fisher/chi-square for categorical data; variables must be explicitly typed.

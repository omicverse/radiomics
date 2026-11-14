# Implementation provenance

This repository was written independently for the requested four-stage radiomics/clinical workflow. The supplied notebooks were read to identify user-facing steps and parameter requirements. No claim of a formal clean-room process is made: the implementer did read those notebooks.

No Onekey package source, notebooks, YAML presets, figures, manuscript text, encrypted bytecode, pretrained weights, or datasets are redistributed here. The local encrypted runtime was not decrypted. Only newly written modules, newly written documentation, and synthetic-data tutorials are included.

Computation uses separately installed, named libraries: SimpleITK, PyRadiomics, NumPy, pandas, SciPy, scikit-learn, statsmodels, matplotlib, seaborn and joblib. Optional adapters use XGBoost, LightGBM and imbalanced-learn. Their licenses remain separate from the license of this repository.

The paired DeLong implementation uses empirical pairwise placement values, counting tied scores as one half. The two class-specific sample covariance contributions determine the variance of the paired AUC difference. It was written directly from this statistical construction; tests compare it to an independent explicit pairwise-matrix calculation. It is not copied from the source notebook's `delong` module.

The nomogram uses an additive rescaling of fitted logistic coefficients. A coefficient's minimum contribution over probability inputs [0,1] is moved into the baseline intercept; nonnegative remaining contributions are converted into points. The sum maps through the fitted logit exactly. A regression test checks this numerical identity.

Public algorithm references:

- [DeLong et al. (1988), comparing correlated ROC areas](https://pubmed.ncbi.nlm.nih.gov/3203132/)
- [PyRadiomics configuration](https://pyradiomics.readthedocs.io/en/latest/customization.html)
- [scikit-learn: data leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: stacking and cross-validated predictions](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)

Synthetic image and clinical-data generators are provided in `examples/demo.py`. Generated AUCs or p-values are software validation artifacts and have no clinical interpretation.

The first Rust migration ports numerical metrics from this repository's own Python implementation and statistical definitions. The Rust crate is tested against the existing Python reference and independent numerical fixtures. It does not port protected third-party source or replace the external feature extraction and modeling engines.

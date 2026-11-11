# Validation record

Initial local validation: 2026-11-11, Windows, Python 3.9.21.

| Component | Version |
|---|---|
| NumPy | 1.23.0 |
| pandas | 1.3.5 |
| SciPy | 1.13.1 |
| scikit-learn | 1.0.2 |
| statsmodels | 0.13.2 |
| PyRadiomics | 3.1.0 |
| pytest | 8.3.5 |

## Completed checks

- The initial 20-test suite passed. The local run reported 16 third-party deprecation warnings from the older matplotlib/seaborn/distutils stack; warnings were not globally suppressed by the package.
- Real SimpleITK/PyRadiomics smoke extraction: known ROI volume and direct voxel mean matched the corresponding extracted values. Mismatched geometry, missing ROI, missing file pairs and record-on-error behavior were tested.
- All eight classifiers were fitted and produced probabilities, including installed XGBoost and LightGBM.
- Perturbing held-out image-feature values and labels left fitted training coefficients and OOF predictions unchanged.
- Identical-prediction DeLong tests returned p=1; tied scores and pairwise placement covariance were checked against a direct matrix reference.
- Known binary confusion-matrix rates and DCA net benefit were checked numerically.
- Categorical clinical encoding, baseline missing counts, reference-category OR analysis, strict ID joins and model persistence were exercised.
- Nomogram points were checked against the fitted model's predicted probabilities exactly.
- All four notebooks executed sequentially with a real Python kernel, with no error cells. Source notebooks are published without generated outputs; executed copies remain local under `artifacts/executed_notebooks/`.
- Full synthetic script: 120 NIfTI images, 90 training and 30 held-out subjects; actual PyRadiomics extraction, nested selection, radiomics and clinical modeling, OOF fusion, all three-model metrics, DeLong, DCA, H–L, calibration and nomogram outputs completed.
- Wheel packaging completed with `pip wheel . --no-deps --no-build-isolation`. No PyPI publication was performed.

The synthetic cohort deliberately does not guarantee that the fused model outperforms either component. Its scores are not evidence of clinical accuracy. The pipeline does not search for a favorable synthetic seed or test split.

## Reproduction

```bash
python -m pytest -q
python examples/demo.py --output examples/output
python tools/run_notebooks.py
```

Use a fresh demo output directory on reruns. `tools/run_notebooks.py` uses the current Python interpreter as its Jupyter kernel and writes executed copies to ignored `artifacts/` paths. An image-capable environment and the `test` dependencies are required.

GitHub Actions runs separate modern-Python core checks and an image-enabled full demo. Remote CI status is available in the repository Actions tab; local success is not a substitute for an uncompleted CI run.

## Remaining limits

No real clinical cohort validation; no multicenter or repeated-patient inference; no upstream OmicVerse modification. Optional SMOTE/learning-curve and alternative embedding paths are not covered by the initial end-to-end notebooks. Third-party version/platform compatibility is limited to the environments actually exercised locally or by CI. No equivalence claim is made against the encrypted source implementation's hidden defaults.

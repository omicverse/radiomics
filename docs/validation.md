# Validation record

Initial local validation: 2026-11-11, Windows, Python 3.9.21.

## Rust migration validation, 2026-11-14

- Rust 1.100.0/MSVC and PyO3 0.30.1 compiled the independent core and CPython ABI3 extension locally.
- Six Rust unit tests passed. `cargo fmt --check` and workspace clippy with warnings denied passed.
- The expanded Python suite passed **37 tests** with the compiled Rust backend, including numerical parity across 10 random seeds, ties, class imbalance, boundary thresholds, identical predictions, positive-label mappings, seeded bootstrap and H–L grouping.
- The inherited seaborn 0.11.2 failed with matplotlib 3.9.4 during the first isolated-environment run. Using matplotlib 3.7.5 in that virtual environment resolved the plotting incompatibility; the original environment was not changed. The successful run reported 15 third-party plotting deprecation warnings.
- The 120-image synthetic end-to-end demo completed with the Rust metric backend. Outputs agreed with the prior Python demo to floating-point precision.
- All four notebooks executed again using the native backend, with executed copies under `artifacts/rust-notebooks/`.
- A Windows `cp39-abi3-win_amd64` wheel was built, installed into an isolated target directory, imported from that directory, and used to execute native metrics. The source distribution and standalone core crate also packaged successfully.
- See [Rust migration notes](rust.md) for the exact scope, build commands and reproducible timings. This is metric-kernel migration, not a full Rust extraction/modeling rewrite.

The initial Python-only validation below remains a historical record.

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

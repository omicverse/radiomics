# Rust core: stage 1

The repository remains an independently importable Python package. OmicVerse only needs a normal `import ov_radiomics`; no code is copied into OmicVerse and no registration or monkey patch is involved.

## Architecture

```text
ov_radiomics Python API: validation, DataFrames, plotting, library adapters
    |
    +-- _native (PyO3, built with maturin, ABI3 CPython 3.9+)
    |      |
    |      +-- ov-radiomics-core (Rust, no dependencies, no Python API)
    |
    +-- Python reference implementation selected explicitly or if extension absent
```

`crates/radiomics-core` accepts Rust slices and returns structured numerical values or input errors. It does not depend on Python, NumPy, PyRadiomics, sklearn, or a dataframe framework. This makes the computational kernel reusable, but an R binding or CLI has **not** been implemented.

`crates/radiomics-python` is a thin PyO3 boundary. It copies validated arrays into owned Rust vectors and releases the Python GIL while computing. It does not claim zero-copy transfer. Distribution quantiles and tail probabilities remain SciPy calculations.

## What moved

| Public Python function | Rust calculation | Still in Python |
|---|---|---|
| `binary_metrics` | Rank AUC, threshold counts, Brier sum, DeLong variance; bootstrap replicate AUC | Validation, final table, CI quantiles and seeded bootstrap draws |
| `delong_test` | Tied-score placements, paired AUC-difference variance | Normal tail probability, tabular result |
| `decision_curve` | Threshold counts using sorting and suffix sums | Input/output layout, treat-all / treat-none rows |
| `hosmer_lemeshow` | Sum of grouped Pearson contributions | pandas quantile grouping, df and chi-square tail |
| `evaluate` | Dispatches each group through `binary_metrics` | Cohort/model grouping |

DCA preserves threshold order, repeated thresholds and the `p >= threshold` rule. It uses O(n log n + t log n) work instead of scanning all observations for every threshold. The measured change combines algorithmic improvement and native execution.

Feature extraction, geometry inspection, feature selection, model fitting, clinical regression and plotting have not been ported in this stage. No claim is made about IBSI feature parity, image-extraction acceleration, or end-to-end ML speedup.

## Stable API and fallback

```python
import ov_radiomics as rad

print(rad.backend_info())
# {'metrics_backend': 'rust', 'rust_core_version': '0.1.0'}
result = rad.binary_metrics(y_true, y_score)                 # auto
native = rad.binary_metrics(y_true, y_score, backend='rust')
reference = rad.binary_metrics(y_true, y_score, backend='python')
```

The new argument is appended to existing signatures. Outputs retain column names, index order, threshold semantics, positive-class mapping, and seeded NumPy bootstrap draws. Floating-point reductions can differ at rounding precision; parity tests use 1e-12 tolerance for the checked inputs.

An unbuilt source checkout can execute the Python reference. Normal wheel builds contain Rust. `auto` only falls back for an absent extension, not for a DLL loading failure. The extension validates its own inputs as well as the Python wrapper, so calling a native function directly produces Python exceptions for invalid numerical input.

## Build and test

Install Rust >= 1.85 and the platform linker; use a Python virtual environment:

```bash
python -m pip install 'maturin>=1.9,<2'
maturin develop --release
cargo test -p ov-radiomics-core --locked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
python -m pytest -q
maturin build --release --locked --out artifacts/wheel-rust
```

The approved local Windows setup keeps Rust under ignored `.build-tools/` and Python build packages in `.venv/`. The virtual environment reuses read-only scientific packages from the existing Python environment; newly installed build/plotting packages remain local. No system PATH was modified. To use that setup in a new PowerShell process:

```powershell
. .\tools\windows-rust-env.ps1
python -c "import ov_radiomics as rad; print(rad.backend_info())"
```

The local inherited seaborn 0.11.2 is incompatible with matplotlib 3.9 (removed `register_cmap`). Local validation uses matplotlib 3.7.5 inside `.venv`, without altering the original environment. Fresh CI environments resolve their own modern plotting stack.

## Validation and benchmark

Local environment on 2026-11-14: Rust 1.100.0 MSVC, PyO3 0.30.1, maturin 1.16.0, Python 3.9.21. Rust unit tests cover exact AUC/placements/variance, confusion counts, DCA thresholds, H–L, invalid inputs, ties and signed zero. Python parity tests cover 10 random seeds, class imbalance, ties, boundary thresholds, positive-label strings, fixed-seed bootstrap, grouped H–L, backend dispatch and missing/broken extension handling.

```bash
python tools/benchmark_metrics.py
python examples/demo.py --output artifacts/rust-demo
```

The benchmark includes public-API validation, vector conversion, computation and pandas output. It uses 7 warm repetitions with alternating backend order and reports the median. These are local measurements, not universal performance guarantees:

| Operation | n | Python ms | Rust ms | Python / Rust |
|---|---:|---:|---:|---:|
| Binary metrics with DeLong CI | 100 | 1.1405 | 0.1842 | 6.19 |
| Paired DeLong | 100 | 0.1868 | 0.1528 | 1.22 |
| DCA, 999 thresholds | 100 | 4.1650 | 0.8327 | 5.00 |
| Binary metrics with DeLong CI | 10,000 | 8.2618 | 1.2352 | 6.69 |
| Paired DeLong | 10,000 | 3.4428 | 1.2647 | 2.72 |
| DCA, 999 thresholds | 10,000 | 19.5143 | 1.2838 | 15.20 |

## Packaging status

The version remains 0.1.0; no PyPI or crates.io publication is performed by these changes. CI builds and tests Windows/Linux wheels, checks pure-Python fallback from source, and runs the existing image-enabled demo. Compiled wheels are development artifacts. A public package release should follow an explicit version/release decision.

Build-system references: [maturin mixed project layout](https://www.maturin.rs/project_layout.html), [PyO3 modules](https://pyo3.rs/v0.30.1/module.html).

"""Compare full public calls, including validation, conversion and DataFrame output.

Run after building the Rust extension; results do not measure image extraction or ML.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path
import numpy as np
import pandas as pd
import ov_radiomics as rad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/rust-metrics-benchmark.json')
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    if rad.backend_info()['metrics_backend'] != 'rust':
        raise RuntimeError('Build the Rust extension before benchmarking.')
    rng = np.random.RandomState(2026)
    rows = []
    for n in (100, 10000):
        y = np.tile([0, 1], n//2)
        rng.shuffle(y)
        a, b = rng.rand(2, n)
        calls = {
            'binary_metrics': lambda backend: rad.binary_metrics(y,a,backend=backend),
            'delong_test': lambda backend: rad.delong_test(y,a,b,backend=backend),
            'decision_curve_999_thresholds': lambda backend: rad.decision_curve(y,a,thresholds=np.linspace(.001,.999,999),backend=backend),
        }
        for operation, call in calls.items():
            python_result, rust_result = call('python'), call('rust')
            comparator = pd.testing.assert_frame_equal if isinstance(python_result,pd.DataFrame) else pd.testing.assert_series_equal
            comparator(python_result,rust_result,rtol=1e-10,atol=1e-12)
            timings = {'python': [], 'rust': []}
            for repetition in range(args.repeats):
                # Alternate execution order to reduce warm-cache/order bias.
                for backend in (('python','rust') if repetition % 2 == 0 else ('rust','python')):
                    start = time.perf_counter()
                    call(backend)
                    timings[backend].append(time.perf_counter()-start)
            medians = {backend:statistics.median(samples) for backend,samples in timings.items()}
            rows.append({'operation':operation,'n':n,'python_ms':medians['python']*1000,
                         'rust_ms':medians['rust']*1000,'ratio_python_over_rust':medians['python']/medians['rust']})
    result = {'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,
              'pandas':pd.__version__,'backend':rad.backend_info(),'seed':2026,'repeats':args.repeats,
              'scope':'Warm public calls; includes boundary conversions. Not a full workflow benchmark.',
              'results':rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()

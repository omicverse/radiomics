"""Cross-language numerical parity and backend-selection contracts."""
import numpy as np
import pandas as pd
import pytest
from ov_radiomics import metrics
from ov_radiomics import _backend


@pytest.fixture
def native():
    return pytest.importorskip('ov_radiomics._native')


@pytest.mark.parametrize('seed', range(10))
def test_randomized_all_metrics(native, seed):
    rng = np.random.RandomState(seed)
    y = np.r_[np.zeros(19, dtype=int), np.ones(31, dtype=int)]
    rng.shuffle(y)
    # Many ties, exact boundaries, class imbalance and non-sorted input.
    a, b = rng.choice(np.linspace(0, 1, 11), size=(2, len(y)))
    for threshold in (0.0, 0.5, 1.0):
        expected = metrics.binary_metrics(y, a, threshold=threshold, backend='python')
        actual = metrics.binary_metrics(y, a, threshold=threshold, backend='rust')
        pd.testing.assert_series_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_series_equal(metrics.delong_test(y,a,b,backend='rust'),
                                  metrics.delong_test(y,a,b,backend='python'), rtol=1e-12, atol=1e-12)
    thresholds = [.7, .001, .5, .5, .999]
    pd.testing.assert_frame_equal(metrics.decision_curve(y, {'a':a, 'b':b}, thresholds=thresholds, backend='rust'),
                                  metrics.decision_curve(y, {'a':a, 'b':b}, thresholds=thresholds, backend='python'), rtol=1e-12, atol=1e-12)


def test_bootstrap_preserves_seed_and_positive_label(native):
    y = ['benign']*15 + ['tumor']*17
    p = np.random.RandomState(3).rand(len(y))
    kwargs = dict(positive_label='tumor', ci_method='bootstrap', n_bootstrap=75, random_state=19)
    pd.testing.assert_series_equal(metrics.binary_metrics(y,p,backend='rust',**kwargs),
                                  metrics.binary_metrics(y,p,backend='python',**kwargs), rtol=1e-12, atol=1e-12)


def test_hl_keeps_pandas_bins_and_tail_probability(native):
    y = np.tile([0,1],100)
    p = np.random.RandomState(9).choice(np.linspace(.05,.95,13), len(y))
    expected = metrics.hosmer_lemeshow(y,p,n_bins=9,backend='python')
    actual = metrics.hosmer_lemeshow(y,p,n_bins=9,backend='rust')
    pd.testing.assert_frame_equal(actual['table'], expected['table'])
    assert actual['df'] == expected['df']
    assert actual['statistic'] == pytest.approx(expected['statistic'],rel=1e-12)
    assert actual['p_value'] == pytest.approx(expected['p_value'],rel=1e-12)


def test_special_auc_and_zero_variance(native):
    y = [0,0,1,1]
    for scores in ([.5]*4, [0,0,1,1], [1,1,0,0], [0.,-0.,-0.,0.]):
        pd.testing.assert_series_equal(metrics.binary_metrics(y,scores,backend='rust'),
                                      metrics.binary_metrics(y,scores,backend='python'), rtol=1e-12, atol=1e-12)
        assert metrics.delong_test(y,scores,scores,backend='rust').p_value == 1
    assert metrics.delong_test(y,[0,0,1,1],[1,1,0,0],backend='rust').p_value == 0


def test_invalid_native_inputs_return_python_exceptions(native):
    for labels, probabilities in [([], []), ([0,0],[.1,.2]), ([0,1],[float('nan'),.4]),
                                 ([0,1],[float('inf'),.3]), ([0,1],[.1]), ([0,1],[-.1,.3])]:
        with pytest.raises(ValueError):
            native.auc(labels,probabilities)
    with pytest.raises(ValueError):
        native.net_benefit([0,1],[.1,.9],[1.0])
    with pytest.raises(ValueError):
        native.hl_statistic([5,5,5],[1.,2.,4.],[0.,2.,4.])


def test_backend_selection_and_missing_module(monkeypatch):
    def absent(name):
        raise ModuleNotFoundError('not built',name='ov_radiomics._native')
    monkeypatch.setattr(_backend,'import_module',absent)
    assert _backend.resolve_backend('auto') is None
    assert _backend.resolve_backend('python') is None
    assert _backend.backend_info()['metrics_backend'] == 'python'
    with pytest.raises(ImportError,match='not built'):
        _backend.resolve_backend('rust')
    with pytest.raises(ValueError,match='backend'):
        _backend.resolve_backend('invalid')
    assert metrics.binary_metrics([0,0,1,1],[.1,.4,.35,.8]).auc == .75


def test_broken_extension_is_not_silently_ignored(monkeypatch):
    def broken(name):
        raise ImportError('DLL load failed')
    monkeypatch.setattr(_backend,'import_module',broken)
    with pytest.raises(ImportError,match='DLL load failed'):
        _backend.resolve_backend('auto')


def test_auto_dispatches_to_actual_rust(native,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Python AUC was unexpectedly called')
    monkeypatch.setattr(metrics,'roc_auc_score',forbidden)
    assert _backend.backend_info()['metrics_backend'] == 'rust'
    assert metrics.binary_metrics([0,0,1,1],[.1,.4,.35,.8]).auc == .75

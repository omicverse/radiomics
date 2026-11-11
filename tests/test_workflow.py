import sys
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.base import clone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ov_radiomics as rad


@pytest.fixture
def cohort():
    rng = np.random.RandomState(42)
    n = 100
    y = np.tile([0, 1], n//2)
    frame = pd.DataFrame({'ID': [f'P{i:03d}' for i in range(n)], 'label': y,
        'signal': y*1.3+rng.normal(size=n), 'noise': rng.normal(size=n),
        'age': 55+y*6+rng.normal(size=n)*10, 'sex': rng.choice(['F', 'M'], n)})
    return rad.split_data(frame, 'label', test_size=.25)


def test_target_and_join_contracts(cohort):
    with pytest.raises(ValueError, match='already exists'):
        rad.split_data(cohort, 'label')
    with pytest.raises(ValueError, match='ID sets differ'):
        rad.join_data(cohort[['ID']], cohort[['ID', 'age']].iloc[:-1])
    with pytest.raises(ValueError, match='unique'):
        rad.join_data(pd.concat([cohort[['ID']]]*2), cohort[['ID', 'age']])
    assert len(rad.join_data(cohort[['ID']], cohort[['ID', 'age']])) == len(cohort)


def test_selector_handles_anticorrelation_and_missing():
    rng = np.random.RandomState(7)
    X = pd.DataFrame({'a': rng.normal(size=80), 'b': rng.normal(size=80)})
    X['negative_a'] = -X.a
    X['constant'] = 1
    X['missing'] = np.nan
    y = np.tile([0, 1], 40)
    selector = rad.FeatureSelector(univariate=None, lasso=False).fit(X, y)
    assert selector.selected_features_ == ['a', 'b']
    changed = X.iloc[:2].copy()
    changed['a'] += 100
    expected = (changed[['a','b']]-X[['a','b']].mean())/X[['a','b']].std(ddof=0)
    pd.testing.assert_frame_equal(selector.transform(changed), expected)
    assert clone(selector).get_params() == selector.get_params()


def test_nested_lasso(cohort):
    train = cohort.query('group == "train"')
    selector = rad.FeatureSelector(univariate=None, missing='median', cv=3, alphas=[.001,.01,.05])
    selector.fit(train[['signal', 'noise']], train.label)
    assert len(selector.lasso_cv_results_) == 3
    assert 'signal' in selector.selected_features_
    assert len(selector.linear_score(train[['signal','noise']])) == len(train)
    rad.plot_lasso_path(selector)
    rad.plot_lasso_cv(selector)
    rad.plot_feature_weights(selector.coefficients_)
    plt.close('all')


def test_holdout_cannot_change_fits_or_oof(cohort):
    kwargs = dict(target='label', features=['signal','noise'], models=['LR'], cv=3,
        selector=rad.FeatureSelector(univariate=None, lasso=False, missing='median'))
    first = rad.train_models(cohort, **kwargs)
    changed = cohort.copy()
    test = changed.group == 'test'
    changed.loc[test, 'signal'] = 99999
    changed.loc[test, 'label'] = 1-changed.loc[test, 'label']
    second = rad.train_models(changed, **kwargs)
    pd.testing.assert_frame_equal(first.oof_predictions, second.oof_predictions)
    np.testing.assert_array_equal(first.models['LR']['classifier'].coef_, second.models['LR']['classifier'].coef_)
    assert set(first.oof_predictions.ID) == set(cohort.loc[~test, 'ID'])
    assert len(first.oof_predictions) == (~test).sum()


def test_clinical_fusion_and_serialization(cohort, tmp_path):
    rad_result = rad.train_models(cohort, 'label', ['signal','noise'], cv=3, scale='standard')
    clinic_result = rad.train_models(cohort, 'label', ['age','sex'], categorical=['sex'], scale='standard', cv=3)
    parts = {'Rad': rad_result, 'Clinical': clinic_result}
    oof = rad.fusion_data(parts)
    fusion = rad.fit_fusion(oof)
    holdout = rad.fusion_data(parts, cohort='test')
    with pytest.raises(ValueError, match='OOF'):
        rad.fit_fusion(holdout)
    probability = fusion.predict_proba(holdout)
    assert probability.between(0, 1).all()
    pd.testing.assert_series_equal(probability, fusion.predict_proba(holdout.assign(y_true=1-holdout.y_true)))
    rad_result.save(tmp_path)
    import joblib
    restored = joblib.load(tmp_path/'training.joblib')
    pd.testing.assert_series_equal(rad_result.predict_proba(cohort), restored.predict_proba(cohort))
    assert len(rad.evaluate(rad_result)) == 2
    fig = rad.plot_nomogram(fusion)
    mapping = fig.nomogram_mapping
    beta = fusion.model.coef_[0]
    points = ((holdout[fusion.score_columns].to_numpy()*beta-np.minimum(beta,0))/mapping['logit_per_point']).sum(axis=1)
    np.testing.assert_allclose(expit(mapping['baseline_logit']+points*mapping['logit_per_point']), probability)
    plt.close('all')


def test_metrics_known_values_and_ties():
    y = np.array([0,0,1,1])
    p = np.array([.1,.4,.35,.8])
    metric = rad.binary_metrics(y,p)
    assert metric.auc == .75
    assert metric.specificity == 1
    assert metric.sensitivity == .5
    assert rad.delong_test(y,p,p).p_value == 1
    assert rad.binary_metrics(y,np.full(4,.5)).auc == .5
    curve = rad.decision_curve(y,p,thresholds=[.5])
    assert curve.query('model == "Model"').net_benefit.iloc[0] == .25
    with pytest.raises(ValueError):
        rad.decision_curve(y,p,thresholds=[1])
    with pytest.raises(ValueError, match='indices'):
        rad.binary_metrics(pd.Series(y),pd.Series(p,index=[3,2,1,0]))
    with pytest.raises(ValueError, match='three distinct'):
        rad.hosmer_lemeshow(y,np.full(4,.5))


def test_delong_against_pairwise_reference():
    rng = np.random.RandomState(3)
    y = np.r_[np.zeros(27), np.ones(23)]
    a, b = rng.choice(np.linspace(0,1,9), (2,50))
    def reference(p):
        positive, negative = p[y==1,None], p[None,y==0]
        matrix = (positive>negative)+.5*(positive==negative)
        return matrix.mean(axis=1), matrix.mean(axis=0)
    av, an = reference(a)
    bv, bn = reference(b)
    variance = np.var(av-bv,ddof=1)/len(av)+np.var(an-bn,ddof=1)/len(an)
    expected_z = (av.mean()-bv.mean())/np.sqrt(variance)
    assert rad.delong_test(y,a,b).z == pytest.approx(expected_z)


def test_clinical_tables(cohort):
    frame = cohort.copy()
    frame.loc[0,'age'] = np.nan
    baseline = rad.baseline_table(frame,'label',['age','sex'],continuous=['age'],split='group')
    assert len(baseline) == 4
    assert sum(baseline['class_0_missing']) == 1
    train = frame.query('group == "train"')
    analysis = rad.multivariable_logistic(train,'label',['age','sex'],categorical=['sex'],reference={'sex':'F'})
    assert analysis.reference == {'sex':'F'}
    assert 'sex_M' in analysis.table.term.values
    assert len(rad.univariate_logistic(train,'label',['age','sex'],categorical=['sex'])) == 2


@pytest.mark.parametrize('name',['LR','SVM','KNN','RandomForest','ExtraTrees','MLP','XGBoost','LightGBM'])
def test_all_models(name,cohort):
    if name == 'XGBoost':
        pytest.importorskip('xgboost')
    if name == 'LightGBM':
        pytest.importorskip('lightgbm')
    params = {'n_estimators':5} if name in ['RandomForest','ExtraTrees','XGBoost','LightGBM'] else {}
    if name == 'MLP':
        params = {'solver':'lbfgs','max_iter':1000,'hidden_layer_sizes':(5,)}
    result = rad.train_models(cohort,'label',['signal','noise'],cv=2,models=[name],scale='standard',model_params={name:params})
    assert result.predictions.probability.between(0,1).all()


def test_plots(cohort):
    y = cohort.label
    p = pd.Series(expit(cohort.signal), index=cohort.index)
    rad.plot_roc(y,{'Rad':p})
    rad.plot_calibration(y,p,n_bins=5)
    rad.plot_dca(rad.decision_curve(y,p),ylim=(-.1,.6))
    rad.plot_confusion_matrix(y,p)
    rad.plot_prediction_scores(y,p)
    rad.plot_correlation(cohort[['signal','noise','age']].corr())
    rad.plot_correlation(cohort[['signal','noise','age']].corr(),cluster=True)
    rad.plot_embedding(cohort[['signal','noise']],y)
    rad.plot_feature_summary(['original_firstorder_Mean','original_shape_Volume'])
    rad.plot_feature_pvalues(pd.DataFrame({'feature':['a','b','c','d'], 'p_value':[.01,.02,.1,.2]}))
    rad.plot_cv_results(pd.DataFrame({'model':['LR']*3,'auc':[.7,.8,.9]}))
    rad.plot_metrics(pd.DataFrame({'model':['LR','SVM'],'group':['test']*2,'auc':[.7,.8]}))
    assert len(plt.get_fignums()) == 12
    plt.close('all')
def test_image_backend_not_imported_on_package_import():
    import subprocess
    result = subprocess.run([sys.executable,'-c',
        'import sys; import ov_radiomics; assert "radiomics" not in sys.modules; assert "SimpleITK" not in sys.modules'],
        capture_output=True,text=True)
    assert result.returncode == 0, result.stderr

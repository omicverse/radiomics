"""Four-step synthetic demonstration, including real PyRadiomics extraction.

Run: python examples/demo.py --output examples/output
No patient data, downloaded data, or Onekey runtime are used.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import ov_radiomics as rad


def make_data(output, n=120):
    import SimpleITK as sitk
    output = Path(output)
    root = output/'synthetic_images'
    if (root/'images').exists():
        raise FileExistsError('Demo output exists. Choose a fresh --output directory to avoid mixing runs.')
    (root/'images').mkdir(parents=True)
    (root/'masks').mkdir()
    rng = np.random.RandomState(2026)
    target = np.tile([0,1], n//2)
    rng.shuffle(target)
    rows = []
    for i, y in enumerate(target):
        identifier = f'SYNTHETIC_{i:03d}.nii.gz'
        latent = .8*y + rng.normal()
        voxels = rng.normal(50+4*latent, 8+rng.uniform(0,4), (14,14,14)).astype(np.float32)
        mask_array = np.zeros(voxels.shape, dtype=np.uint8)
        width = rng.randint(5,9)
        mask_array[3:3+width,3:10,3:10] = 1
        for array, directory in [(voxels,'images'),(mask_array,'masks')]:
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((1.2,1.2,2.0))
            sitk.WriteImage(image,str(root/directory/identifier))
        rows.append({'ID':identifier,'label':int(y),'age':55+5*y+rng.normal()*10,
                     'sex':rng.choice(['F','M']),'marker':.5*y+rng.normal()})
    clinical = rad.split_data(pd.DataFrame(rows),'label',test_size=.25,random_state=42)
    clinical.to_csv(output/'clinical.csv',index=False)
    return root, clinical


def step1(output):
    output = Path(output)
    image_root, clinical = make_data(output)
    samples = rad.match_images(image_root)
    rad.check_images(samples).to_csv(output/'geometry.csv',index=False)
    # Original-only parameters are for this small synthetic demo, not a modality prescription.
    extracted = rad.extract(samples, params={
        'imageType':{'Original':{}},
        'featureClass':{'firstorder':['Mean','Median','Variance','Skewness','Kurtosis','Entropy'],
                        'shape':['VoxelVolume','Sphericity']},
        'setting':{'binWidth':5}}, n_jobs=1)
    extracted.save(output/'extraction')
    data = rad.join_data(extracted.features,clinical[['ID','label','group']])
    columns = list(extracted.features.columns.drop('ID'))
    # A permissive demo screen keeps the example stable. Real studies should prespecify it.
    selector = rad.FeatureSelector(missing='median',univariate='mannwhitney',p_threshold=1.0,
        corr_threshold=.9,cv=3,alphas=[.001,.01,.05],random_state=42)
    result = rad.train_models(data,'label',columns,selector=selector,models=['LR','RandomForest'],cv=3,
        random_state=42,model_params={'RandomForest':{'n_estimators':50}})
    result.save(output/'rad')
    selected = result.models['LR']['preprocess']
    selected.univariate_results_.to_csv(output/'feature_statistics.csv',index=False)
    selected.coefficients_.to_csv(output/'lasso_coefficients.csv')
    for name, ax in [('feature_summary',rad.plot_feature_summary(extracted.features)),
                     ('lasso_path',rad.plot_lasso_path(selected)),('lasso_cv',rad.plot_lasso_cv(selected)),
                     ('feature_weights',rad.plot_feature_weights(selected.coefficients_)),
                     ('rad_cv',rad.plot_cv_results(result.cv_results))]:
        ax.figure.savefig(output/f'{name}.png',dpi=130,bbox_inches='tight')
        plt.close(ax.figure)
    return result


def step2(output):
    output = Path(output)
    clinical = pd.read_csv(output/'clinical.csv')
    features = ['age','sex','marker']
    baseline = rad.baseline_table(clinical,'label',features,continuous=['age','marker'],categorical=['sex'],split='group')
    baseline.to_csv(output/'baseline.csv',index=False)
    train = clinical.query('group == "train"')
    rad.univariate_logistic(train,'label',features,categorical=['sex'],reference={'sex':'F'}).to_csv(output/'univariate.csv',index=False)
    rad.multivariable_logistic(train,'label',features,categorical=['sex'],reference={'sex':'F'}).table.to_csv(output/'multivariable.csv',index=False)
    # The demonstration prespecifies clinical features; the baseline test cohort never selects them.
    return baseline


def step3(output):
    output = Path(output)
    clinical = pd.read_csv(output/'clinical.csv')
    result = rad.train_models(clinical,'label',['age','sex','marker'],categorical=['sex'],
        scale='standard',models=['LR','RandomForest'],cv=3,random_state=42,
        model_params={'RandomForest':{'n_estimators':50}})
    result.save(output/'clinical')
    return result


def step4(output):
    output = Path(output)
    parts = {'Rad':joblib.load(output/'rad/training.joblib'),
             'Clinical':joblib.load(output/'clinical/training.joblib')}
    fusion = rad.fit_fusion(rad.fusion_data(parts,model='LR'))
    joblib.dump(fusion,output/'fusion.joblib')
    tables, comparisons, hl_rows = [], [], []
    for cohort in ('train','test'):
        data = rad.fusion_data(parts,model='LR',cohort=cohort)
        data['Fusion'] = fusion.predict_proba(data)
        data.to_csv(output/f'{cohort}_probabilities.csv',index=False)
        scores = data[['Rad','Clinical','Fusion']]
        for name in scores:
            metric = rad.binary_metrics(data.y_true,scores[name])
            metric['model'], metric['group'] = name, cohort
            tables.append(metric)
            hl = rad.hosmer_lemeshow(data.y_true,scores[name],n_bins=5)
            hl_rows.append({'model':name,'group':cohort,**{k:v for k,v in hl.items() if k!='table'}})
        for comparator in ('Rad','Clinical'):
            comparison = rad.delong_test(data.y_true,data.Fusion,data[comparator])
            comparison['comparison'], comparison['group'] = 'Fusion vs '+comparator,cohort
            comparisons.append(comparison)
        dca = rad.decision_curve(data.y_true,scores)
        dca.to_csv(output/f'{cohort}_dca.csv',index=False)
        for name, ax in [('roc',rad.plot_roc(data.y_true,scores)),
                         ('calibration',rad.plot_calibration(data.y_true,scores,n_bins=5)),
                         ('dca',rad.plot_dca(dca,ylim=(-.1,.65))),
                         ('confusion',rad.plot_confusion_matrix(data.y_true,data.Fusion)),
                         ('probabilities',rad.plot_prediction_scores(data.y_true,data.Fusion))]:
            ax.figure.savefig(output/f'{cohort}_{name}.png',dpi=130,bbox_inches='tight')
            plt.close(ax.figure)
    metrics = pd.DataFrame(tables)
    metrics.to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(output/'delong.csv',index=False)
    pd.DataFrame(hl_rows).to_csv(output/'hosmer_lemeshow.csv',index=False)
    figure = rad.plot_nomogram(fusion)
    figure.savefig(output/'nomogram.png',dpi=130,bbox_inches='tight')
    plt.close(figure)
    (output/'provenance.json').write_text(json.dumps({
        'data':'Entirely synthetic; not clinical evidence', 'seed':2026,
        'n':120,'split_seed':42,'test_fraction':.25,
        'train_metrics':'Apparent performance, not unbiased validation',
        'fusion_training':'OOF base-model probabilities from the training set',
        'ov_radiomics_version':rad.__version__},indent=2),encoding='utf-8')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default='examples/output')
    args = parser.parse_args()
    for step in (step1,step2,step3,step4):
        print(step.__name__,flush=True)
        result = step(args.output)
    print(result.to_string(index=False))

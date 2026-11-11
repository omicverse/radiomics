"""Maintain runnable notebooks from original tutorial cells (no source-notebook copying)."""
from pathlib import Path
from textwrap import dedent
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def write(name, cells):
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {'display_name':'Python 3','language':'python','name':'python3'}
    notebook.cells = [(nbformat.v4.new_markdown_cell if kind == 'md' else nbformat.v4.new_code_cell)(dedent(source).strip()) for kind, source in cells]
    nbformat.write(notebook, ROOT/'examples'/name)


COMMON = '''
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import ov_radiomics as rad

# Run notebooks in order, with examples/ as the working directory.
output = Path("output/notebook_demo")
'''


write('01_radiomics.ipynb', [
('md', '''# 1. 图像、特征提取与影像模型

本教程生成 120 例**合成** NIfTI 图像、ROI 和临床记录。它用于验证软件，不能解释成临床研究结果。

准备环境：在仓库根目录执行 `python -m pip install -e ".[image,test]"`。从 `examples/` 目录启动 Jupyter，按 01 → 04 顺序运行。

真实研究只需要替换图像目录和临床表。样本 ID 必须精确匹配；每行是一位独立患者。本版不处理同一患者多病灶/多扫描。
'''),
('code', COMMON),
('md', '''## 数据与固定划分
示例生成器使用固定种子，提前按结局分层保留 25% 测试集。生成器拒绝覆盖已有数据；重新运行时请换一个新的 `output` 路径。'''),
('code', '''from demo import make_data
image_root, clinical = make_data(output, n=120)
clinical.groupby(["group", "label"]).size()
'''),
('md', '''## 配对与检查
`match_images` 按 `.nii` / `.nii.gz` 文件名匹配，`check_images` 检查尺寸、spacing、origin、direction 及 ROI。默认不会自动修正 mask。'''),
('code', '''samples = rad.match_images(image_root)
qc = rad.check_images(samples, roi_label=1)
assert qc.ok.all(), qc.loc[~qc.ok]
qc.head()
'''),
('md', '''## 提取参数
`params` 接受 PyRadiomics YAML 文件或字典。这里仅提取 Original 的少量特征以缩短教程时间；这不是 CT/MR 的通用推荐参数。可自行增加 LoG/Wavelet、纹理类别，并明确设置重采样和 binWidth。
'''),
('code', '''params = {
    "imageType": {"Original": {}},
    "featureClass": {
        "firstorder": ["Mean", "Median", "Variance", "Skewness", "Kurtosis", "Entropy"],
        "shape": ["VoxelVolume", "Sphericity"],
    },
    "setting": {"binWidth": 5},
}
extracted = rad.extract(samples, params=params, roi_label=1, n_jobs=1)
extracted.save(output / "extraction")
extracted.features.head()
'''),
('code', '''rad.plot_feature_summary(extracted.features)
plt.show()
'''),
('md', '''## 训练内筛选和交叉验证
依次做缺失值处理、Z-score、单因素检验、绝对相关性过滤、LASSO。

教程使用宽松的 `p_threshold=1.0`，用于稳定演示。研究应预先确定筛选规则；可用 `univariate=None` 或 `lasso=False` 关闭相应步骤。

`alpha="cv"` 在内部交叉验证中重新拟合筛选流程；外部模型交叉验证也会重新拟合整条 Pipeline。不会先对全体数据筛选。
'''),
('code', '''data = rad.join_data(extracted.features, clinical[["ID", "label", "group"]])
features = extracted.features.columns.drop("ID").tolist()
selector = rad.FeatureSelector(
    missing="median", univariate="mannwhitney", p_threshold=1.0,
    corr_method="pearson", corr_threshold=0.9,
    lasso=True, alpha="cv", cv=3, alphas=[0.001, 0.01, 0.05], random_state=42,
)
rad_result = rad.train_models(
    data, target="label", features=features, selector=selector,
    models=["LR", "RandomForest"], cv=3, random_state=42,
    model_params={"RandomForest": {"n_estimators": 50}},
)
rad_result.save(output / "rad")
rad_result.cv_results
'''),
('md', '''其他模型名：`SVM`、`KNN`、`ExtraTrees`、`MLP`、`XGBoost`、`LightGBM`。后两者需要 boost 可选依赖。模型比较使用所有折，不能挑选表现最好的测试划分。'''),
('code', '''fitted_selector = rad_result.models["LR"]["preprocess"]
print(fitted_selector.selected_features_)
rad.plot_lasso_path(fitted_selector)
rad.plot_lasso_cv(fitted_selector)
rad.plot_feature_weights(fitted_selector.coefficients_)
rad.plot_cv_results(rad_result.cv_results)
plt.show()
'''),
('code', '''rad.evaluate(rad_result, threshold=0.5)
'''),
('md', '''`train` 是表观性能；`test` 是固定留出集性能。这里的合成数据指标只说明程序可运行，不说明任何模型的临床价值。'''),
])

write('02_clinical_statistics.ipynb', [
('md', '''# 2. 临床基线、单因素与多因素分析

读取步骤 1 的同一份患者划分。基线表保留缺失计数，不先填补原始数据。
'''),('code', COMMON),
('code', '''clinical = pd.read_csv(output / "clinical.csv")
features = ["age", "sex", "marker"]
continuous = ["age", "marker"]
categorical = ["sex"]
clinical.head()
'''),
('md', '''`split="group"` 表示在 train/test 各自内部比较结局类别，不是检验 train 与 test 的差异。连续变量可选 Mann–Whitney 或 Welch，分类变量按设置使用 Fisher/卡方。'''),
('code', '''baseline = rad.baseline_table(
    clinical, target="label", features=features,
    continuous=continuous, categorical=categorical, split="group",
    continuous_test="mannwhitney", categorical_test="auto",
)
baseline.to_csv(output / "baseline.csv", index=False)
baseline
'''),
('md', '''## 训练集的 Logistic 推断
显式设置分类变量参考组与 95% CI。采用完整病例分析，返回使用/删除的样本数；完全分离、秩不足或不收敛会报错。

这里的非惩罚 Logit 用于 OR/CI 推断，与预测用的 sklearn LR 区分。示例预先指定临床变量；不根据测试组 p 值选择变量。
'''),
('code', '''train = clinical.loc[clinical.group == "train"]
univariate = rad.univariate_logistic(
    train, "label", features, categorical=categorical,
    reference={"sex": "F"}, confidence=0.95,
)
univariate.to_csv(output / "univariate.csv", index=False)
univariate
'''),
('code', '''multivariable = rad.multivariable_logistic(
    train, "label", features, categorical=categorical,
    reference={"sex": "F"}, confidence=0.95,
)
print("Used:", multivariable.n_used, "Dropped:", multivariable.n_dropped)
multivariable.table.to_csv(output / "multivariable.csv", index=False)
multivariable.table
'''),
('md', '''若实际研究根据单因素结果筛选临床变量，应只使用训练数据；若还要报告筛选后模型的交叉验证性能，变量筛选本身也必须纳入交叉验证。这里不提供自动逐步回归。'''),
])

write('03_clinical_model.ipynb', [
('md', '''# 3. 临床模型

复用影像模型的同一患者集合、结局和 train/test 划分。分类编码、缺失值填补、标准化均在训练折拟合。
'''),('code', COMMON),
('code', '''clinical = pd.read_csv(output / "clinical.csv")
clinical_result = rad.train_models(
    clinical, target="label", features=["age", "sex", "marker"],
    continuous=["age", "marker"], categorical=["sex"],
    scale="standard", models=["LR", "RandomForest"],
    cv=3, random_state=42,
    model_params={"RandomForest": {"n_estimators": 50}},
)
clinical_result.save(output / "clinical")
clinical_result.cv_results
'''),
('md', '''`scale=None` 可关闭数值变量标准化。分类变量采用 one-hot 编码；测试时未知水平按编码器的 `handle_unknown="ignore"` 处理。

`.oof_predictions` 的每一行来自未使用该样本训练的基础模型，将用于下一步融合。
'''),
('code', '''clinical_result.oof_predictions.head()
'''),
('code', '''rad.plot_cv_results(clinical_result.cv_results)
metrics = rad.evaluate(clinical_result, threshold=0.5)
rad.plot_metrics(metrics, metric="auc")
plt.show()
metrics
'''),
('code', '''test = clinical_result.predictions.query('group == "test" and model == "LR"')
rad.plot_roc(test.y_true, test.probability)
rad.plot_confusion_matrix(test.y_true, test.probability, threshold=0.5)
plt.show()
'''),
])

write('04_fusion_evaluation.ipynb', [
('md', '''# 4. 概率融合、模型比较与列线图

将训练集的影像和临床**折外预测概率**输入 Logistic 融合。`fusion_data` 按 ID 严格对齐，还会核对结局与折编号。

不能用测试标签重新拟合融合模型。基础模型 OOF 数据参与融合拟合后，也不能再把融合在这些数据上的表现视为独立验证。
'''),('code', COMMON),
('code', '''parts = {
    "Rad": joblib.load(output / "rad" / "training.joblib"),
    "Clinical": joblib.load(output / "clinical" / "training.joblib"),
}
oof = rad.fusion_data(parts, model="LR", cohort="oof")
fusion = rad.fit_fusion(oof, target="y_true", score_columns=["Rad", "Clinical"])
joblib.dump(fusion, output / "fusion.joblib")
oof.head()
'''),
('md', '''## 固定测试集预测
沿用训练好的基础模型和融合模型，下面没有任何 `fit(test)`。
'''),
('code', '''test = rad.fusion_data(parts, model="LR", cohort="test")
test["Fusion"] = fusion.predict_proba(test)
scores = test[["Rad", "Clinical", "Fusion"]]
test.to_csv(output / "test_probabilities.csv", index=False)
metrics = pd.DataFrame({name: rad.binary_metrics(test.y_true, scores[name], threshold=0.5) for name in scores}).T
metrics.to_csv(output / "test_metrics.csv")
metrics
'''),
('code', '''rad.plot_roc(test.y_true, scores)
rad.plot_calibration(test.y_true, scores, n_bins=5)
curves = rad.decision_curve(test.y_true, scores, thresholds=np.linspace(.01,.99,99))
rad.plot_dca(curves, ylim=(-.1,.65))
plt.show()
'''),
('md', '''## 配对 AUC 比较与校准检验
DeLong 接收同一批独立患者的配对预测，返回普通 p 值。H–L 分组遇到重复概率时可能减少实际组数，并返回对应自由度与分组表；p 值大不代表已证明校准良好。
'''),
('code', '''delong = pd.DataFrame({name: rad.delong_test(test.y_true, test.Fusion, test[name]) for name in ["Rad","Clinical"]}).T
delong.to_csv(output / "delong.csv")
delong
'''),
('code', '''hl = rad.hosmer_lemeshow(test.y_true, test.Fusion, n_bins=5)
print({key: value for key, value in hl.items() if key != "table"})
hl["table"]
'''),
('code', '''rad.plot_prediction_scores(test.y_true, test.Fusion)
rad.plot_confusion_matrix(test.y_true, test.Fusion)
figure = rad.plot_nomogram(fusion, risk_ticks=[.1,.25,.5,.75,.9], figsize=(10,6))
figure.savefig(output / "nomogram.png", dpi=150, bbox_inches="tight")
plt.show()
'''),
('md', '''列线图直接来自已拟合 LR 的系数，预测变量是两个基础模型的阳性概率。它不会重新训练模型。

此四步教程全部基于合成数据；完成执行表示功能链路可运行，不构成临床验证。
'''),
])

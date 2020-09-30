# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
#%load_ext autoreload
#%autoreload 2

# %%
import numpy as np

# %%
from osic_pulmonary_fibrosis_progression import io, datasource, lightgbm, dataset, task, config, metric

# %%
conf = config.Config()

# %%
all_source, test_source = io.load_osic_pulmonary_fibrosis_progression_csv(use_pseudo_baselines=True, ignore_bad_ct=True)

# %%
train_source, val_source = datasource.train_validate_split(all_source)

# %%
train_dataset = dataset.CTDataset(train_source, use_ct_data=False)
val_dataset = dataset.CTDataset(val_source, use_ct_data=False)

# %%
models, oof_fvc = task.ct_features.train_all_folds(conf, all_source, 4, use_ct_data=False)

# %%
predictions_fvc = task.ct_features.infer_with_all_folds(conf, test_source, models, use_ct_data=False)

# %%
task.tabular.show_feature_importance(models, train_dataset.features)

# %%
all_source.df['FVC_pred'] = oof_fvc
test_source.df['FVC_pred'] = predictions_fvc

# %%
score = metric.calc_laplace_log_likelihood(all_source.df["FVC"].values, np.array([100] * len(all_source)), all_source.df["FVC_pred"].values)
print(score)

# %%
confidence = metric.calc_best_confidence(all_source.df["FVC"].values, all_source.df["FVC_pred"].values)

# %%
# optimized score
all_source.df['Confidence'] = confidence
score = metric.calc_laplace_log_likelihood(all_source.df["FVC"].values, all_source.df['Confidence'].values, all_source.df["FVC_pred"].values)
print(score)

# %%
models, oof_confidence = task.ct_features.train_all_folds(conf, all_source, 4,  target="Confidence", use_ct_data=False)

# %%
predictions_confidence= task.ct_features.infer_with_all_folds(conf, test_source, models, use_ct_data=False)

# %%
all_source.df["Confidence"] = oof_confidence
score = metric.calc_laplace_log_likelihood(all_source.df["FVC"].values, all_source.df["Confidence"].values, all_source.df["FVC_pred"].values)
print(score)

# %%
test_source.df['Confidence'] = predictions_confidence

# %%
io.save_result(test_source)

# %%

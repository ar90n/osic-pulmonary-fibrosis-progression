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
import matplotlib.pyplot as plt

import torchtoolbox.transform as transforms

# %%
from osic_pulmonary_fibrosis_progression import io, datasource, lightgbm, dataset, task, config, metric, transforms as my_transforms

# %%
conf = config.Config()

# %%
train_transform = transforms.Compose([
    my_transforms.DropSlice(),
    my_transforms.LungMask(),
    my_transforms.DescribeVolume()
])

# %%
all_source, test_source = io.load_osic_pulmonary_fibrosis_progression_csv(use_pseudo_baselines=False)

# %%
train_source, val_source = datasource.train_validate_split(all_source)

# %%
train_dataset = dataset.CTDataset(train_source, transforms=train_transform)
val_dataset = dataset.CTDataset(val_source, transforms=train_transform)

# %%
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4), sharex=True)
axL.imshow(train_dataset[21][0].pixel_array[12,:,:])
axR.imshow(train_dataset[21][0].metadata["mask"][12,:,:] == 1)
fig.show()

# %%

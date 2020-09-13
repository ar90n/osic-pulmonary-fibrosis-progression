from pathlib import Path

import imageio as io
import numpy as np
import torch
from torch.utils.data import Dataset

from .datasource import DataSource


class TabularDataset(Dataset):
    def __init__(
        self, source: DataSource, train: bool = True, target=None, features=None
    ):
        self.source = source
        self.train = train

        if target is None:
            self.target = self.get_target_column()
        else:
            self.target = target

        if features is None:
            self.features = self.get_feature_columns()
        else:
            self.features = features

    def __getitem__(self, index):
        meta = np.array(
            self.source.df.iloc[index][self.features].values, dtype=np.float32
        )
        if self.train:
            y = self.source.df.iloc[index][self.target]
            return meta, y
        else:
            return meta

    def __len__(self):
        return len(self.source.df)

    @classmethod
    def get_target_column(cls):
        return "FVC"

    @classmethod
    def get_feature_columns(cls, use_one_hot_encoding: bool = False):
        smoking_status_featues = ["SmokingStatus"]
        if use_one_hot_encoding:
            smoking_status_featues = [
                "SmokingStatus_Currently smokes",
                "SmokingStatus_Ex-smoker",
                "SmokingStatus_Never smoked",
            ]

        return [
            "Sex",
            "base_FVC",
            "base_Percent",
            "base_Age",
            "Week_passed",
            *smoking_status_featues,
        ]

from pathlib import Path
from typing import List, Optional

import imageio as io
import numpy as np
import torch
from torch.utils.data import Dataset

from .datasource import DataSource


class TabularDataset(Dataset):
    def __init__(
        self,
        source: DataSource,
        train: bool = True,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
        use_one_hot_encoding: bool = False,
    ) -> None:
        self.source = source
        self.train = train
        self.use_one_hot_encoding = use_one_hot_encoding

        if target is None:
            self.target = self.get_default_target_column()
        else:
            self.target = target

        if features is None:
            self.features = self.get_default_feature_columns()
        else:
            self.features = features

    def __getitem__(self, index: int) -> np.ndarray:
        meta = np.array(
            self.source.df.iloc[index][self.features].values, dtype=np.float32
        )
        if self.train:
            y = self.source.df.iloc[index][self.target]
            return meta, y
        else:
            return meta

    def __len__(self) -> int:
        return len(self.source.df)

    def get_default_target_column(self) -> str:
        return "FVC"

    def get_default_feature_columns(self) -> List[str]:
        smoking_status_featues = ["SmokingStatus"]
        if self.use_one_hot_encoding:
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

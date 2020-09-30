import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union, cast

import imageio as io
import numpy as np
import torch
from torch.utils.data import Dataset

from .datasource import DataSource
from .dcm import load_stack


@dataclass
class Record:
    pixel_array: np.ndarray
    metadata: Mapping[str, Any]
    tabular: np.ndarray


class CTDataset(Dataset):
    def __init__(
        self,
        source,
        train: bool = True,
        transforms: Optional[Callable] = None,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
        use_one_hot_encoding: bool = False,
        use_ct_data: bool = True,
    ) -> None:
        self.source = source
        self.train = train

        if transforms is not None:
            transforms = transforms
        self.transforms = transforms

        if target is None:
            target = self.get_default_target_column()
        self.target = target

        if features is None:
            features = self.get_default_feature_columns(use_one_hot_encoding)
        self._tabular_features = features

        self.use_ct_data = use_ct_data

    def __getitem__(self, index: int) -> Union[Record, Tuple[Record, float]]:
        img_root = self.get_img_root(index)
        cur_row = self.source.df.iloc[index]

        pixel_array = np.empty(0)
        metadata = {}
        tabular = np.array(cur_row[self._tabular_features].values, dtype=np.float32)
        if self.use_ct_data:
            patient_img_path = img_root / cur_row["Patient"]
            stack = load_stack(patient_img_path)
            if self.transforms:
                stack = self.transforms(stack)
            pixel_array = stack.pixel_array
            metadata = stack.metadata
            tabular = self._get_concatenated(stack.metadata, tabular)

        record = Record(pixel_array, metadata, tabular)
        if self.train:
            y = cur_row[self.target]
            return record, y
        else:
            return record

    def _get_metadata_feature_keys(
        self, metadata: Optional[Mapping[str, Any]] = None
    ) -> List[str]:
        if metadata is None:
            ret = self.__getitem__(0)
            metadata = ret[0].metadata if self.train else ret.metadata

        keys = sorted(
            [
                k
                for k, v in metadata.items()
                if type(v) == float or type(v) == np.float64
            ]
        )
        return keys

    def _get_concatenated(
        self, metadata: Mapping[str, Any], tabular: np.ndarray
    ) -> np.ndarray:
        metadata_feature_keys = self._get_metadata_feature_keys(metadata)
        metadata_features = np.array([metadata[k] for k in metadata_feature_keys])
        return np.concatenate([tabular, metadata_features])

    def get_img_root(self, index: int):
        return self.source.roots

    def __len__(self) -> int:
        return len(self.source.df)

    @property
    def features(self) -> List[str]:
        return [*self._tabular_features, *self._get_metadata_feature_keys()]

    @classmethod
    def get_default_target_column(cls) -> str:
        return "FVC"

    @classmethod
    def get_default_feature_columns(cls, use_one_hot_encoding) -> List[str]:
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

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Any, Callable, Mapping
from functools import lru_cache
import imageio as io
import numpy as np
import torch
from torch.utils.data import Dataset
from pydicom import dcmread

from .datasource import DataSource


@dataclass
class Metadata:
    location: float
    pixel_spacing: np.ndarray
    image_type: Any


@dataclass
class Slice:
    pixel_array: np.ndarray
    metadata: Metadata


@dataclass
class Stack:
    pixel_array: np.ndarray
    metadata: Mapping[str, Any]


def _is_valid_slice(s: Slice) -> bool:
    image_type = s.metadata.image_type
    return image_type[0] == "ORIGINAL" and image_type[2] != "LOCALIZER"


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

        if target is None:
            self.target = self.get_default_target_column()
        else:
            self.target = target

        if features is None:
            self.features = self.get_default_feature_columns(use_one_hot_encoding)
        else:
            self.features = features

    def __getitem__(self, index: int) -> np.ndarray:
        tabular = np.array(
            self.source.df.iloc[index][self.features].values, dtype=np.float32
        )
        if self.train:
            y = self.source.df.iloc[index][self.target]
            return tabular, y
        else:
            return tabular

    def __len__(self) -> int:
        return len(self.source.df)

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


class CTDataset(Dataset):
    def __init__(
        self,
        source,
        train: bool = True,
        transforms: Optional[Callable] = None,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
        use_one_hot_encoding: bool = False,
    ) -> None:
        self.source = source
        self.train = train

        if transforms is not None:
            transforms = lru_cache(maxsize=2048)(transforms)
        self.transforms = transforms

        if target is None:
            target = TabularDataset.get_default_target_column()
        self.target = target

        if features is None:
            features = TabularDataset.get_default_feature_columns(use_one_hot_encoding)
        self.features = features

    def _ct_order(self, dcm):
        return dcm.ImagePositionPatient[-1]

    def _ct_rescale(self, pixel_array: np.ndarray, dcm):
        intercept = dcm.RescaleIntercept if hasattr(dcm, "RescaleIntercept") else 0
        slope = dcm.RescaleSlope if hasattr(dcm, "RescaleSlope") else 1
        return (pixel_array * slope + intercept).astype(np.int16)

    def _flip_if_need(self, pixel_array: np.ndarray, dcm):
        return np.flip(
            pixel_array,
            np.where(np.array(dcm.ImageOrientationPatient)[[0, 4]][::-1] < 0)[0],
        )

    def _load_dcm(self, dcm_path: Path):
        dcm = dcmread(dcm_path)
        metadata = Metadata(
            dcm.ImagePositionPatient[-1], dcm.PixelSpacing[0], dcm.ImageType,
        )
        pixel_array = dcm.pixel_array
        pixel_array = self._ct_rescale(pixel_array, dcm)
        pixel_array = self._flip_if_need(pixel_array, dcm)
        return Slice(pixel_array, metadata)

    def _sort_and_filter_slices(self, slices: List[Slice]) -> List[Slice]:
        slices = [s for s in slices if _is_valid_slice(s)]
        return sorted(slices, key=lambda s: s.metadata.location)

    def _validate(self, slices: List[Slice]) -> bool:
        if len(slices) == 0:
            raise ValueError("Input slices are empty")

        shapes = []
        spacings = []
        for s in slices:
            shapes.append(s.pixel_array.shape)
            spacings.append(s.metadata.pixel_spacing)
        if len(set(shapes)) != 1:
            raise ValueError("non-uniform shape slices")
        if len(set(spacings)) != 1:
            raise ValueError("non-uniform spacing slices")

    def _load_stack(self, root: Path):
        dcm_slices = [self._load_dcm(p) for p in root.glob("*.dcm")]
        dcm_slices = self._sort_and_filter_slices(dcm_slices)
        self._validate(dcm_slices)

        stacked_pixel_array = np.stack([s.pixel_array for s in dcm_slices], axis=0)
        stacked_locations = np.hstack([s.metadata.location for s in dcm_slices]).ravel()
        metadata = {
            "locations": stacked_locations,
            "pixel_spacing": dcm_slices[0].metadata.pixel_spacing,
            "image_type": dcm_slices[0].metadata.image_type,
        }
        return Stack(stacked_pixel_array, metadata)

    def __getitem__(self, index: int) -> np.ndarray:
        img_root = self.get_img_root(index)
        cur_row = self.source.df.iloc[index]

        patient_img_path = img_root / cur_row["Patient"]
        stack = self._load_stack(patient_img_path)
        if self.transforms:
            stack = self.transforms(stack)

        tabular = np.array(cur_row[self.features].values, dtype=np.float32)

        if self.train:
            y = cur_row[self.target]
            return (stack, tabular), y
        else:
            return (stack, tabular)

    def get_img_root(self, index: int):
        return self.source.roots

    def __len__(self) -> int:
        return len(self.source.df)

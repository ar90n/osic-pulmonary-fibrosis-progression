import os
import random
from itertools import tee
from typing import List, Tuple, Mapping
from collections import defaultdict

import numpy as np

from .dataset import Stack
from .lungmask import lung_segmentation


def _calc_tissue_ratio_by_pixels(pixels: np.ndarray) -> Mapping[str, float]:
    hu_ranges = {
        "air": (-1024, -950),
        "lung": (-850, -600),
        "muscle": (35, 55),
        "fat": (-120, -90),
    }

    counts = defaultdict(int)
    for cat, (min_hu, max_hu) in hu_ranges.items():
        cat_mask = np.logical_and(min_hu <= pixels, pixels < max_hu)
        counts[cat] += np.sum(cat_mask)
    counts["other"] = pixels.size - sum(counts.values())

    for k in counts.keys():
        counts[k] /= pixels.size
    return counts


class LungMask:
    def __init__(self):
        pass

    def __call__(self, stack: Stack):
        mask = lung_segmentation(stack.pixel_array)
        metadata = {**stack.metadata, "mask": mask}

        return Stack(stack.pixel_array, metadata)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class DropSlice:
    def __init__(self, interval: float = 10.0):
        self._interval = interval

    def __call__(self, stack: Stack):
        if "locations" not in stack.metadata:
            return stack
        locations = stack.metadata["locations"]

        use_indice = []
        prev_loc = -float("inf")
        for i, loc in enumerate(locations):
            if self._interval < (loc - prev_loc):
                use_indice.append(i)
                prev_loc = loc

        pixel_array = stack.pixel_array[use_indice]
        metadata = {**stack.metadata, "locations": locations[use_indice]}
        return Stack(pixel_array, metadata=metadata)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class DescribeVolume:
    def __init__(self):
        pass

    def __call__(self, stack: Stack):
        volume = self._calc_volume(stack)
        tissue_ratio = self._calc_tissue_ratio(stack)
        metadata = {**stack.metadata, **volume, **tissue_ratio}
        return Stack(stack.pixel_array, metadata=metadata)

    def _calc_volume(self, stack: Stack) -> Mapping[str, float]:
        mask = stack.metadata["mask"]
        locations = stack.metadata["locations"]
        pixel_spacing = stack.metadata["pixel_spacing"]

        lower_iter, upper_Iter = tee(zip(mask, locations))
        next(upper_Iter)

        unit_area = pixel_spacing * pixel_spacing

        right_volume = 0
        left_volume = 0
        for (upper_mask, upper_loc), (lower_mask, lower_loc) in zip(
            upper_Iter, lower_iter
        ):
            interval = upper_loc - lower_loc
            right_volume += np.sum(lower_mask == 1) * interval * unit_area
            left_volume += np.sum(lower_mask == 2) * interval * unit_area

        metadata = {
            "left_volume": left_volume,
            "right_volume": right_volume,
        }
        return metadata

    def _calc_tissue_ratio(self, stack: Stack) -> Mapping[str, float]:
        mask = stack.metadata["mask"]
        right_mask = mask == 1
        left_mask = mask == 2

        slices = stack.pixel_array
        right_lung_pixels = slices[right_mask]
        left_lung_pixels = slices[left_mask]

        right_counts = _calc_tissue_ratio_by_pixels(right_lung_pixels)
        left_counts = _calc_tissue_ratio_by_pixels(left_lung_pixels)

        return {
            **{f"right_{k}": v for k, v in right_counts.items()},
            **{f"left_{k}": v for k, v in right_counts.items()},
        }

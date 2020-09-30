import os
import random
from collections import defaultdict
from itertools import tee
from typing import Any, List, Mapping, Tuple

import numpy as np

from .dcm import Stack
from .lungmask import lung_segmentation


def _calc_tissue_ratio_by_pixels(
    pixels: np.ndarray, rescale_type: Any
) -> Mapping[str, float]:
    hu_ranges = {
        "air": (-1024, -950),
        "lung": (-850, -600),
        "muscle": (35, 55),
        "fat": (-120, -90),
    }

    if rescale_type != "HU":
        return {**{k: float("nan") for k in hu_ranges.keys()}, "other": float("nan")}

    counts = {**{k: 0 for k in hu_ranges.keys()}, "other": pixels.size}
    for cat, (min_hu, max_hu) in hu_ranges.items():
        cat_mask = np.logical_and(min_hu <= pixels, pixels < max_hu)
        cat_count = np.sum(cat_mask)
        counts[cat] += cat_count
        counts["other"] -= cat_count

    return {k: v / pixels.size for k, v in counts.items()}


class LungMask:
    def __init__(self):
        pass

    def __call__(self, stack: Stack):
        mask = lung_segmentation(stack.pixel_array)
        metadata = {**stack.metadata, "mask": mask}

        return Stack(stack.pixel_array, metadata=metadata)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class DropSlice:
    def __init__(self, interval: float = 10.0):
        self._interval = interval

    def _get_locations(self, stack: Stack) -> np.ndarray:
        if "locations" in stack.metadata:
            return stack.metadata["locations"]

        return np.array([])

    def __call__(self, stack: Stack):
        locations = self._get_locations(stack)
        # if locations.size == 0:
        #    return stack

        use_indice = []
        prev_loc = -float("inf")
        for i, loc in enumerate(locations):
            # print(self._interval < (loc - prev_loc))
            if loc is None or self._interval < (loc - prev_loc):
                use_indice.append(i)
                if loc is not None:
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
        pixel_spacings = stack.metadata["pixel_spacings"]

        lower_iter, upper_Iter = tee(zip(mask, locations, pixel_spacings))
        next(upper_Iter)

        right_volume = 0.0
        left_volume = 0.0
        for (upper_mask, upper_loc, _), (lower_mask, lower_loc, lower_spacing) in zip(
            upper_Iter, lower_iter
        ):
            if upper_loc is None or lower_loc is None:
                return {
                    "left_volume": float("nan"),
                    "right_volume": float("nan"),
                }

            interval = upper_loc - lower_loc
            unit_area = lower_spacing * lower_spacing
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
        # use dropped slices here
        right_lung_pixels = slices[right_mask]
        left_lung_pixels = slices[left_mask]

        rescale_type = stack.metadata["rescale_type"]
        right_counts = _calc_tissue_ratio_by_pixels(right_lung_pixels, rescale_type)
        left_counts = _calc_tissue_ratio_by_pixels(left_lung_pixels, rescale_type)

        return {
            **{f"right_{k}": v for k, v in right_counts.items()},
            **{f"left_{k}": v for k, v in right_counts.items()},
        }

    def __repr__(self):
        return f"{self.__class__.__name__}"

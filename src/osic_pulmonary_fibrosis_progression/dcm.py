import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, cast

import numpy as np
from pydicom import dcmread

from .util import cache


@dataclass
class Metadata:
    location: Optional[float]
    pixel_spacing: np.ndarray
    image_type: Any
    rescale_type: Any


@dataclass
class Slice:
    pixel_array: Optional[np.ndarray]
    metadata: Metadata


@dataclass
class Stack:
    pixel_array: np.ndarray
    metadata: Mapping[str, Any]


def _ct_rescale(pixel_array: np.ndarray, dcm):
    intercept = dcm.RescaleIntercept if hasattr(dcm, "RescaleIntercept") else 0
    slope = dcm.RescaleSlope if hasattr(dcm, "RescaleSlope") else 1
    return (pixel_array * slope + intercept).astype(np.int16)


def _flip_if_need(pixel_array: np.ndarray, dcm):
    orientation = getattr(dcm, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    return np.flip(pixel_array, np.where(np.array(orientation)[[0, 4]][::-1] < 0)[0])


def _get_location(dcm):
    if not hasattr(dcm, "ImagePositionPatient"):
        return None
    return dcm.ImagePositionPatient[-1]


def _is_valid_slice(s: Optional[Slice]) -> bool:
    if s is None:
        return False
    return True


def _sort_and_filter_slices(maybe_slices: List[Optional[Slice]]) -> List[Slice]:
    slices = [cast(Slice, s) for s in maybe_slices if _is_valid_slice(s)]

    locs = [s.metadata.location for s in slices if s.metadata.location is not None]
    if len(locs) == 0:
        return slices

    max_loc = max(locs)
    return sorted(
        slices,
        key=lambda s: max_loc if s.metadata.location is None else s.metadata.location,
    )


def _load_dcm(dcm_path: Path):
    dcm = dcmread(dcm_path)
    try:
        metadata = Metadata(
            _get_location(dcm),
            dcm.PixelSpacing[0],
            dcm.ImageType,
            getattr(dcm, "RescaleType", "HU"),
        )
    except Exception as e:
        print(f"{e.args} - {dcm_path}", file=sys.stderr)
        return None

    if metadata.location is None:
        print(f"location is missing - {dcm_path}", file=sys.stderr)
    if metadata.rescale_type != "HU":
        print(f"rescale_type is {metadata.rescale_type} - {dcm_path}", file=sys.stderr)

    print(dcm_path)
    if not hasattr(dcm, "pixel_array"):
        return None

    pixel_array = cast(np.ndarray, dcm.pixel_array)
    pixel_array = _ct_rescale(pixel_array, dcm)
    pixel_array = _flip_if_need(pixel_array, dcm)
    return Slice(pixel_array, metadata)


def _validate(slices: List[Slice]) -> None:
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


@cache
def load_stack(root: Path):
    dcm_slices = [_load_dcm(p) for p in root.glob("*.dcm")]
    dcm_slices = _sort_and_filter_slices(dcm_slices)
    try:
        _validate(dcm_slices)
    except ValueError as e:
        print(f"{e.args[0]} - {root}", file=sys.stderr)

    stacked_pixel_array = np.stack(
        [cast(np.ndarray, s.pixel_array) for s in dcm_slices], axis=0
    )
    stacked_locations = np.array([cast(float, s.metadata.location) for s in dcm_slices])
    stacked_pixel_spacings = np.hstack(
        [s.metadata.pixel_spacing for s in dcm_slices]
    ).ravel()
    metadata = {
        "locations": stacked_locations,
        "pixel_spacings": stacked_pixel_spacings,
        "image_type": dcm_slices[0].metadata.image_type,
        "rescale_type": dcm_slices[0].metadata.rescale_type,
    }
    stack = Stack(stacked_pixel_array, metadata)
    return stack

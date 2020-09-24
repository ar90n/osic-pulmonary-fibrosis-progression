import os
from pathlib import Path

from lungmask import mask
import SimpleITK as sitk
import numpy as np

_LUNGMASK_WEIGHTS_ROOT = Path(
    os.environ.get("LUNGMASK_WEIGHTS_ROOT", "../input/my-osic2020-data")
)
_R231_PTH_PATH = _LUNGMASK_WEIGHTS_ROOT / "unet_r231-d5d2fc3d.pth"

mask.model_urls[("unet", "R231")] = (str(_R231_PTH_PATH), 3)


def lung_segmentation(img: np.ndarray) -> np.ndarray:
    input_image = sitk.GetImageFromArray(img)
    input_image.SetDirection(np.eye(3).ravel())
    return mask.apply(input_image)

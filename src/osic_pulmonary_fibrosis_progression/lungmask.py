from lungmask import mask
import SimpleITK as sitk
import numpy as np

def lung_segmentation(img: np.ndarray) -> np.ndarray:
    input_image = sitk.GetImageFromArray(img)
    input_image.SetDirection(np.eye(3).ravel())
    return mask.apply(input_image) 

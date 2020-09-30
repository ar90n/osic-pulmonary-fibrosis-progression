import math
from functools import partial

import numpy as np
import scipy as sp
from tqdm import tqdm


def calc_laplace_log_likelihood(
    target: np.ndarray, confidence: np.ndarray, pred: np.ndarray
) -> float:
    if target.ndim != 1 or confidence.ndim != 1 or pred.ndim != 1:
        raise ValueError("Values have invalid shapes. All ndims must be 1.")

    sigma_clipped = np.clip(confidence, 70, None)
    delta = np.clip(np.abs(target - pred), None, 1000)
    score = -math.sqrt(2) * delta / sigma_clipped - np.log(math.sqrt(2) * sigma_clipped)
    return np.average(score)


def calc_best_confidence(target: np.ndarray, pred: np.ndarray):
    results = []
    for t, p in tqdm(zip(target, pred), total=len(target)):
        _loss = lambda x: -calc_laplace_log_likelihood(np.array([t]), x, np.array([p]))
        initial_confidence = np.array([100])
        result = sp.optimize.minimize(_loss, initial_confidence, method="SLSQP")
        results.append(result["x"][0])
    return np.array(results)

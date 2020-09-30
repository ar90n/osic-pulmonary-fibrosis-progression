import os
from dataclasses import dataclass
from distutils.util import strtobool
from typing import Optional


@dataclass
class Config:
    n_fold: int
    use_pseudo_baselines: bool
    ignore_bad_ct: bool


def get_config() -> Config:
    n_fold = int(os.environ.get("KAGGLE_N_FOLD", 4))
    use_pseudo_baselines = strtobool(
        os.environ.get("KAGGLE_USE_PSEUDO_BASELINES", "False")
    )
    ignore_bad_ct = strtobool(os.environ.get("KAGGLE_IGNORE_BAD_CT", "True"))
    return Config(
        n_fold=n_fold,
        use_pseudo_baselines=use_pseudo_baselines,
        ignore_bad_ct=ignore_bad_ct,
    )

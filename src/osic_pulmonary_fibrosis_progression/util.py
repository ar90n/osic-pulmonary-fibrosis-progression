import gc
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional

import torch
from pytorch_lightning import seed_everything as pl_seed_evertything

from .config import Config

try:
    import torch_xla
    import torch_xla.core.xla_model as xm

    has_xla = True
except ImportError:
    has_xla = False


def get_python_type():
    try:
        from IPython.core.getipython import get_ipython

        if "terminal" in get_ipython().__module__:
            return "ipython"
        else:
            return "jupyter"
    except (ImportError, NameError, AttributeError):
        return "python"


def seed_everything(seed: int = 47) -> None:
    pl_seed_evertything(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def initialize(config: Config) -> None:
    warnings.simplefilter("ignore")
    seed_everything()


def is_tpu_available() -> bool:
    return has_xla and ("TPU_NAME" in os.environ)


def is_kaggle() -> bool:
    return "KAGGLE_URL_BASE" in os.environ


def is_notebook() -> bool:
    return get_python_type() == "jupyter"


def get_device(n: Optional[int] = None):
    if is_tpu_available():
        return xm.xla_device(n=n, devkind="TPU")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_input() -> Path:
    input_path = os.environ.get("KAGGLE_INPUT_PATH")
    if input_path is not None:
        return Path(input_path)
    return Path.cwd().parent / "input"


def clean_up():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def to_device(tensors, device=None):
    if device is None:
        device = get_device()

    if isinstance(tensors, Iterable):
        ret = [t.to(device) for t in tensors]
        if isinstance(tensors, tuple):
            ret = tuple(ret)
        return ret

    return tensors.to(device)


def search_best_model_path(checkpoint_path: Optional[str]) -> Optional[str]:
    if checkpoint_path is not None and checkpoint_path != "":
        return checkpoint_path

    for p in Path.cwd().glob("*.ckpt"):
        return str(p)
    else:
        return None


def exit() -> None:
    if is_kaggle() and is_notebook():
        _ = [0] * 64 * 1000000000
    else:
        sys.exit(1)


def get_random_name(seed: Optional[int] = None) -> str:
    left_words = [
        "admiring",
        "adoring",
        "affectionate",
        "agitated",
        "amazing",
        "angry",
        "awesome",
        "beautiful",
        "blissful",
        "bold",
        "boring",
        "brave",
        "busy",
        "crazy",
    ]

    right_words = [
        "rabbit",
        "lion",
        "whale",
        "dog",
        "cat",
        "bird",
        "butterfly",
        "zebra",
    ]

    seed = random.randint(0, 1024) if seed is None else seed
    left_word = left_words[seed % len(left_words)]
    right_word = right_words[seed % len(right_words)]
    return f"{left_word}-{right_word}"


def get_osic_pulmonary_fibrosis_progression_root() -> Path:
    return get_input() / "osic-pulmonary-fibrosis-progression"

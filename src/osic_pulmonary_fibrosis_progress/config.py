import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    pass


def get_config() -> Config:
    return Config()

from ..config import Config
from ..util import get_random_name


def get_logger_name(config: Config) -> str:
    return get_random_name() if config.logger_name is None else config.logger_name

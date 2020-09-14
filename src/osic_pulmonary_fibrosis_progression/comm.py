from .util import is_tpu_available


def get_world_size() -> int:
    if not is_tpu_available():
        return 1

    # TODO; Use torch_xla
    return 8

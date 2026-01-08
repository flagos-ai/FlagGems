from functools import lru_cache

import torch

from flag_gems.runtime import torch_device_fn


@lru_cache(maxsize=1)
def get_device_id() -> int:
    try:
        return torch_device_fn.current_device()
    except Exception:
        return 0


@lru_cache(maxsize=1)
def get_device_properties():
    device_id = get_device_id()
    try:
        return torch_device_fn.get_device_properties(device_id)
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_device_capability() -> tuple[int, int]:
    device_id = get_device_id()
    try:
        return torch_device_fn.get_device_capability(device_id)
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(device_id)
    except Exception:
        pass
    return (0, 0)


@lru_cache(maxsize=1)
def get_device_info() -> tuple[int, int, int]:
    props = get_device_properties()
    l2_cache_size = None
    sm_count = None
    if props is not None:
        l2_cache_size = getattr(props, "L2_cache_size", None) or getattr(
            props, "l2_cache_size", None
        )
        sm_count = getattr(props, "multi_processor_count", None) or getattr(
            props, "multiProcessorCount", None
        )
    if l2_cache_size is None:
        # default L2 cache size to 40MB for A100
        l2_cache_size = 40 * 1024 * 1024
    if sm_count is None:
        # default sm_count to 108 for A100
        sm_count = 108
    return get_device_id(), l2_cache_size, sm_count


def get_l2_cache_size() -> int:
    return get_device_info()[1]


def get_sm_count() -> int:
    return get_device_info()[2]

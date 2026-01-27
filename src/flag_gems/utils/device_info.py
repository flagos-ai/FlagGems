from dataclasses import dataclass
from functools import lru_cache

import torch

from flag_gems.runtime import torch_device_fn


@dataclass(frozen=True)
class DeviceInfo:
    device_id: int
    l2_cache_size: int
    sm_count: int


@lru_cache(maxsize=1)
def get_device_id() -> int:
    try:
        return torch_device_fn.current_device()
    except Exception:
        import warnings

        warnings.warn(
            "[device_info] Failed to get current device, fallback to device_id=0."
        )
        return 0


@lru_cache(maxsize=1)
def get_device_properties():
    device_id = get_device_id()
    try:
        return torch_device_fn.get_device_properties(device_id)
    except Exception:
        import warnings

        warnings.warn(
            f"[device_info] Failed to get device properties for device_id={device_id}, fallback to None."
        )
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
    import warnings

    warnings.warn(
        f"[device_info] Failed to get device capability for device_id={device_id}, fallback to (0, 0)."
    )
    return (0, 0)


@lru_cache(maxsize=1)
def get_device_info() -> DeviceInfo:
    props = get_device_properties()
    l2_cache_size = None
    sm_count = None
    if props is not None:
        l2_cache_size = None
        if hasattr(props, "L2_cache_size"):
            l2_cache_size = props.L2_cache_size
        elif hasattr(props, "l2_cache_size"):
            l2_cache_size = props.l2_cache_size
        sm_count = getattr(props, "multi_processor_count", None) or getattr(
            props, "multiProcessorCount", None
        )
    import warnings

    if l2_cache_size is None:
        warnings.warn(
            "[device_info] Failed to get l2_cache_size, fallback to 40MB (A100 default)."
        )
        # default L2 cache size to 40MB for A100
        l2_cache_size = 40 * 1024 * 1024
    if sm_count is None:
        warnings.warn(
            "[device_info] Failed to get sm_count, fallback to 108 (A100 default)."
        )
        # default sm_count to 108 for A100
        sm_count = 108
    return DeviceInfo(
        device_id=get_device_id(),
        l2_cache_size=l2_cache_size,
        sm_count=sm_count,
    )


def get_l2_cache_size() -> int:
    return get_device_info().l2_cache_size


def get_sm_count() -> int:
    return get_device_info().sm_count

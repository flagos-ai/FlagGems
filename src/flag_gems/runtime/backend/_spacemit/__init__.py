import importlib.util
import os
from collections import namedtuple
from typing import Any

from backend_utils import VendorInfoBase  # noqa: E402

if (
    importlib.util.find_spec("triton.backends.spine_triton") is not None
    or importlib.util.find_spec("triton.backends.spacemit") is not None
):
    from .utils.config_pre_hook import setup_triton_config  # noqa: E402

    setup_triton_config()

    import triton  # noqa: E402

    try:
        from triton.backends.spine_triton.driver import CPUDriver  # noqa: E402
    except ImportError:
        from triton.backends.spacemit.driver import CPUDriver  # noqa: E402

    triton.runtime.driver.set_active(CPUDriver())  # noqa: E402


vendor_info = VendorInfoBase(
    vendor_name="spacemit",
    device_name="cpu",
    device_query_cmd="spacemit-tcm-smi",
)


class _DeviceGuard:
    def __init__(self, index: int):
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = self.idx

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = self.prev_idx
        return False


class _DeviceWrapper:
    def __init__(self, device: Any):
        ...

    def __enter__(self):
        ...

    def __exit__(self, type: Any, value: Any, traceback: Any):
        ...
        return False

    @staticmethod
    def current_device():
        """Return device index for kernel cache. CPU backend always uses device 0."""
        return 0

    @staticmethod
    def get_device_properties(device: Any = None):
        DeviceProperties = namedtuple(
            "DeviceProperties", ["multi_processor_count", "total_memory"]
        )
        return DeviceProperties(
            multi_processor_count=os.cpu_count() or 1,
            total_memory=64 * 1024**3,
        )


CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["*"]

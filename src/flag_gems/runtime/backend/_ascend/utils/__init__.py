import torch

CORE_NUM = 40

try:
    import triton.runtime.driver as driver

    CORE_NUM = driver.active.utils.get_device_properties(torch.npu.current_device())[
        "num_vectorcore"
    ]
except (ImportError, AttributeError, RuntimeError, KeyError):
    CORE_NUM = 40

__all__ = [
    "CORE_NUM",
]

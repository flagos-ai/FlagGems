import torch_ptpu  # noqa: F401
from backend_utils import VendorDescriptor

from .monkey_patch import apply_sunrise_monkey_patches

vendor_info = VendorDescriptor(
    vendor_name="sunrise",
    device_name="ptpu",
    device_query_cmd="pt_smi",
    triton_extra_name="tang",
    dispatch_key="PrivateUse1",
    fp64_enabled=False,
    bf16_enabled=False,
    int64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["*"]

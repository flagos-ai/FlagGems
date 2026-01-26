import torch
import torch_ptpu
from backend_utils import VendorInfoBase  # noqa: E402
import os
vendor_info = VendorInfoBase(
    vendor_name="sunrise",
    # device_name="cuda",
    device_name="ptpu",
    device_query_cmd="pt_smi",
    triton_extra_name="tang",
    dispatch_key="PrivateUse1",
)

CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["*"]

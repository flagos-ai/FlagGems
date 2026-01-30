import torch
try:
    from torch_gcu import transfer_to_gcu
except:
    print(f'torch_gcu not installed')

from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name="enflame",
    device_name="gcu",
    device_query_cmd="",
    dispatch_key="PrivateUse1",
)

CUSTOMIZED_UNUSED_OPS = (

)

__all__ = ["*"]

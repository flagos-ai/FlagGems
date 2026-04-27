from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name="amd",
    device_name="cuda",
    device_query_cmd="rocm-smi",
)

CUSTOMIZED_UNUSED_OPS = (
    "add",
    "cos",
    "cumsum",
)


__all__ = ["*"]

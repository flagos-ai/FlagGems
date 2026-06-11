from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
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

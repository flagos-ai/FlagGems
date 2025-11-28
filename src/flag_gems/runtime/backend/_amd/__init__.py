from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="amd", device_name="cuda", device_query_cmd="rocm-smi"
)

CUSTOMIZED_UNUSED_OPS = ("cumsum", "cos", "add")


__all__ = ["*"]

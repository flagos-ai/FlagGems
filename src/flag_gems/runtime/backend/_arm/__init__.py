from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="arm", device_name="cpu", device_query_cmd="cat /proc/cpuinfo"
)

CUSTOMIZED_UNUSED_OPS = ("cos", "add")

__all__ = ["*"]

from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="nvidia",
    device_name="cuda",
    device_query_cmd="",
    tle_enabled=True,
)

ARCH_MAP = {
    "9": "hopper",
    "8": "ampere",
}

CUSTOMIZED_UNUSED_OPS = (
    "add",
    "cos",
    "cumsum",
)

__all__ = ["*"]

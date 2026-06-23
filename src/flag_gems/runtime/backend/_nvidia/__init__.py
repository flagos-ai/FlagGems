from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="nvidia",
    device_name="cuda",
    device_query_cmd="",
    tle_enabled=True,
)

"""
Mapping from NVIDIA GPU compute capability major version
to architecture codename.

Example:
  8.x -> Ampere (A100)
  9.x -> Hopper (H100)
"""

ARCH_MAP = {
    "9": "hopper",
    "8": "ampere",
}


"""
Tuple of operation names to exclude,  empty tuple means all operations are enabled.

Example:
    CUSTOMIZED_UNUSED_OPS = ("add", "cos")
"""

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]

from backend_utils import VendorInfoBase  # noqa: E402
from triton.runtime import driver  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="aipu",
    device_name="aipu",
    device_query_cmd="aipu",
    dispatch_key="PrivateUse1",
)

# The aipu backend is loaded dynamically, so here need to active first.
driver.active.get_active_torch_device()

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]

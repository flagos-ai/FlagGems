from backend_utils import VendorDescriptor  # noqa: E402
from triton.runtime import driver  # noqa: E402

vendor_info = VendorDescriptor(
    vendor_name="aipu",
    device_name="aipu",
    device_query_cmd="aipu",
    dispatch_key="PrivateUse1",
    fp64_enabled=False,
    bf16_enabled=False,
    int64_enabled=False,
)

# The aipu backend is loaded dynamically, so here need to active first.
driver.active.get_active_torch_device()

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]

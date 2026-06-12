from flag_gems.runtime.backend.backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="iluvatar",
    device_name="cuda",
    device_query_cmd="ixsmi",
    fp64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]

from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="mthreads",
    device_name="musa",
    device_query_cmd="mthreads-gmi",
    fp64_enabled=False,
    tle_enabled=True,
)

CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["*"]

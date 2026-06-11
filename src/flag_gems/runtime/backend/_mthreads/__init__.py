from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="mthreads",
    device_name="musa",
    device_query_cmd="mthreads-gmi",
    fp64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["*"]

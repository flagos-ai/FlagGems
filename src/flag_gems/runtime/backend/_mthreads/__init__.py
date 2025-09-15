from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="mthreads", device_name="musa", device_query_cmd="mthreads-gmi"
)

CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["*"]

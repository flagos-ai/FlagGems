from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="iluvatar", device_name="cuda", device_query_cmd="ixsmi"
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]

from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name="metax", device_name="cuda", device_query_cmd="mx-smi"
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["vendor_info"]

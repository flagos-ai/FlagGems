from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="metax", device_name="cuda", device_query_cmd="mx-smi"
)

__all__ = ["vendor_info"]

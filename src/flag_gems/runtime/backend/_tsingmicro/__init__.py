from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="tsingmicro",
    device_name="txda",
    device_query_cmd="tsm_smi",
    dispatch_key="PrivateUse1",
)

CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["*"]

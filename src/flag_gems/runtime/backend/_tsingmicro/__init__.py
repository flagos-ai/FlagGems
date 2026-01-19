from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
<<<<<<< HEAD
    vendor_name="tsingmicro", 
    device_name="txda", 
    device_query_cmd="tsm_smi", 
=======
    vendor_name="tsingmicro",
    device_name="txda",
    device_query_cmd="tsm_smi",
>>>>>>> 055fd0b6 (add TSINGMICRO txda backend)
    dispatch_key="PrivateUse1",
)

CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["*"]

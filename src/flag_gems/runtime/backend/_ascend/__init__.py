from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="ascend",
    device_name="npu",
    triton_extra_name="ascend",
    device_query_cmd="npu-smi info",
    dispatch_key="PrivateUse1",
)

CUSTOMIZED_UNUSED_OPS = ("contiguous",)


__all__ = ["*"]

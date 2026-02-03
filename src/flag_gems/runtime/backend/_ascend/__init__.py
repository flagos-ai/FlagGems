from backend_utils import VendorInfoBase  # noqa: E402

def get_triton_extra_name():
    try:
        import triton
        from packaging import version

        if version.parse(triton.__version__) <= version.parse("3.2.0"):
            return "ascend"
        else:
            return "cann"
    except Exception:
        return "ascend"

vendor_info = VendorInfoBase(
    vendor_name="ascend",
    device_name="npu",
    device_query_cmd="npu-smi info",
    dispatch_key="PrivateUse1",
    triton_extra_name=get_triton_extra_name(),
)

CUSTOMIZED_UNUSED_OPS = (
    "contiguous",
    "sort",
    "sort_stable",
    "topk",
)


__all__ = ["*"]

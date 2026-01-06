from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="kunlunxin",
    device_name="cuda",
    device_query_cmd="xpu-smi",
    triton_extra_name="xpu",
)

CUSTOMIZED_UNUSED_OPS = (
    "cumsum",
    "unique",
    "randperm",
    "cummin",
    "topk",
    "sort",
)


__all__ = ["*"]

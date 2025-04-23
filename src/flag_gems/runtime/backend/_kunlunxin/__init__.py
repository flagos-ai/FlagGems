from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="kunlunxin", device_name="cuda", device_query_cmd="xpu-smi"
)

CUSTOMIZED_UNUSED_OPS = (
    "cumsum",
    "unique",
    "isin",
    "randperm",
    "cummin",
    "topk",
    "sort",
)


__all__ = ["*"]

from backend_utils import VendorDescriptor  # noqa: E402

vendor_info = VendorDescriptor(
    vendor_name="kunlunxin",
    device_name="cuda",
    device_query_cmd="xpu-smi",
    triton_extra_name="xpu",
    fp64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = (
    "cummin",
    "cumsum",
    "randperm",
    "sort",
    "topk",
    "unique",
)


__all__ = ["*"]

from flag_gems.runtime.backend.backend_utils import VendorDescriptor


def get_triton_extra_name():
    try:
        import triton
        from packaging import version

        if version.parse(triton.__version__) >= version.parse("3.6.0"):
            return "corex"
        return "cuda"
    except Exception:
        return "cuda"


vendor_info = VendorDescriptor(
    vendor_name="iluvatar",
    device_name="cuda",
    device_query_cmd="ixsmi",
    triton_extra_name=get_triton_extra_name(),
    fp64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]

from backend_utils import VendorInfoBase  # noqa: E402

from .heuristics_config_utils import HEURISTICS_CONFIGS

global specific_ops, unused_ops
vendor_info = VendorInfoBase(
    vendor_name="nvidia", device_name="cuda", device_query_cmd="nvidia-smi"
)


def OpLoader():
    global specific_ops, unused_ops
    if specific_ops is None:
        from . import ops  # noqa: F403

        specific_ops = ops.get_specific_ops()
        unused_ops = ops.get_unused_ops()


__all__ = ["HEURISTICS_CONFIGS", "vendor_info", "OpLoader"]

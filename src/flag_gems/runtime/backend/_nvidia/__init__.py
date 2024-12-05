from backend_utils import VendorInfoBase  # noqa: E402

from .ops import *  # noqa: F403

vendor_info = VendorInfoBase(
    vendor_name="nvidia", device_name="cuda", device_query_cmd="nvidia-smi"
)


def get_register_ops():
    return (("add.Tensor", add, False),)


def get_unused_op():
    return ("cumsum", "cos")


__all__ = ["*"]

from backend_utils import VendorInfoBase  # noqa: E402

from .heuristics_config_utils import HEURISTICS_CONFIGS

global specific_ops, unused_ops
specific_ops = None
unused_ops = None
vendor_info = VendorInfoBase(
    vendor_name="tsingmicro", 
    device_name="txda",
    device_query_cmd="ls",
    dispatch_key="PrivateUse1",
)


CUSTOMIZED_UNUSED_OPS = (
    #"randperm",  # skip now
    #"sort",  # skip now
    #"multinomial",  # skip now
    #"_upsample_bicubic2d_aa",  # skip now
    #"batch_norm",  #
    #"pad",
    #"constant_pad_nd",  #
)

def OpLoader():
    global specific_ops, unused_ops
    if specific_ops is None:
        from . import ops  # noqa: F403

        specific_ops = ops.get_specific_ops()
        unused_ops = ops.get_unused_ops()


__all__ = ["*"]

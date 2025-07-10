import torch  # noqa: F401
import torch_mlu  # noqa: F401

from flag_gems.runtime.backend.backend_utils import VendorInfoBase  # noqa: E402

from .utils import DEVICE_COUNT  # noqa: F401
from .utils import MAX_GRID_SIZE_X  # noqa: F401
from .utils import MAX_GRID_SIZE_Y  # noqa: F401
from .utils import MAX_GRID_SIZE_Z  # noqa: F401
from .utils import MAX_GRID_SIZES  # noqa: F401
from .utils import MAX_NRAM_SIZE  # noqa: F401
from .utils import TOTAL_CLUSTER_NUM  # noqa: F401
from .utils import TOTAL_CORE_NUM  # noqa: F401

try:
    from torch_mlu.utils.model_transfer import transfer  # noqa: F401
except ImportError:
    pass

vendor_info = VendorInfoBase(
    vendor_name="cambricon",
    device_name="mlu",
    device_query_cmd="cnmon",
    dispatch_key="PrivateUse1",
)

CUSTOMIZED_UNUSED_OPS = (
    "randperm",  # skip now
    "sort",  # skip now
    "multinomial",  # skip now
    "_upsample_bicubic2d_aa",  # skip now
    "sort_stable",
)

__all__ = ["*"]

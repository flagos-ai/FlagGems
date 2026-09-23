import logging

import torch
import triton
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic
from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

_SMALL_NUMEL = 200000  # below this the single-pass reversed flip beats two-pass

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def _rot90_copy_pw(src):
    return src


def rot90(input, k=1, dims=[0, 1]):
    logger.debug("GEMS_KUNLUNXIN ROT90")
    x = input
    if not x.is_contiguous():
        x = x.contiguous()

    dim0, dim1 = dims[0], dims[1]
    k_norm = ((k % 4) + 4) % 4

    if k_norm == 0:
        return x.clone()
    if k_norm == 1:
        if x.numel() <= _SMALL_NUMEL:
            return x.flip([dim1]).transpose(dim0, dim1)
        out_shape = list(x.shape)
        out_shape[dim0], out_shape[dim1] = out_shape[dim1], out_shape[dim0]
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        # Materialize the transposed view with the copy-family recipe (same as
        # permute_copy): tle takes the whole transfer (a 64x64 on-chip transpose
        # for the 2D case), the pointwise kernel keeps the rest. No
        # `torch.ops.aten._copy_from` -- it dispatches to the XPU fallback.
        transposed = x.transpose(dim0, dim1)
        if not tle_copy(transposed, out):
            _rot90_copy_pw(transposed, out0=out)
        return out.flip([dim0])
    if k_norm == 2:
        return x.flip([dim0, dim1])
    return x.flip([dim0]).transpose(dim0, dim1)

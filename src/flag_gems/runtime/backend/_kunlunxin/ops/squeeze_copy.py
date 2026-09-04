import logging

import torch
import triton
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def _squeeze_copy_flat(src):
    return src


def squeeze_copy(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SQUEEZE_COPY")
    squeezed_shape = tuple(s for s in x.shape if s != 1)
    out = torch.empty(squeezed_shape, dtype=x.dtype, device=x.device, layout=x.layout)
    if out.numel() == 0:
        return out

    src = x.contiguous() if not x.is_contiguous() else x
    _squeeze_copy_flat(src.view(-1), out0=out.view(-1))
    return out

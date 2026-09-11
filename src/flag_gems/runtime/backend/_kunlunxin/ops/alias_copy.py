import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

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
    kunlunAutoGrid=True,
    unroll_num=8,
)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _alias_copy_out_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets, mask=mask), mask=mask)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def alias_copy_func(x):
    return x


def alias_copy(x: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN ALIAS_COPY")
    if x.numel() == 0:
        return torch.empty_like(x)
    return alias_copy_func(x)


def alias_copy_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN ALIAS_COPY_OUT")
    if x.dtype != out.dtype:
        raise RuntimeError("alias_copy_out: dtype of input and output must match.")
    if x.numel() != out.numel():
        raise RuntimeError(
            "alias_copy_out: input and output must have the same number of elements."
        )
    if x.device != out.device:
        raise RuntimeError(
            "alias_copy_out: input and output must be on the same device."
        )
    if out.numel() == 0:
        return out
    if x.is_contiguous() and out.is_contiguous():
        block_size = 8192
        grid = triton.cdiv(x.numel(), block_size)
        with torch_device_fn.device(x.device):
            _alias_copy_out_kernel[(grid,)](
                x, out, x.numel(), BLOCK_SIZE=block_size, num_warps=4
            )
    else:
        alias_copy_func(x, out0=out)
    return out

import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.shape_utils import heuristics_for_num_warps, volume

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseDtypeConvert=True,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, "DEFAULT")],
    num_outputs=1,
    config=config_,
)
@triton.jit
def fill_scalar_func(inp, value_scalar):
    return tl.full(inp.shape, value_scalar, dtype=inp.dtype)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, "DEFAULT")],
    num_outputs=1,
    config=config_,
)
@triton.jit
def fill_tensor_func(inp, value):
    return value


def fill_scalar(input, value):
    logger.debug("GEMS_KUNLUNXIN FILL")
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_scalar_func(input, value, out0=out)


def fill_scalar_out(input, value, *, out=None):
    logger.debug("GEMS_KUNLUNXIN FILL_SCALAR_OUT")
    if out is None:
        return fill_scalar(input, value)
    with torch_device_fn.device(input.device):
        fill_scalar_func(input, value, out0=out)
    return out


def fill_tensor(input, value):
    if not value.is_cuda:
        return fill_scalar(input, value.item())
    logger.debug("GEMS_KUNLUNXIN FILL")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_tensor_func(input, value, out0=out)


@triton.jit
def _fill_tensor_out_kernel(out_ptr, n_elements, value, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(
        out_ptr + offs,
        tl.full([BLOCK_SIZE], value, dtype=out_ptr.dtype.element_ty),
        mask=mask,
    )


def fill_tensor_out(input, value, *, out=None):
    logger.debug("GEMS_KUNLUNXIN FILL_TENSOR_OUT")
    if out is None:
        return fill_tensor(input, value)
    if value.is_cuda and value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    N = volume(input.shape)
    if N == 0:
        return out
    grid_fn = (12, 1, 1)
    block_size = triton.next_power_of_2(triton.cdiv(N, 12))
    num_warps = heuristics_for_num_warps(block_size)
    with torch_device_fn.device(input.device):
        _fill_tensor_out_kernel[grid_fn](
            out,
            N,
            value.item(),
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            isCloseDtypeConvert=True,
        )
    return out


def fill_tensor_(self, value):
    if not value.is_cuda:
        return fill_scalar_(self, value.item())
    logger.debug("GEMS_KUNLUNXIN FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    with torch_device_fn.device(self.device):
        fill_tensor_func(self, value, out0=self)
    return self


def fill_scalar_(self, value):
    logger.debug("GEMS_KUNLUNXIN FILL_SCALAR_")
    with torch_device_fn.device(self.device):
        fill_scalar_func(self, value, out0=self)
    return self

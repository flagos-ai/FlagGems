import logging

import torch
import triton
import triton.language as tl
from triton.language.extra.mlu.libdevice import fast_erf, fast_tanh

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner, tl_extra_shim

from ..utils import TOTAL_CORE_NUM
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

exp = tl_extra_shim.exp


@libentry()
@libtuner(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 65536}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 131072}, num_stages=1, num_warps=1),
    ],
    key=["n_elements"],
)
@triton.jit
def gelu_none_kernel(X_ptr, OUT_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    block_start = block_start.to(tl.int64)
    scale: tl.constexpr = 0.7071067811
    for off in range(block_start, n_elements, step):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(X_ptr + offsets, mask=mask)
        x_f32 = x.to(tl.float32)
        result = 0.5 * x_f32 + 0.5 * x_f32 * fast_erf(x_f32 * scale)
        tl.store(OUT_ptr + offsets, result.to(x.dtype), mask=mask)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_tanh(x, inplace):
    x_f32 = x.to(tl.float32)
    output = 0.5 * x_f32 + 0.5 * x_f32 * fast_tanh(
        x_f32 * 0.79788456 + x_f32 * 0.79788456 * 0.044715 * x_f32 * x_f32
    )
    return output


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_none(x, dy):
    scale1: tl.constexpr = 0.7071067811
    scale2: tl.constexpr = 0.3989422803
    x_fp32 = x.to(tl.float32)
    x_sqrt = scale1 * x_fp32
    dydx = scale2 * x_fp32 * exp(-x_sqrt * x_sqrt) + 0.5 * fast_erf(x_sqrt) + 0.5
    dx = dydx * dy
    return dx


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_tanh(x, dy):
    x_fp32 = x.to(tl.float32)
    c1 = 0.79788456
    c2 = 0.044715
    tanh_out = fast_tanh(c1 * x_fp32 + c1 * x_fp32 * c2 * x_fp32 * x_fp32)
    dydx = (
        0.5 * ((x - x * tanh_out * tanh_out) * (c1 + 0.1070322243 * x_fp32 * x_fp32))
        + 0.5
        + 0.5 * tanh_out
    )
    dx = dydx * dy
    return dx


def gelu(self, *, approximate="none"):
    logger.debug("GEMS_CAMBRICON GELU FORWARD")
    if approximate == "tanh":
        return gelu_tanh(self, False)
    else:
        A = self.contiguous()
        out = torch.empty_like(A)
        N = A.numel()
        if N == 0:
            return out
        grid_fn = lambda meta: (
            min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
        )
        with torch_device_fn.device(A.device):
            gelu_none_kernel[grid_fn](A, out, N)
        return out


def gelu_backward(grad_output, self, *, approximate="none"):
    logger.debug("GEMS_CAMBRICON GELU BACKWARD")
    if approximate == "tanh":
        return gelu_backward_tanh(self, grad_output)
    else:
        return gelu_backward_none(self, grad_output)


def gelu_(A, *, approximate="none"):
    logger.debug("GEMS_CAMBRICON GELU_ FORWARD")
    if approximate == "tanh":
        return gelu_tanh(A, True, out0=A)
    else:
        A_contig = A.contiguous()
        N = A_contig.numel()
        if N == 0:
            return A
        grid_fn = lambda meta: (
            min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
        )
        with torch_device_fn.device(A.device):
            gelu_none_kernel[grid_fn](A_contig, A_contig, N)
        if not A.is_contiguous():
            A.copy_(A_contig)
        return A

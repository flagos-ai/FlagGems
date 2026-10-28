import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=4,
)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def threshold_kernel(self, threshold, value):
    if self.dtype == tl.float16:
        big = tl.full((), 1.0e30, dtype=self.dtype)
        d = (self - threshold) * big
        m = tl.minimum(1.0, tl.maximum(0.0, d))
        return self * m + value * (1.0 - m)
    d = (self - threshold) * 1.0e30
    m = tl.minimum(1.0, tl.maximum(0.0, d))
    return value + (self - value) * m


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def threshold_backward_kernel(grad_output, self, threshold):
    # grad_input = grad_output * (self > threshold)
    # Avoid tensor-vs-scalar compare (B1 gets scalarized into a per-lane branch,
    # extremely slow); instead build a 0/1 mask via saturating clamp so it maps
    # to 512-bit vnminf/vnmaxf.
    # (self - threshold) > 0 -> *1e30 saturates to >=1 -> min(1,max(0,..))=1; else 0.
    go = grad_output.to(tl.float32)
    s = self.to(tl.float32)
    m = tl.minimum(1.0, tl.maximum(0.0, (s - threshold) * 1.0e30))
    return (go * m).to(grad_output.dtype)


_THRESHOLD_BWD_BLOCK = 16384
_THRESHOLD_BWD_BLOCK_SMALL = 8192
_THRESHOLD_BWD_WARPS = 1
_TB_TILE = 65536


@triton.jit
def _threshold_bwd_grid_kernel(
    grad,
    self,
    out,
    n_elements,
    threshold,
    TILE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * TILE + tl.arange(0, TILE)
    if NEED_MASK:
        m = offs < n_elements
        go = tl.load(grad + offs, mask=m).to(tl.float32)
        s = tl.load(self + offs, mask=m).to(tl.float32)
        msk = tl.minimum(1.0, tl.maximum(0.0, (s - threshold) * 1.0e30))
        tl.store(out + offs, go * msk, mask=m)
    else:
        go = tl.load(grad + offs).to(tl.float32)
        s = tl.load(self + offs).to(tl.float32)
        msk = tl.minimum(1.0, tl.maximum(0.0, (s - threshold) * 1.0e30))
        tl.store(out + offs, go * msk)


@triton.jit
def _threshold_backward_bits_kernel(
    grad,
    self,
    out,
    n_elements,
    threshold_bits,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        m = offs < n_elements
        x = tl.load(grad + offs, mask=m)
        y = tl.load(self + offs, mask=m)
        yb = y.to(tl.float32).to(tl.uint32, bitcast=True)
        keep = (yb > threshold_bits) & (yb < 0x7F800000)
        tl.store(out + offs, x * keep.to(x.dtype), mask=m)
    else:
        x = tl.load(grad + offs)
        y = tl.load(self + offs)
        yb = y.to(tl.float32).to(tl.uint32, bitcast=True)
        keep = (yb > threshold_bits) & (yb < 0x7F800000)
        tl.store(out + offs, x * keep.to(x.dtype))


def threshold(self, threshold, value):
    logger.debug("GEMS_KUNLUNXIN THRESHOLD")
    output = threshold_kernel(self, threshold, value)
    return output


def threshold_(self, threshold, value):
    logger.debug("GEMS_KUNLUNXIN THRESHOLD_")
    threshold_kernel(self, threshold, value, out0=self)
    return self


_TB_SMALL_NUMEL = 1 << 20


def threshold_backward(grad_output, self, threshold):
    logger.debug("GEMS_KUNLUNXIN THRESHOLD_BACKWARD")
    n = grad_output.numel()
    if 0 < n <= _TB_SMALL_NUMEL:
        go = grad_output.contiguous()
        s = self.contiguous()
        out_dtype = torch.result_type(grad_output, self)
        grad_input = torch.empty_like(go, dtype=out_dtype)
        grid = (triton.cdiv(n, _TB_TILE),)
        need_mask = (n % _TB_TILE) != 0
        _threshold_bwd_grid_kernel[grid](
            go, s, grad_input, n, threshold, _TB_TILE, need_mask, num_warps=4
        )
        return grad_input
    return threshold_backward_kernel(grad_output, self, threshold)
